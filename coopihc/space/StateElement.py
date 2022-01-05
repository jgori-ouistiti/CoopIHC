import copy
import numpy
import json
import itertools
import warnings

from coopihc.helpers import flatten
from coopihc.space.Space import Space
from coopihc.space.utils import (
    SpaceLengthError,
    StateNotContainedError,
    StateNotContainedWarning,
    NumpyFunctionNotHandledWarning,
    RedefiningHandledFunctionWarning,
)

# Read link below to understand how to subclass numpy.ndarray
# https://numpy.org/doc/stable/user/basics.subclassing.html


# To do: explain what happens with ufunc when output is not in the space and shouldn't be
# How to mix out_of_bounds_mode using precedence + explain
# Test array_func


class StateElement(numpy.ndarray):
    """StateElement

    An Element of a State. This is basically a version of numpy ndarrays where values are associated to a space.


    .. code-block:: python

        # Discrete
        discr_space = discrete_space([1, 2, 3])
        x = StateElement([2], discr_space, out_of_bounds_mode="error")
        # Continuous
        cont_space = continuous_space(-numpy.ones((2, 2)), numpy.ones((2, 2)))
        x = StateElement(
            numpy.array([[0, 0], [0, 0]]), cont_space, out_of_bounds_mode="error"
        )
        # Multidiscrete
        multidiscr_space = multidiscrete_space(
            [
                numpy.array([1, 2, 3]),
                numpy.array([1, 2, 3, 4, 5]),
                numpy.array([1, 3, 5, 8]),
            ]
        )
        x = StateElement([1, 5], multidiscr_space, out_of_bounds_mode="error")


    :param input_array: value array-like
    :type input_array: numpy array-like
    :param spaces: Spaces where input_array takes value
    :type spaces: `Space<coopihc.space.Space.Space>`
    :param out_of_bounds_mode: what to do when the value is outside the bound, defaults to "warning". Possible values are

    * "error" --> raises a StateNotContainedError
    * "warning" --> raises a StateNotContainedWarning
    * "clip" --> clips the data to force it to belong to the space
    * "silent" --> Values not in the space are accepted silently (behavior should become equivalent to numpy.ndarray). Broadcasting and type casting may still be applied
    * "raw" --> No data transformation is applied. This is faster than the other options, because the preprocessing of input data is short-circuited. However, this provides no tolerance for misspecified input.

    :type out_of_bounds_mode: str, optional

    """

    __precedence__ = {"error": 0, "warning": 2, "clip": 1, "silent": 3, "raw": 4}
    HANDLED_FUNCTIONS = {}
    SAFE_FUNCTIONS = ["all"]

    @staticmethod
    def _clip(value, space):
        """Simple wrapper for numpy clip"""
        if value not in space:
            return numpy.clip(value, space.low, space.high)

    def __new__(
        cls,
        input_array,
        spaces,
        *args,
        out_of_bounds_mode="warning",
        **kwargs,
    ):
        """__new__, see https://numpy.org/doc/stable/user/basics.subclassing.html"""
        input_array = numpy.asarray(
            StateElement._process_input_values(
                input_array,
                spaces,
                out_of_bounds_mode,
            )
        )
        obj = input_array.view(cls)
        obj.spaces = spaces
        obj.out_of_bounds_mode = out_of_bounds_mode
        obj.kwargs = kwargs
        return obj

    def __array_finalize__(self, obj):
        """__array_finalize__, see https://numpy.org/doc/stable/user/basics.subclassing.html"""
        if obj is None:
            return
        spaces = getattr(obj, "spaces", None)
        out_of_bounds_mode = getattr(obj, "out_of_bounds_mode", None)
        self.spaces = spaces
        self.out_of_bounds_mode = out_of_bounds_mode
        self.kwargs = getattr(obj, "kwargs", {})

    def __array_ufunc__(self, ufunc, method, *input_args, out=None, **kwargs):
        """__array_ufunc__, see https://numpy.org/doc/stable/user/basics.subclassing.html.


        This can lead to some issues with some numpy universal functions. For example, modf returns the fractional and integral part of an array in the form of two new StateElements. In that case, the fractional part is necessarily in ]0,1[, whatever the actual space of the original StateElement, but the former will receive the latter's space. To deal with that case, it is suggested to select a proper "out_of_bounds_mode", perhaps dynamically, and to change the space attribute of the new object afterwards if actually needed.

        """
        args = []
        argmode = "raw"
        # Input and Output conversion to numpy
        for _input in input_args:
            if isinstance(_input, StateElement):
                args.append(_input.view(numpy.ndarray))
                if (
                    StateElement.__precedence__[_input.out_of_bounds_mode]
                    < StateElement.__precedence__[argmode]
                ):
                    argmode = _input.out_of_bounds_mode
            else:
                args.append(_input)

        outputs = out
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, StateElement):
                    out_args.append(output.view(numpy.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        # Actually apply the ufunc to numpy array
        result = getattr(ufunc, method)(*args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        # Back conversion to StateElement. Only pass results in this who need to be processed (types that subclass numpy.number, e.g. exclude booleans)
        if isinstance(result, (numpy.ndarray, StateElement)):
            if issubclass(result.dtype.type, numpy.number):
                result = StateElement._process_input_values(
                    result, self.spaces, self.out_of_bounds_mode
                )

        # In place
        if method == "at":
            if isinstance(input_args[0], StateElement):
                input_args[0].spaces = self.spaces
                input_args[0].out_of_bounds_mode = argmode
                input_args[0].kwargs = self.kwargs

        if ufunc.nout == 1:
            result = (result,)

        result = tuple(
            (numpy.asarray(_result).view(StateElement) if output is None else output)
            for _result, output in zip(result, outputs)
        )
        if result and isinstance(result[0], StateElement):
            result[0].spaces = self.spaces
            result[0].out_of_bounds_mode = argmode
            result[0].kwargs = kwargs

        return result[0] if len(result) == 1 else result

    def __array_function__(self, func, types, args, kwargs):
        """__array_function__ [summary]

        If func is not know to be handled by StateElement, then pass it to Numpy. In that case, a numpy array is returned.
        If an implementation for func has been provided that is specific to StateElement via the ``implements`` decorator, then call that one.

        See `this NEP<https://numpy.org/neps/nep-0018-array-function-protocol.html>`_ as well as `this Numpy doc<https://numpy.org/doc/stable/user/basics.dispatch.html>`_ ifor more details on how to implement __array_function__.


        """
        # Calls default numpy implementations and returns a numpy ndarray
        if func not in self.HANDLED_FUNCTIONS:
            if func.__name__ not in self.SAFE_FUNCTIONS:
                warnings.warn(
                    NumpyFunctionNotHandledWarning(
                        "Numpy function with name {} is currently not implemented for this object with type {}, and CoopIHC is returning a numpy.ndarray object. If you want to have a StateElement object returned, consider implementing your own version of this function and using the implements decorator (example in the decorator's documentation) to add it to the StateElement, as well as formulating a PR to have it included in CoopIHC core code.".format(
                            func.__name__, type(self)
                        )
                    )
                )
            return (super().__array_function__(func, types, args, kwargs)).view(
                numpy.ndarray
            )
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, StateElement) for t in types):
            return NotImplemented
        return self.HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __iter__(self):
        self.n = 0
        self.max = len(self.spaces)
        self._iterable_space = iter(self.spaces)
        self._iterable_value = iter(self.view(numpy.ndarray))
        return self

    def __next__(self):
        if self.n < self.max:
            result = StateElement(
                self._iterable_value.__next__(),
                self._iterable_space.__next__(),
                out_of_bounds_mode=self.out_of_bounds_mode,
            )
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, key):
        """__getitem__

        Includes an extra mechanism, to automatically extract values with the corresponding spaces, which slightly abuses the slice and indexing notations:

        .. code-block:: python

        global cont_space
        x = StateElement(numpy.array([[0.0, 0.1], [0.2, 0.3]]), cont_space)
        assert x[0, 0] == 0.0
        assert x[0, 0, {"spaces": True}] == StateElement(
            numpy.array([[0.0]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
        )



        :param key: [description]
        :type key: [type]
        :raises NotImplementedError: [description]
        :return: [description]
        :rtype: [type]
        """
        if isinstance(key, tuple) and key[-1] == {"spaces": True}:
            key = key[:-1]
            item = super().__getitem__(key)
            try:
                spaces = self.spaces[key]
            except TypeError:  # if discrete space
                spaces = self.spaces
            return StateElement(
                item.view(numpy.ndarray).ravel(),
                spaces,
                out_of_bounds_mode=self.out_of_bounds_mode,
                **self.kwargs,
            )
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """__setitem__

        Simply calles numpy's __setitem__ after having checked input values.

        :param key: [description]
        :type key: [type]
        :param value: [description]
        :type value: [type]
        """
        value = StateElement._process_input_values(
            value, self.spaces, self.out_of_bounds_mode
        )
        super().__setitem__(key, value)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "StateElement({}, {}, out_of_bounds_mode = '{}')".format(
            numpy.ndarray.__repr__(self.view(numpy.ndarray)),
            self.spaces.__repr__(),
            self.out_of_bounds_mode,
        )

    def reset(self, value=None):
        """reset

        Reset the StateElement to a random value, by sampling the underlying space.

        .. code-block:: python

            x = StateElement(numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error")
            for i in range(1000):
                x.reset()

            # Forced reset
            x.reset(0.59 * numpy.ones((2, 2)))
            assert (
                x
                == StateElement(
                    0.59 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error"
                )
            ).all()

        :param value: reset value for forced reset, defaults to None
        :type value: numpy.ndarray, optional
        """
        if value is None:
            self[:] = self.spaces.sample()
        else:
            # self[:] = StateElement._process_input_values(
            #     value, self.spaces, self.out_of_bounds_mode
            # )
            self[:] = value

    def serialize(self):
        """Generate a JSON representation of StateElement.

        .. code-block:: python

            x = StateElement(numpy.array([2]), discr_space)
            assert x.serialize() == {
                "values": [2],
                "spaces": {
                    "array_list": [1, 2, 3],
                    "space_type": "discrete",
                    "seed": None,
                    "contains": "soft",
                },
            }

        :return: JSON blob
        :rtype: dictionnary

        """

        return {
            "values": self.tolist(),
            "spaces": self.spaces.serialize(),
        }

    def equals(self, other, mode="soft"):
        """equals

        Returns False if other is not equal to self.

        Soft mode is currently equivalent to __eq__ inherited from numpy.ndarray.
        In Hard mode, contrary to __eq__, the spaces and out of bounds mode are also compared.

        .. code-block:: python


            discr_space = discrete_space([1, 2, 3])
            new_discr_space = discrete_space([1, 2, 3, 4])
            x = StateElement(numpy.array(1), discr_space)
            y = StateElement(numpy.array(1), discr_space)
            assert x.equals(y)
            assert x.equals(y, mode="hard")
            z = StateElement(numpy.array(2), discr_space)
            assert not x.equals(z)
            w = StateElement(numpy.array(1), new_discr_space)
            assert w.equals(x)
            assert not w.equals(x, "hard")

        :param other: object to compare to
        :type other: StateElement, numpy.ndarray
        :param mode: [description], defaults to "soft"
        :type mode: str, optional
        :return: [description]
        :rtype: [type]
        """
        result = self == other
        if mode == "soft":
            return result
        if not isinstance(other, StateElement):
            return False
        if self.spaces != other.spaces:
            return numpy.full(self.shape, False)
        if self.out_of_bounds_mode != other.out_of_bounds_mode:
            return numpy.full(self.shape, False)
        return numpy.full(self.shape, True)

    def cast(self, other, mode="center"):
        """cast

        Cast value from one space to another. Not implemented yet, old code in comment.
        """
        raise NotImplementedError

    def _tabulate(self):
        """_tabulate

        outputs a list ready for tabulate.tabulate(), as well as the number of lines of the generated table.

        Examples:

        .. code-block::
            x = StateElement(1, discr_space)
            print(x._tabulate())
            >>> ([[array([1], dtype=int16), 'Discr(3)']], 1)

            cont_space = autospace(-numpy.ones((3, 3)), numpy.ones((3, 3)))
            x = StateElement(numpy.zeros((3, 3)), cont_space)
            print(x._tabulate())
            >>> ([[array([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]], dtype=float32), '\nCont(3, 3)\n']], 3)

            x = StateElement(
                numpy.array([[1], [1], [8]]), multidiscr_space, out_of_bounds_mode="error"
            )
            print(x._tabulate())
            >>> ([[array([[1],
                    [1],
                    [8]], dtype=int16), '\nMultiDiscr[2, 5, 4]\n']], 3)


        :return: list ready for tabulate.tabulate(), line numbers
        :rtype: tuple(list, int)
        """
        try:
            space = ["" for i in range(self.shape[0])]
        except IndexError:  # for shape (X,)
            space = [""]
        _index = int(len(space) / 2)
        if self.spaces.space_type == "continuous":
            space[_index] = "Cont{}".format(self.spaces.shape)
            array = self.view(numpy.ndarray)
        elif self.spaces.space_type == "discrete":
            space[_index] = "Discr({})".format(self.spaces.N)
            array = self.view(numpy.ndarray)[0]
        elif self.spaces.space_type == "multidiscrete":
            space[_index] = "MultiDiscr{}".format(self.spaces.N)
            array = self.view(numpy.ndarray).reshape(-1)
        else:
            raise NotImplementedError

        return ([[array, "\n".join(space)]], self.shape[0])

    # def _discrete2continuous(self, other, mode="center"):
    #         values = []
    #         for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
    #             if not (not ss.continuous and os.continuous):
    #                 raise AttributeError(
    #                     "Use this only to go from a discrete to a continuous space"
    #                 )

    #             _range = ss.range[0].squeeze()

    #             if mode == "edges":
    #                 N = len(_range)
    #                 ls = numpy.linspace(os.low, os.high, N)
    #                 shift = 0
    #             elif mode == "center":
    #                 N = len(_range) + 1
    #                 ls = numpy.linspace(os.low, os.high, N)
    #                 shift = (ls[1] - ls[0]) / 2

    #             values.append(ls[list(_range).index(sv)] + shift)
    #         return values

    #     def _continuous2discrete(self, other, mode="center"):
    #         values = []
    #         for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
    #             if not (ss.continuous and not os.continuous):
    #                 raise AttributeError(
    #                     "Use this only to go from a continuous to a discrete space"
    #                 )

    #             _range = (ss.high - ss.low).squeeze()
    #             if mode == "edges":

    #                 _remainder = (sv.squeeze() - ss.low.squeeze()) % (_range / os.N)
    #                 index = int(
    #                     (sv.squeeze() - ss.low.squeeze() - _remainder) / _range * os.N
    #                 )
    #             elif mode == "center":
    #                 N = os.N - 1
    #                 _remainder = (sv.squeeze() - ss.low.squeeze() + (_range / 2 / N)) % (
    #                     _range / (N)
    #                 )

    #                 index = int(
    #                     (sv.squeeze() - ss.low.squeeze() - _remainder + _range / 2 / N)
    #                     / _range
    #                     * N
    #                     + 1e-5
    #                 )  # 1e-5 --> Hack to get around floating point arithmetic
    #             values.append(os.range[0].squeeze()[index])

    #         return values

    #     def _continuous2continuous(self, other):
    #         values = []
    #         for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
    #             if not (ss.continuous and os.continuous):
    #                 raise AttributeError(
    #                     "Use this only to go from a continuous to a continuous space"
    #                 )
    #             s_range = ss.high - ss.low
    #             o_range = os.high - os.low
    #             s_mid = (ss.high + ss.low) / 2
    #             o_mid = (os.high + os.low) / 2

    #             values.append((sv - s_mid) / s_range * o_range + o_mid)
    #             return values

    #     def _discrete2discrete(self, other):

    #         values = []
    #         for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
    #             if ss.continuous or os.continuous:
    #                 raise AttributeError(
    #                     "Use this only to go from a discrete to a discrete space"
    #                 )
    #             values.append(
    #                 os.range[0].squeeze()[ss.range[0].squeeze().tolist().index(sv)]
    #             )
    #             return values

    #     def cast(self, other, mode="center"):
    #         """Convert values from one StateElement with one space to another with another space, whenever a one-to-one mapping is possible. Equally spaced discrete spaces are assumed.

    #         :param StateElement other: The StateElement to which the value is converted to.
    #         :param type mode: How to convert between discrete and continuous spaces. If mode == 'center', then the continuous domain will finish at the center of the edge discrete items. If mode == 'edges', the continous domain will finisg at the edges of the edge discrete items.
    #         :return: A new StateElement with the space from other and the converted value from self.
    #         :rtype: StateElement

    #         """
    #         if not isinstance(other, StateElement):
    #             raise TypeError(
    #                 "input arg {} must be of type StateElement".format(str(other))
    #             )

    #         values = []
    #         for s, o in zip(self, other):
    #             for sv, ss, ov, os in zip(
    #                 s["values"], s["spaces"], o["values"], o["spaces"]
    #             ):
    #                 if not ss.continuous and os.continuous:
    #                     value = s._discrete2continuous(o, mode=mode)
    #                 elif ss.continuous and os.continuous:
    #                     value = s._continuous2continuous(o)
    #                 elif ss.continuous and not os.continuous:
    #                     value = s._continuous2discrete(o, mode=mode)
    #                 elif not ss.continuous and not os.continuous:
    #                     if ss.N == os.N:
    #                         value = s._discrete2discrete(o)
    #                     else:
    #                         raise ValueError(
    #                             "You are trying to match a discrete space to another discrete space of different size."
    #                         )
    #                 else:
    #                     raise NotImplementedError
    #                 values.extend(value)

    #         return StateElement(
    #             values=values,
    #             spaces=other["spaces"],
    #             clipping_mode=self._mix_modes(other),
    #             typing_priority=self.typing_priority,
    #         )

    # if _str == "cast" or _str == "all":

    # y.reset()
    # targetdomain = StateElement(
    #     values=None,
    #     spaces=[
    #         coopihc.space.Space(
    #             [
    #                 -numpy.ones((2, 1), dtype=numpy.float32),
    #                 numpy.ones((2, 1), dtype=numpy.float32),
    #             ]
    #         )
    #         for j in range(3)
    #     ],
    # )
    # res = y.cast(targetdomain)

    # b = StateElement(
    #     values=5,
    #     spaces=coopihc.space.Space(
    #         [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
    #     ),
    # )

    # a = StateElement(
    #     values=0,
    #     spaces=coopihc.space.Space(
    #         [
    #             numpy.array([-1], dtype=numpy.float32),
    #             numpy.array([1], dtype=numpy.float32),
    #         ]
    #     ),
    # )
    # # C2D
    # continuous = []
    # discrete = []
    # for elem in numpy.linspace(-1, 1, 200):
    #     a["values"] = elem
    #     continuous.append(a["values"][0].squeeze().tolist())
    #     discrete.append(a.cast(b, mode="center")["values"][0].squeeze().tolist())
    # import matplotlib.pyplot as plt

    # plt.plot(continuous, discrete, "b*")
    # plt.show()

    # # D2C

    # continuous = []
    # discrete = []
    # for elem in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
    #     b["values"] = elem
    #     discrete.append(elem)
    #     continuous.append(b.cast(a, mode="edges")["values"][0].squeeze().tolist())
    # import matplotlib.pyplot as plt

    # plt.plot(discrete, continuous, "b*")
    # plt.show()

    # # C2C

    # a = StateElement(
    #     values=0,
    #     spaces=coopihc.space.Space(
    #         [
    #             numpy.array([-2], dtype=numpy.float32),
    #             numpy.array([1], dtype=numpy.float32),
    #         ]
    #     ),
    # )
    # b = StateElement(
    #     values=3.5,
    #     spaces=coopihc.space.Space(
    #         [
    #             numpy.array([3], dtype=numpy.float32),
    #             numpy.array([4], dtype=numpy.float32),
    #         ]
    #     ),
    # )
    # c1 = []
    # c2 = []
    # for elem in numpy.linspace(-2, 1, 100):
    #     a["values"] = elem
    #     c1.append(a["values"][0].squeeze().tolist())
    #     c2.append(a.cast(b)["values"][0].squeeze().tolist())
    # import matplotlib.pyplot as plt

    # plt.plot(c1, c2, "b*")
    # plt.show()

    # # D2D
    # a = StateElement(
    #     values=5,
    #     spaces=coopihc.space.Space(
    #         [numpy.array([i for i in range(11)], dtype=numpy.int16)]
    #     ),
    # )
    # b = StateElement(
    #     values=5,
    #     spaces=coopihc.space.Space(
    #         [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
    #     ),
    # )

    # d1 = []
    # d2 = []
    # for i in range(11):
    #     a["values"] = i
    #     d1.append(i)
    #     d2.append(a.cast(b)["values"][0].squeeze().tolist())
    # import matplotlib.pyplot as plt

    # plt.plot(d1, d2, "b*")
    # plt.show()

    @classmethod
    def implements(cls, numpy_function):
        """implements

        Register an __array_function__ implementation for StateElement objects. Example usage for the amax function, with incomplete implementation (only continuous space is targeted). The steps are:

            1. get all the attributes from the StateElement
            2. convert the StateElement to a numpy ndarray
            3. Apply the numpy amax function, and get the corresponding space via argmax
            4. Cast the corresponding space to a Space, the numpy ndarray to a StateElement, and reattach all attributes

        .. code-block:: python

            @StateElement.implements(numpy.amax)
            def amax(arr, **keywordargs):
                spaces, out_of_bounds_mode, kwargs = (
                    arr.spaces,
                    arr.out_of_bounds_mode,
                    arr.kwargs,
                )
                obj = arr.view(numpy.ndarray)
                argmax = numpy.argmax(obj, **keywordargs)
                index = numpy.unravel_index(argmax, arr.spaces.shape)
                obj = numpy.amax(obj, **keywordargs)
                obj = numpy.asarray(obj).view(StateElement)
                if arr.spaces.space_type == "continuous":
                    obj.spaces = autospace(
                        numpy.atleast_2d(arr.spaces.low[index[0], index[1]]),
                        numpy.atleast_2d(arr.spaces.high[index[0], index[1]]),
                    )
                else:
                    raise NotImplementedError
                obj.out_of_bounds_mode = arr.out_of_bounds_mode
                obj.kwargs = arr.kwargs
                return obj



        """

        def decorator(func):
            if cls.HANDLED_FUNCTIONS.get(numpy_function, None) is None:
                cls.HANDLED_FUNCTIONS[numpy_function] = func
            else:
                raise RedefiningHandledFunctionWarning(
                    "You are redefining the existing method {} of StateElement."
                )
            return func

        return decorator

    @staticmethod
    def _process_input_values(numpy_input_array, spaces, out_of_bounds_mode):
        if spaces is None or out_of_bounds_mode is None:
            return numpy_input_array
        if out_of_bounds_mode == "raw":
            return numpy_input_array
        else:
            dtype_container = []
            multidiscrete = True
            # Deal with multidiscrete vs continuous and discrete
            if len(spaces) == 1:
                multidiscrete = False
                numpy_input_array = [numpy_input_array]

            for ni, (v, s) in enumerate(
                itertools.zip_longest(numpy_input_array, spaces)
            ):
                # Make sure value is same input type (shape + dtype) as space. Space specs take over value specs.
                if v is not None:

                    v = numpy.array(v).reshape(s.shape).astype(s.dtype)
                    dtype_container.append(s.dtype.type)
                if v not in s:
                    if out_of_bounds_mode == "error":
                        raise StateNotContainedError(
                            "Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                                str(v), type(v), str(s), str(s.low), str(s.high)
                            )
                        )
                    elif out_of_bounds_mode == "warning":
                        warnings.warn(
                            StateNotContainedWarning(
                                "Warning: Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                                    str(v), type(v), str(s), str(s.low), str(s.high)
                                )
                            )
                        )
                    elif out_of_bounds_mode == "clip":
                        v = StateElement._clip(v, s)
                    else:
                        pass
                numpy_input_array[ni] = v

        common_dtype = numpy.find_common_type(dtype_container, [])
        if multidiscrete:
            return [nip.astype(common_dtype) for nip in numpy_input_array]
        else:
            return numpy_input_array[0].astype(common_dtype)
