from array import array
import copy
import numpy
import json
import itertools
import warnings

from coopihc.base.Space import BaseSpace
from coopihc.base.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
    SpaceNotSeparableError,
)


class StateElement(numpy.ndarray):
    """StateElement

    The container for an element of a state. A numpy array, with an associated space.


    .. code-block:: python

        # Discrete Set
        x = StateElement(2, integer_set(3))
        # Continuous Interval
        x = StateElement(
            numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))), out_of_bounds_mode="error"
        )


    :param input_object: value
    :type input_object: numpy array-like
    :param space: space where input_object takes value
    :type space: `Space<coopihc.base.Space.BaseSpace>`
    :param out_of_bounds_mode: what to do when the value is outside the bound, defaults to "warning". Possible values are:

        * "error" --> raises a StateNotContainedError
        * "warning" --> raises a StateNotContainedWarning
        * "clip" --> clips the data to force it to belong to the space
        * "silent" --> Values not in the space are accepted silently (behavior is roughly equivalent to a regular numpy.ndarray). Broadcasting and type casting may still be applied
        * "raw" --> No data transformation is applied. This is faster than the other options, because the preprocessing of input data is short-circuited. However, this provides no tolerance for ill-specified input.

    :type out_of_bounds_mode: str, optional

    A few examples for out_of_bounds_mode behavior:

    .. code-block:: python

        # Error
        x = StateElement(2, integer_set(3), out_of_bounds_mode="error") # Passes
        x = StateElement(4, integer_set(3), out_of_bounds_mode="error") # raises a ``StateNotContainedError``

        # Warning
        x = StateElement(2, integer_set(3), out_of_bounds_mode="warning") # Passes
        x = StateElement(4, integer_set(3), out_of_bounds_mode="warning") # Passes, but warns with ``StateNotContainedWarning``

        # Clip
        x = StateElement(4, integer_set(3), out_of_bounds_mode="clip")
        assert x == numpy.array([2])

    """

    # Simple static two-way dict
    __precedence__ = {
        "error": 0,
        "warning": 2,
        "clip": 1,
        "silent": 3,
        "raw": 4,
        "0": "error",
        "2": "warning",
        "1": "clip",
        "3": "silent",
        "4": "raw",
    }

    HANDLED_FUNCTIONS = {}
    SAFE_FUNCTIONS = ["all"]

    @staticmethod
    def _clip(value, space):
        """Simple wrapper for numpy clip"""
        if value not in space:
            return numpy.asarray(numpy.clip(value, space.low, space.high))

    @property
    def spacetype(self):
        return self.space.spacetype

    def __new__(cls, input_object, space, out_of_bounds_mode="warning"):
        """__new__, see https://numpy.org/doc/stable/user/basics.subclassing.html"""

        input_object = numpy.asarray(
            StateElement._process_input_values(
                input_object,
                space,
                out_of_bounds_mode,
            )
        )
        obj = input_object.view(cls)
        obj.space = space
        obj.out_of_bounds_mode = out_of_bounds_mode
        return obj

    def __array_finalize__(self, obj):
        """__array_finalize__, see https://numpy.org/doc/stable/user/basics.subclassing.html"""
        if obj is None:
            return
        space = getattr(obj, "space", None)
        out_of_bounds_mode = getattr(obj, "out_of_bounds_mode", None)
        self.space = space
        self.out_of_bounds_mode = out_of_bounds_mode

    @property
    def dtype(self):
        return self.space.dtype

    @property
    def seed(self):
        return self.space.seed

    # Code below is kept as an example in case one day we decide on overriding numpy functions (again).
    # def __array_ufunc__(self, ufunc, method, *input_args, out=None, **kwargs):
    #     """__array_ufunc__, see https://numpy.org/doc/stable/user/basics.subclassing.html.

    #     This can lead to some issues with some numpy universal functions. For example, modf returns the fractional and integral part of an array in the form of two new StateElements. In that case, the fractional part is necessarily in ]0,1[, whatever the actual space of the original StateElement, but the former will receive the latter's space. To deal with that case, it is suggested to select a proper "out_of_bounds_mode", perhaps dynamically, and to change the space attribute of the new object afterwards if actually needed.

    #     """
    #     args = []
    #     argmode = "raw"
    #     # Input and Output conversion to numpy
    #     for _input in input_args:
    #         if isinstance(_input, StateElement):
    #             args.append(_input.view(numpy.ndarray))
    #             if (
    #                 StateElement.__precedence__[_input.out_of_bounds_mode]
    #                 < StateElement.__precedence__[argmode]
    #             ):
    #                 argmode = _input.out_of_bounds_mode
    #         else:
    #             args.append(_input)

    #     outputs = out
    #     if outputs:
    #         out_args = []
    #         for output in outputs:
    #             if isinstance(output, StateElement):
    #                 out_args.append(output.view(numpy.ndarray))
    #             else:
    #                 out_args.append(output)
    #         kwargs["out"] = tuple(out_args)
    #     else:
    #         outputs = (None,) * ufunc.nout

    #     # Actually apply the ufunc to numpy array
    #     result = getattr(ufunc, method)(*args, **kwargs)
    #     if result is NotImplemented:
    #         return NotImplemented
    #     # Back conversion to StateElement. Only pass results in this who need to be processed (types that subclass numpy.number, e.g. exclude booleans)
    #     if isinstance(result, (numpy.ndarray, StateElement)):
    #         if issubclass(result.dtype.type, numpy.number):
    #             result = StateElement._process_input_values(
    #                 result, self.space, self.out_of_bounds_mode
    #             )

    #     # In place
    #     if method == "at":
    #         if isinstance(input_args[0], StateElement):
    #             input_args[0].space = self.space
    #             input_args[0].out_of_bounds_mode = argmode
    #             input_args[0].kwargs = self.kwargs

    #     if ufunc.nout == 1:
    #         result = (result,)

    #     result = tuple(
    #         (numpy.asarray(_result).view(StateElement) if output is None else output)
    #         for _result, output in zip(result, outputs)
    #     )
    #     if result and isinstance(result[0], StateElement):
    #         result[0].space = self.space
    #         result[0].out_of_bounds_mode = argmode
    #         result[0].kwargs = kwargs

    #     return result[0] if len(result) == 1 else result

    # def __array_function__(self, func, types, args, kwargs):
    #     """__array_function__ [summary]

    #     If func is not know to be handled by StateElement, then pass it to Numpy. In that case, a numpy array is returned.
    #     If an implementation for func has been provided that is specific to StateElement via the ``implements`` decorator, then call that one.

    #     See `this NEP<https://numpy.org/neps/nep-0018-array-function-protocol.html>`_ as well as `this Numpy doc<https://numpy.org/doc/stable/user/basics.dispatch.html>`_ ifor more details on how to implement __array_function__.

    #     """
    #     # Calls default numpy implementations and returns a numpy ndarray
    #     if func not in self.HANDLED_FUNCTIONS:
    #         if func.__name__ not in self.SAFE_FUNCTIONS:
    #             warnings.warn(
    #                 NumpyFunctionNotHandledWarning(
    #                     "Numpy function with name {} is currently not implemented for this object with type {}, and CoopIHC is returning a numpy.ndarray object. If you want to have a StateElement object returned, consider implementing your own version of this function and using the implements decorator (example in the decorator's documentation) to add it to the StateElement, as well as formulating a PR to have it included in CoopIHC core code.".format(
    #                         func.__name__, type(self)
    #                     )
    #                 )
    #             )
    #         return (super().__array_function__(func, types, args, kwargs)).view(
    #             numpy.ndarray
    #         )
    #     # Note: this allows subclasses that don't override
    #     # __array_function__ to handle MyArray objects
    #     if not all(issubclass(t, StateElement) for t in types):
    #         return NotImplemented
    #     return self.HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, key):
        """__getitem__

        Includes an extra mechanism, to automatically extract values with the corresponding space, which slightly abuses the slice and indexing notations:

        .. code-block:: python

            x = StateElement(1, integer_set(3))
            assert x[..., {"space": True}] == x
            assert x[..., {"space": True}] is x
            assert x[...] == x


            x = StateElement(
            numpy.array([[0.0, 0.1], [0.2, 0.3]]), box_space(numpy.ones((2, 2)))
            assert x[0, 1, {"space": True}] == StateElement(0.1, box_space(numpy.float64(1)))

        """

        if isinstance(key, tuple) and key[-1] == {"space": True}:
            key = key[:-1]
            item = super().__getitem__(key)

            if key == (Ellipsis,):
                return self
            try:
                space = self.space[key]
            except SpaceNotSeparableError:
                return self

            return StateElement(
                item.view(numpy.ndarray),
                space,
                out_of_bounds_mode=self.out_of_bounds_mode,
            )
        else:
            try:
                return self.view(numpy.ndarray)[key]
            except IndexError:
                # If one-element slice
                try:
                    if key.start == 0 and key.stop == 1 and self.shape == ():
                        return self.view(numpy.ndarray)
                except AttributeError:
                    return self.view(numpy.ndarray)[...]

    def __setitem__(self, key, value):
        """__setitem__

        Simply calls numpy's __setitem__ after having checked input values.

        .. code-block:: python

            x = StateElement(1, integer_set(3))
            x[...] = 2
            assert x == StateElement(2, integer_set(3))
            with pytest.warns(StateNotContainedWarning):
                x[...] = 4

        """
        # The except clause is here to support numpy broadcasting when indexing.
        value = StateElement._process_input_values(
            value, self.space[key], self.out_of_bounds_mode
        )
        super().__setitem__(key, value)

    def __iter__(self):
        """Numpy-style __iter__

        Iterating over the value in Numpy style, with the corresponding space.

        :return: _description_
        :rtype: _type_
        """
        self._iterable_space = iter(self.space)
        self._iterable_value = super().__iter__()
        return self

    def __next__(self):
        """Numpy-style __next__

        Next value in Numpy style, with corresponding space.

        :return: _description_
        :rtype: _type_
        """
        space = (
            self._iterable_space.__next__()
        )  # make sure that space is called before so that it can raise StopIteration before having an IndexError on the values
        value = self._iterable_value.__next__()
        return StateElement(
            value,
            space,
            out_of_bounds_mode=self.out_of_bounds_mode,
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"StateElement({numpy.ndarray.__repr__(self.view(numpy.ndarray))}, {self.space.__repr__()}, '{self.out_of_bounds_mode}')"

    def reset(self, value=None):
        """reset

        Reset the StateElement to a random or chosen value, by sampling the underlying space.

        .. code-block:: python

            x = StateElement(numpy.ones((2, 2)), box_space(numpy.ones((2, 2))))
            x.reset()

            # Forced reset
            x.reset(0.59 * numpy.ones((2, 2)))

        :param value: reset value for forced reset, defaults to None. If None, samples randomly from the space.
        :type value: numpy.ndarray, optional
        """
        if value is None:
            value = self.space.sample()

        # Ellipsis to deal with any dim array
        self[...] = numpy.asarray(value).reshape(self.space.shape).astype(self.dtype)

    def serialize(self):
        """Generate a JSON representation of StateElement.

        .. code-block:: python

            x = StateElement(numpy.array([2]), integer_set(3))
            assert x.serialize() == {
                "values": 2,
                "space": {
                    "space": "CatSet",
                    "seed": None,
                    "array": [0, 1, 2],
                    "dtype": "dtype[int64]",
                },
            }

        :return: JSON serializable content.
        :rtype: dictionnary

        """

        return {
            "values": self.tolist(),
            "space": self.space.serialize(),
        }

    def equals(self, other, mode="soft"):
        """equals

        Soft mode is equivalent to __eq__ inherited from numpy.ndarray.
        In Hard mode, contrary to __eq__, the space and out of bounds mode are also compared.

        .. code-block:: python


            int_space = integer_set(3)
            other_int_space = integer_set(4)
            x = StateElement(numpy.array(1), int_space)
            y = StateElement(numpy.array(1), other_int_space)
            assert x.equals(y)
            assert not x.equals(y, mode="hard")

        :param other: object to compare to
        :type other: StateElement, numpy.ndarray
        :param mode: [description], defaults to "soft"
        :type mode: str, optional
        :return: [description]
        :rtype: [type]
        """
        if mode == "soft":
            return self == other
        if not isinstance(other, StateElement):
            return False
        if self.space != other.space:
            return numpy.full(self.shape, False)
        if self.out_of_bounds_mode != other.out_of_bounds_mode:
            return numpy.full(self.shape, False)
        return numpy.full(self.shape, True)

    def cast(self, other, mode="center"):
        """Convert values of a StateElement taking values in one space to those of another space, if a one-to-one mapping is possible.

        Equally spaced discrete space are assumed when converting between continuous and discrete space.

        The mode parameter indicates how the discrete space is mapped to a continuous space. If ``mode = 'edges'``, then the continuous space will prefectly overlap with unit width intervals of the discrete space. Otherwise, the continuous space' boundaries will match with the center of the two extreme intervals of the discrete space. Examples below, including visualisations.


        .. code-block:: python

            discr_box_space = box_space(low=numpy.int8(1), high=numpy.int8(3))
            cont_box_space = box_space(low=numpy.float64(-1.5), high=numpy.float64(1.5))


            + discrete2continuous:

            .. code-block:: python

                x = StateElement(1, discr_box_space)
                ret_stateElem = x.cast(cont_box_space, mode="edges")
                assert ret_stateElem == StateElement(-1.5, cont_box_space)
                ret_stateElem = x.cast(cont_box_space, mode="center")
                assert ret_stateElem == StateElement(-1, cont_box_space)


            + continuous2continuous:

            .. code-block:: python

                    x = StateElement(0, cont_box_space)
                    ret_stateElem = x.cast(discr_box_space, mode="center")
                    assert ret_stateElem == StateElement(2, discr_box_space)
                    ret_stateElem = x.cast(discr_box_space, mode="edges")
                    assert ret_stateElem == StateElement(2, discr_box_space)

                    center = []
                    edges = []
                    for i in numpy.linspace(-1.5, 1.5, 100):
                        x = StateElement(i, cont_box_space)
                        ret_stateElem = x.cast(discr_box_space, mode="center")
                        if i < -0.75:
                            assert ret_stateElem == StateElement(1, discr_box_space)
                        if i > -0.75 and i < 0.75:
                            assert ret_stateElem == StateElement(2, discr_box_space)
                        if i > 0.75:
                            assert ret_stateElem == StateElement(3, discr_box_space)
                        center.append(ret_stateElem.tolist())

                        ret_stateElem = x.cast(discr_box_space, mode="edges")
                        if i < -0.5:
                            assert ret_stateElem == StateElement(1, discr_box_space)
                        if i > -0.5 and i < 0.5:
                            assert ret_stateElem == StateElement(2, discr_box_space)
                        if i > 0.5:
                            assert ret_stateElem == StateElement(3, discr_box_space)

                        edges.append(ret_stateElem.tolist())

                    import matplotlib.pyplot as plt

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(
                        numpy.linspace(-1.5, 1.5, 100), numpy.array(center) - 0.05, "+", label="center"
                    )
                    ax.plot(
                        numpy.linspace(-1.5, 1.5, 100), numpy.array(edges) + 0.05, "o", label="edges"
                    )
                    ax.legend()
                    plt.show()

            + continuous2continuous: (currently only works if all elements of the lower and upper bounds are equal (e.g. autospace([[-1,-1]],[[1,1]]) would work, but not autospace([[-1,-2]],[[1,1]]))

            .. code-block:: python

                cont_space = box_space(numpy.full((2, 2), 1), dtype=numpy.float32)
                other_cont_space = box_space(
                    low=numpy.full((2, 2), 0), high=numpy.full((2, 2), 4), dtype=numpy.float32
                )

                for i in numpy.linspace(-1, 1, 100):
                    x = StateElement(numpy.full((2, 2), i), cont_space)
                    ret_stateElement = x.cast(other_cont_space)
                    assert (ret_stateElement == (x + 1) * 2).all()

            + discrete2discrete:

            .. code-block:: python

                discr_box_space = box_space(low=numpy.int8(1), high=numpy.int8(4))
                other_discr_box_space = box_space(low=numpy.int8(11), high=numpy.int8(14))

                for i in [1, 2, 3, 4]:
                    x = StateElement(i, discr_box_space)
                    ret_stateElement = x.cast(other_discr_box_space)
                    assert ret_stateElement == x + 10


        :param other: Space to cast values to. Also works with a StateElement.
        :type other: :py:class:`Space <coopihc.base.Space.Space>`
        :param mode: how to map discrete and continuous space, defaults to "center". See examples in the documentation.
        :type mode: str, optional
        """
        if not isinstance(other, (StateElement, BaseSpace)):
            raise TypeError(
                "input arg {} of type {} must be of type StateElement or Space".format(
                    str(other), type(other)
                )
            )

        if isinstance(other, StateElement):
            mix_outbounds = min(
                self.__precedence__[self.out_of_bounds_mode],
                self.__precedence__[other.out_of_bounds_mode],
            )

            mix_outbounds = self.__precedence__[str(mix_outbounds)]
            other = other.space

        else:
            mix_outbounds = self.out_of_bounds_mode

        if self.spacetype == "discrete" and other.spacetype == "continuous":
            value = self._discrete2continuous(other, mode=mode)
        elif self.spacetype == "continuous" and other.spacetype == "continuous":
            value = self._continuous2continuous(other)
        elif self.spacetype == "continuous" and other.spacetype == "discrete":
            value = self._continuous2discrete(other, mode=mode)
        elif self.spacetype == "discrete" and other.spacetype == "discrete":
            if self.space.N == other.N:
                value = self._discrete2discrete(other)
            else:
                raise ValueError(
                    "You are trying to match a discrete space to another discrete space of different size {} != {}.".format(
                        self.space.N, other.N
                    )
                )
        else:
            raise NotImplementedError

        return StateElement(
            numpy.atleast_2d(numpy.array(value)),
            other,
            out_of_bounds_mode=mix_outbounds,
        )

    def _discrete2continuous(self, other, mode="center"):

        if mode == "edges":
            ls = numpy.linspace(other.low, other.high, self.space.N)
            shift = 0
        elif mode == "center":
            ls = numpy.linspace(other.low, other.high, self.space.N + 1)
            shift = (ls[1] - ls[0]) / 2

        value = shift + ls[self.space.array.tolist().index(self[...])]
        return numpy.array(value).reshape((-1, 1))

    def _continuous2discrete(self, other, mode="center"):

        _range = (self.space.high - self.space.low).squeeze()
        if mode == "edges":
            _remainder = (self[...] - self.space.low.squeeze()) % (_range / other.N)
            index = min(
                int((self[...] - self.space.low - _remainder) / _range * other.N),
                other.N - 1,
            )
        elif mode == "center":
            N = other.N - 1
            _remainder = (self[...] - self.space.low + (_range / 2 / N)) % (
                _range / (N)
            )

            index = int(
                (self[...] - self.space.low - _remainder + _range / 2 / N) / _range * N
                + 1e-5
            )  # 1e-5 --> Hack to get around floating point arithmetic
        return other.array.tolist()[index]

    def _continuous2continuous(self, other):

        s_range = self.space.high - self.space.low
        o_range = other.high - other.low
        s_mid = (self.space.high + self.space.low) / 2
        o_mid = (other.high + other.low) / 2

        return (self[...] - s_mid) / s_range * o_range + o_mid

    def _discrete2discrete(self, other):

        return other.array[self.space.array.tolist().index(self[...].tolist())]

    def _tabulate(self):
        """_tabulate

        outputs a list ready for tabulate.tabulate(), as well as the number of lines of the generated table.

        Examples:

        .. code-block::
            >>> x = StateElement(1, integer_set(3))
            >>> x._tabulate()
            ([[array(1), 'CatSet(3)']], 1)
            >>> tabulate(x._tabulate()[0])
            '-  ---------\n1  CatSet(3)\n-  ---------'



            >>> x = StateElement(numpy.zeros((3, 3)), box_space(numpy.ones((3, 3))))
            >>> x._tabulate()
            ([[array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]), '\nNumeric(3, 3)\n']], 3)
            >>> tabulate(x._tabulate()[0])
            '------------  -------------\n[[0. 0. 0.]   Numeric(3, 3)\n [0. 0. 0.]\n [0. 0. 0.]]\n------------  -------------'



        :return: list ready for tabulate.tabulate(), line numbers
        :rtype: tuple(list, int)
        """
        try:
            string_space = ["" for i in range(self.shape[0])]
        except IndexError:  # for shapes (X,) and ()
            string_space = [""]
        try:
            _index = int(len(self) / 2)
        except TypeError:
            _index = 0
        if self.space.__class__.__name__ == "Numeric":
            if self.space.seed is None:
                string_space[
                    _index
                ] = f"Numeric{self.space.shape} - {self.space.dtype}".format()
            else:
                string_space[
                    _index
                ] = f"Numeric{self.space.shape} - {self.space.dtype} (seed:{self.space.seed})".format()
            array = self.view(numpy.ndarray)
        elif self.space.__class__.__name__ == "CatSet":
            if self.space.seed is None:
                string_space[_index] = f"CatSet({self.space.N}) - {self.space.dtype}"
            else:
                string_space[
                    _index
                ] = f"CatSet({self.space.N}) - {self.space.dtype} seed:{self.space.seed}"

            array = self.view(numpy.ndarray)
        else:
            raise NotImplementedError

        try:
            _shape = self.shape[0]
        except IndexError:
            _shape = 1
        return ([[array, "\n".join(string_space)]], _shape)

    # @classmethod
    # def implements(cls, numpy_function):
    #     """implements

    #     Register an __array_function__ implementation for StateElement objects. Example usage for the amax function, with incomplete implementation (only continuous space is targeted). The steps are:

    #         1. get all the attributes from the StateElement
    #         2. convert the StateElement to a numpy ndarray
    #         3. Apply the numpy amax function, and get the corresponding space via argmax
    #         4. Cast the corresponding space to a Space, the numpy ndarray to a StateElement, and reattach all attributes

    #     .. code-block:: python

    #         @StateElement.implements(numpy.amax)
    #         def amax(arr, **keywordargs):
    #             space, out_of_bounds_mode, kwargs = (
    #                 arr.space,
    #                 arr.out_of_bounds_mode,
    #                 arr.kwargs,
    #             )
    #             obj = arr.view(numpy.ndarray)
    #             argmax = numpy.argmax(obj, **keywordargs)
    #             index = numpy.unravel_index(argmax, arr.space.shape)
    #             obj = numpy.amax(obj, **keywordargs)
    #             obj = numpy.asarray(obj).view(StateElement)
    #             if arr.space.space_type == "continuous":
    #                 obj.space = autospace(
    #                     numpy.atleast_2d(arr.space.low[index[0], index[1]]),
    #                     numpy.atleast_2d(arr.space.high[index[0], index[1]]),
    #                 )
    #             else:
    #                 raise NotImplementedError
    #             obj.out_of_bounds_mode = arr.out_of_bounds_mode
    #             obj.kwargs = arr.kwargs
    #             return obj

    #     """

    #     def decorator(func):
    #         if cls.HANDLED_FUNCTIONS.get(numpy_function, None) is None:
    #             cls.HANDLED_FUNCTIONS[numpy_function] = func
    #         else:
    #             raise RedefiningHandledFunctionWarning(
    #                 "You are redefining the existing method {} of StateElement."
    #             )
    #         return func

    #     return decorator

    @staticmethod
    def _process_input_values(input_object, space, out_of_bounds_mode):
        if space is None or out_of_bounds_mode is None:
            return input_object
        if out_of_bounds_mode == "raw":
            return input_object
        try:
            input_object = (
                numpy.asarray(input_object).reshape(space.shape).astype(space.dtype)
            )
        except ValueError:
            if numpy.atleast_1d(numpy.asarray(input_object)).shape == 1:
                input_object = numpy.full(space.shape, input_object, space.dtype)

        if input_object not in space:
            if out_of_bounds_mode == "error":
                raise StateNotContainedError(
                    "Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                        str(input_object),
                        type(input_object),
                        str(space),
                        str(space.low),
                        str(space.high),
                    )
                )
            elif out_of_bounds_mode == "warning":
                warnings.warn(
                    StateNotContainedWarning(
                        "Warning: Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                            str(input_object),
                            type(input_object),
                            str(space),
                            str(space.low),
                            str(space.high),
                        )
                    )
                )
            elif out_of_bounds_mode == "clip":

                input_object = StateElement._clip(input_object, space)
            else:
                pass

        return input_object
