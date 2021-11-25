from collections import OrderedDict
import copy
import numpy
import json
import sys

numpy.set_printoptions(precision=3, suppress=True)


from coopihc.helpers import flatten, hard_flatten
import itertools
from tabulate import tabulate


def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix) :]


from coopihc.helpers import isdefined


class Space:
    """Class used to define in which domain values of state elements live.


    Input should be given as a list of arrays:

        * In case the data is continuous, then provide [low, high], where low and high are arrays that define the values of the lower and upper bounds
        * In case the data is discrete, provide [a1, a2, a3, ...], where the ai's are different ranges corresponding to different subspaces (multidiscrete space)


    If data is discrete it is stored as a column (N,1) array.

    :param [numpy.array] array_list: A list of NumPy arrays that specifies the ranges of the Space.
    :param *args: For future use.
    :param **kwargs: For future use `**kwargs`.
    """

    def __init__(self, array_list, *args, **kwargs):
        self._cflag = None
        self.rng = numpy.random.default_rng()

        # Deal with variable format input
        if isinstance(array_list, numpy.ndarray):
            pass
        else:
            for _array in array_list:
                if not isinstance(_array, numpy.ndarray):
                    raise AttributeError(
                        "Input argument array_list must be or must inherit from numpy.ndarray instance."
                    )
        self._array = array_list

        self._range = None
        self._shape = None
        self._dtype = None

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        if self.continuous:
            return "Space[Continuous({}), {}]".format(self.shape, self.dtype)
        else:
            return "Space[Discrete({}), {}]".format(
                [max(r.shape) for r in self.range], self.dtype
            )

    def __contains__(self, item):
        if item.shape != self.shape:
            try:
                # Don't actually store reshaped item, just see if it works
                item.reshape(self.shape)
            except ValueError:
                return False
        if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
            return False
        if self.continuous:
            return numpy.all(item >= self.low) and numpy.all(item <= self.high)
        else:
            return numpy.array([item[n] in r for n, r in enumerate(self.range)]).all()

    def __iter__(self):
        self.n = 0
        if not self.continuous:
            self.max = len(self)
        else:
            if self.shape[0] != 1:
                self.max = self.shape[0]
                self.__iter__row = True
            elif self.shape[0] == 1 and self.shape[1] != 1:
                self.max = self.shape[1]
                self.__iter__row = False
            else:
                self.max = 1
                self.__iter__row = True
        return self

    def __next__(self):

        if self.n < self.max:
            if not self.continuous:
                result = Space([self._array[self.n]])
            else:
                if self.__iter__row:
                    result = Space([self.low[self.n, :], self.high[self.n, :]])
                else:
                    result = Space([self.low[:, self.n], self.high[:, self.n]])
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def range(self):
        """If the space is continuous, returns low and high arrays, after having checked that they have the same shape. If the space is discrete, returns the list of possible values. The output is reshaped to 2d arrays.

        :return: the 2d-reshaped ranges
        :rtype: list

        """

        if self._range is None:
            if self.continuous:
                if not self._array[0].shape == self._array[1].shape:
                    return AttributeError(
                        "The low {} and high {} ranges don't have matching shapes".format(
                            self._array[0], self._array[1]
                        )
                    )
            self._range = [numpy.atleast_2d(_a) for _a in self._array]
        return self._range

    @property
    def low(self):
        """Return the lower end of the range. For continuous simply return low, for discrete return smallest value or the range array.

        :return: The lower end of the range
        :rtype: numpy.array

        """
        if self.continuous:
            low = self.range[0]
        else:
            low = [min(r.squeeze()) for r in self.range]
        return low

    @property
    def high(self):
        """Return the higher end of the range, see low.

        :return: The higher end of the range
        :rtype: numpy.array

        """
        if self.continuous:
            high = self.range[1]
        else:
            high = [max(r.squeeze()) for r in self.range]
        return high

    @property
    def N(self):
        """Returns the number of elements in the range. Only useful for 1d discrete spaces.

        :return: Description of returned object.
        :rtype: type

        """
        if self.continuous:
            return None
        else:
            if len(self) > 1:
                return None
            else:
                return len(self.range[0].squeeze())

    @property
    def shape(self):
        """Returns the shape of the space, discrete(N) spaces are cast to (N,1).

        :return: Shape of the space
        :rtype: tuple

        """
        if self._shape is None:
            if not self.continuous:
                self._shape = (len(self), 1)
            else:
                self._shape = self.low.shape
        return self._shape

    @property
    def dtype(self):
        """Returns the dtype of the space. If data is in several types, will convert to the common type.

        :return: dtype of the data
        :rtype: numpy.dtype

        """
        if self._dtype is None:
            if len(self._array) == 1:
                self._dtype = self._array[0].dtype
            else:
                self._dtype = numpy.find_common_type([v.dtype for v in self._array], [])
        return self._dtype

    @property
    def continuous(self):
        """Whether the space is continuous or not. Based on the dtype of the provided data.

        :return: is continuous.
        :rtype: boolean

        """
        if self._cflag is None:
            self._cflag = numpy.issubdtype(self.dtype, numpy.inexact)
        return self._cflag

    def sample(self):
        """Uniforly samples from the space.

        :return: random value in the space.
        :rtype: numpy.array

        """
        _l = []
        if self.continuous:
            return (self.high - self.low) * self.rng.random(
                self.shape, dtype=self.dtype
            ) + self.low
        else:
            # The conditional check for __iter__ is here to deal with single value spaces. Will not work if that value happens to be a string, but that is okay.
            return [
                self.rng.choice(r.squeeze(), replace=True).astype(self.dtype)
                if hasattr(r.squeeze().tolist(), "__iter__")
                else r
                for r in self.range
            ]

    def serialize(self):
        """Call this to generate a dict representation of Space.

        :return: dictionary representation of a Space object
        :rtype: dict
        """
        return {"array_list": self._array}

    def from_dict(data):
        """Call this to generate a Space from a representation as a dictionary.

        :param data: dictionary representation of a Space
        :type data: dict
        :return: Space with the supplied information
        :rtype: Space
        """
        # Do not create new instance, if type is already Space
        if type(data) is Space:
            return data
        return Space(**data)


class SpaceLengthError(Exception):
    """Error raised when the space length does not match the value length.

    :meta private:

    """

    __module__ = Exception.__module__


class StateNotContainedError(Exception):
    """Error raised when the value is not contained in the space.

    :meta private:

    """

    __module__ = Exception.__module__


class StateElement:
    """The container that defines a substate. StateElements allow:
        - iteration
        - indexing
        - length
        - comparisons
        - various arithmetic operations

    :param list(numpy.array) values: Value of the substate. Some processing is applied to values if the input does not match the space and correct syntax.
    :param list(Space) spaces: Space (domain) in which value lives. Some processing is applied to spaces if the input does not match the correct syntax.
    :param str clipping_mode: What to do when the value is not in the space. If 'error', raise a StateNotContainedError. If 'warning', print a warning on stdout. If 'clip', automatically clip the value so that it is contained. Defaults to 'warning'.
    :param str typing_priority: What to do when the type of the space does not match the type of the value. If 'space', convert the value to the dtype of the space, if 'value' the converse. Default to space.

    """

    __array_priority__ = 1  # to make __rmatmul__ possible with numpy arrays
    __precedence__ = ["error", "warning", "clip"]

    def __init__(
        self,
        values=None,
        spaces=None,
        clipping_mode="warning",
        typing_priority="space",
    ):
        self.clipping_mode = clipping_mode
        self.typing_priority = typing_priority
        self.__values, self.__spaces = None, None

        if spaces is not None:
            for s in spaces:
                if not isinstance(s, Space):
                    raise AttributeError("Input argument spaces expects Space objects")
        self.spaces = spaces
        self.values = values

    # ============ Properties

    @property
    def values(self):
        """Return values of the StateElement.

        :return: StateElement values.
        :rtype: list(numpy.array)

        """
        return self.__values

    @values.setter
    def values(self, values):
        self.__values = self._preprocess_values(values)

    @property
    def spaces(self):
        """Return spaces of the StateElement.

        :return: StateElement spaces.
        :rtype: list(Space)

        """
        return self.__spaces

    @spaces.setter
    def spaces(self, values):
        self.__spaces = self._preprocess_spaces(values)

    # ============ dunder/magic methods

    # Iteration
    def __iter__(self):
        self.n = 0
        self.max = len(self.spaces)
        return self

    def __next__(self):
        if self.n < self.max:
            result = StateElement(
                values=self.values[self.n],
                spaces=self.spaces[self.n],
                clipping_mode=self.clipping_mode,
            )
            self.n += 1
            return result
        else:
            raise StopIteration

    # Length
    def __len__(self):
        return len(self.spaces)

    # Itemization
    def __setitem__(self, key, item):
        if key == "values":
            setattr(self, key, self._preprocess_values(item))
        elif key == "spaces":
            setattr(self, key, self._preprocess_spaces(item))
        elif key == "clipping_mode":
            setattr(self, key, item)
        else:
            raise ValueError(
                'Key should belong to ["values", "spaces", "clipping_mode"]'
            )

    def __getitem__(self, key):
        if key in ["values", "spaces", "clipping_mode"]:
            return getattr(self, key)
        elif isinstance(key, (int, numpy.int)):
            return StateElement(
                values=self.values[key],
                spaces=self.spaces[key],
                clipping_mode=self.clipping_mode,
            )
        else:
            raise NotImplementedError(
                'Indexing only works with keys ("values", "spaces", "clipping_mode") or integers'
            )

    def __getattr__(self, key):
        _np = sys.modules["numpy"]
        if hasattr(_np, key):
            # adapted from https://stackoverflow.com/questions/13776504/how-are-arguments-passed-to-a-function-through-getattr
            def wrapper(*args, **kwargs):
                return getattr(_np, key)(self.values, *args, **kwargs)

            return wrapper

        raise AttributeError(
            "StateElement does not have attribute {}. Tried to fall back to numpy but it also did not have this attribute".format(
                key
            )
        )

    # Comparison
    def _comp_preface(self, other):
        if isinstance(other, StateElement):
            other = other["values"]
        if not isinstance(other, list):
            other = flatten(other)
        return self, other

    def __eq__(self, other):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.equal(s, o) for s, o in zip(self["values"], other)]
        ).all()

    def __lt__(self, other):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.less(s, o) for s, o in zip(self["values"], other)]
        ).all()

    def __gt__(self, other):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.greater(s, o) for s, o in zip(self["values"], other)]
        ).all()

    def __le__(self, other):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.less_equal(s, o) for s, o in zip(self["values"], other)]
        ).all()

    def __ge__(self, other):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.greater_equal(s, o) for s, o in zip(self["values"], other)]
        ).all()

    # Arithmetic
    def __neg__(self):
        return StateElement(
            values=[-u for u in self["values"]],
            spaces=self.spaces,
            clipping_mode=self.clipping_mode,
        )

    def __add__(self, other):
        _elem, other = self._preface(other)
        _elem["values"] = numpy.add(self["values"], other, casting="same_kind")
        return _elem

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        _elem, other = self._preface(other)
        _elem["values"] = numpy.multiply(self["values"], other, casting="same_kind")

        return _elem

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        _elem, other = self._preface(other)
        _elem["values"] = numpy.power(self["values"], other, casting="same_kind")

    def __matmul__(self, other):
        """
        Coarse implementation. Does not deal with all cases
        """
        if isinstance(other, StateElement):
            matA = self["values"][0]
            matB = other["values"][0]
        elif isinstance(other, numpy.ndarray):
            matA = self.values[0]
            matB = other
        else:
            raise TypeError(
                "rhs of {}@{} should be either a StateElement containing a numpy.ndarray or a numpy.ndarray"
            )

        values = matA @ matB
        low = self.spaces[0].low[: values.shape[0], : values.shape[1]]
        high = self.spaces[0].high[: values.shape[0], : values.shape[1]]

        return StateElement(
            values=values,
            spaces=Space([low, high]),
        )

    def __rmatmul__(self, other):
        se = copy.copy(self)
        if isinstance(other, StateElement):
            matA = self.values[0]
            matB = other.values[0]
        elif isinstance(other, numpy.ndarray):
            matA = self.values[0]
            matB = other
        else:
            raise TypeError(
                "lhs of {}@{} should be either a StateElement containing a numpy.ndarray or a numpy.ndarray"
            )
        values = matB @ matA
        if values.shape == se.spaces[0].shape:
            se["values"] = [values]
            return se

        else:
            low = -numpy.inf * numpy.ones(values.shape)
            high = numpy.inf * numpy.ones(values.shape)

            se.spaces = [
                Space(range=[low, high], shape=values.shape, dtype=values.dtype)
            ]
            se["values"] = [values]
            return se

    # Representation
    def __str__(self):
        return "[StateElement - {}] - Value {} in {}".format(
            self.clipping_mode, self.values, self.spaces
        )

    def __repr__(self):
        return "StateElement([{}] - {},...)".format(
            self.clipping_mode, self.values.__repr__()
        )

    # Copy
    # https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
    # Here we override copy and deepcopy simply because there seems to be a huge overhead in the default deepcopy implementation.
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    # ========== Helpers

    def _preface(self, other):
        if not isinstance(other, (StateElement, numpy.ndarray)):
            other = numpy.array(other)
        if hasattr(other, "values"):
            other = other["values"]

        _elem = StateElement(
            values=self.values,
            spaces=self.spaces,
            clipping_mode=self._mix_modes(other),
        )
        return _elem, other

    def serialize(self):
        """Call this to generate a json representation of StateElement.

        :return: JSON blob-like.
        :rtype: dictionnary

        """
        v_list = []
        for v in self["values"]:
            try:
                json.dumps(v)
                v_list.append(v)
            except TypeError:
                if isinstance(v, numpy.ndarray):
                    v_list.extend(v.tolist())
                elif isinstance(v, numpy.generic):
                    v_list.append(v.item())
                else:
                    raise TypeError("".format(msg))
        return {
            "values": v_list,
            "spaces": [space.serialize() for space in self["spaces"]],
        }

    def from_dict(data):
        """Call this to generate a StateElement from a representation as a dictionary.

        :param data: dictionary representation of a StateElement
        :type data: dict
        :return: StateElement with the supplied information
        :rtype: StateElement
        """
        kwargs = data
        if "spaces" in data:
            kwargs["spaces"] = [
                Space.from_dict(space_dict) for space_dict in data["spaces"]
            ]
        return StateElement(**kwargs)

    def _mix_modes(self, other):
        if hasattr(other, "clipping_mode"):
            return self.__precedence__[
                min(
                    self.__precedence__.index(self.clipping_mode),
                    self.__precedence__.index(other.clipping_mode),
                )
            ]
        else:
            return self.clipping_mode

    def cartesian_product(self):
        """Computes the cartesian product of the space, i.e. produces all possible values of the space. This only makes sense for discrete spaces. If the space mixed continuous and discrete values, the continuous value is kept constant.

        :return: The list of all possible StateElements
        :rtype: list(StateElement)

        """
        lists = []
        for value, space in zip(self.values, self.spaces):
            # print(value, space)
            if not space.continuous:
                for r in space.range:
                    lists.append(r.squeeze().tolist())
            else:
                value = value.squeeze().tolist()
                if not isinstance(value, list):
                    value = [value]
                lists.append(value)
        return [
            StateElement(
                values=list(element),
                spaces=self.spaces,
                clipping_mode=self.clipping_mode,
            )
            for element in itertools.product(*lists)
        ]

    def reset(self, dic=None):
        """Initializes the values of the StateElement in place. If dic is not provided, uniformly samples from the space, else, apply the values supplied by the dictionnary.

        :param dictionnary dic: {key: value} where key is either "values" or "spaces" and value is the actual value of that field.


        """
        if not dic:
            # could add a mode which simulates sampling without replacement
            self["values"] = [space.sample() for space in self.spaces]
        else:
            values = dic.get("values")
            if values is not None:
                self["values"] = values
            spaces = dic.get("spaces")
            if spaces is not None:
                self["spaces"] = spaces

    def _preprocess_spaces(self, spaces):

        spaces = flatten([spaces])
        return spaces

    def _preprocess_values(self, values):
        values = flatten([values])
        # Allow a single None syntax
        try:
            if values == [None]:
                values = [None for s in self.spaces]
        except ValueError:
            pass
        # Check for length match, including pass through flatten
        if len(values) != len(self.spaces):
            _values = []
            k = 0
            for s in self.spaces:
                l = len(s)
                _values.append(values[k : k + l])
                k += l
            if len(_values) == len(self.spaces):
                values = _values
            else:
                raise SpaceLengthError(
                    "The size of the values ({}) being instantiated does not match the size of the space ({})".format(
                        len(values), len(self.spaces)
                    )
                )
        # Make sure values are contained
        for ni, (v, s) in enumerate(zip(values, self.spaces)):
            if v is None:
                continue
            if self.typing_priority == "space":
                v = numpy.array(v).reshape(s.shape).astype(s.dtype)
            elif self.typing_priority == "value":
                v = numpy.array(v).reshape(s.shape)
                s._dtype = v.dtype
            else:
                raise NotImplementedError

            if v not in s:
                if self.clipping_mode == "error":
                    raise StateNotContainedError(
                        "Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                            str(v), type(v), str(s), str(s.low), str(s.high)
                        )
                    )
                elif self.clipping_mode == "warning":
                    print(
                        "Warning: Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                            str(v), type(v), str(s), str(s.low), str(s.high)
                        )
                    )
                elif self.clipping_mode == "clip":
                    v = self._clip(v, s)
                    # v = self._clip_1d(v, s)
                else:
                    pass
            values[ni] = v
        return values

    def _flat(self):
        return (
            self["values"],
            self["spaces"],
            [str(i) for i, v in enumerate(self["values"])],
        )

    # def _clip(self, values,):
    #     values = flatten([values])
    #     for n, (value, space) in enumerate(zip(values, self.spaces)):
    #         values[n] = self._clip_1d(value, space)
    #     return values

    def _clip(self, values, spaces):
        values = flatten([values])
        for n, (value, space) in enumerate(zip(values, spaces)):
            values[n] = self._clip_1d(value, space)
        return values

    def _clip_1d(self, value, space):
        if value not in space:
            if space.continuous:
                return numpy.clip(value, space.low, space.high)
            else:
                if value > space.high:
                    value = numpy.atleast_2d(space.high)
                else:
                    value = numpy.atleast_2d(space.low)
                return value

    def _discrete2continuous(self, other, mode="center"):
        values = []
        for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
            if not (not ss.continuous and os.continuous):
                raise AttributeError(
                    "Use this only to go from a discrete to a continuous space"
                )

            _range = ss.range[0].squeeze()

            if mode == "edges":
                N = len(_range)
                ls = numpy.linspace(os.low, os.high, N)
                shift = 0
            elif mode == "center":
                N = len(_range) + 1
                ls = numpy.linspace(os.low, os.high, N)
                shift = (ls[1] - ls[0]) / 2

            values.append(ls[list(_range).index(sv)] + shift)
        return values

    def _continuous2discrete(self, other, mode="center"):
        values = []
        for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
            if not (ss.continuous and not os.continuous):
                raise AttributeError(
                    "Use this only to go from a continuous to a discrete space"
                )

            _range = (ss.high - ss.low).squeeze()
            if mode == "edges":

                _remainder = (sv.squeeze() - ss.low.squeeze()) % (_range / os.N)
                index = int(
                    (sv.squeeze() - ss.low.squeeze() - _remainder) / _range * os.N
                )
            elif mode == "center":
                N = os.N - 1
                _remainder = (sv.squeeze() - ss.low.squeeze() + (_range / 2 / N)) % (
                    _range / (N)
                )

                index = int(
                    (sv.squeeze() - ss.low.squeeze() - _remainder + _range / 2 / N)
                    / _range
                    * N
                    + 1e-5
                )  # 1e-5 --> Hack to get around floating point arithmetic
            values.append(os.range[0].squeeze()[index])

        return values

    def _continuous2continuous(self, other):
        values = []
        for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
            if not (ss.continuous and os.continuous):
                raise AttributeError(
                    "Use this only to go from a continuous to a continuous space"
                )
            s_range = ss.high - ss.low
            o_range = os.high - os.low
            s_mid = (ss.high + ss.low) / 2
            o_mid = (os.high + os.low) / 2

            values.append((sv - s_mid) / s_range * o_range + o_mid)
            return values

    def _discrete2discrete(self, other):

        values = []
        for sv, ss, os in zip(self["values"][0], self["spaces"][0], other["spaces"][0]):
            if ss.continuous or os.continuous:
                raise AttributeError(
                    "Use this only to go from a discrete to a discrete space"
                )
            values.append(
                os.range[0].squeeze()[ss.range[0].squeeze().tolist().index(sv)]
            )
            return values

    def cast(self, other, mode="center"):
        """Convert values from one StateElement with one space to another with another space, whenever a one-to-one mapping is possible. Equally spaced discrete spaces are assumed.

        :param StateElement other: The StateElement to which the value is converted to.
        :param type mode: How to convert between discrete and continuous spaces. If mode == 'center', then the continuous domain will finish at the center of the edge discrete items. If mode == 'edges', the continous domain will finisg at the edges of the edge discrete items.
        :return: A new StateElement with the space from other and the converted value from self.
        :rtype: StateElement

        """
        if not isinstance(other, StateElement):
            raise TypeError(
                "input arg {} must be of type StateElement".format(str(other))
            )

        values = []
        for s, o in zip(self, other):
            for sv, ss, ov, os in zip(
                s["values"], s["spaces"], o["values"], o["spaces"]
            ):
                if not ss.continuous and os.continuous:
                    value = s._discrete2continuous(o, mode=mode)
                elif ss.continuous and os.continuous:
                    value = s._continuous2continuous(o)
                elif ss.continuous and not os.continuous:
                    value = s._continuous2discrete(o, mode=mode)
                elif not ss.continuous and not os.continuous:
                    if ss.N == os.N:
                        value = s._discrete2discrete(o)
                    else:
                        raise ValueError(
                            "You are trying to match a discrete space to another discrete space of different size."
                        )
                else:
                    raise NotImplementedError
                values.extend(value)

        return StateElement(
            values=values,
            spaces=other["spaces"],
            clipping_mode=self._mix_modes(other),
            typing_priority=self.typing_priority,
        )


class State(OrderedDict):
    """The container that defines states.

    :param *args: Same as collections.OrderedDict
    :param **kwargs: Same as collections.OrderedDict
    :return: A state Object
    :rtype: State

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, dic={}):
        """Initialize the state. See StateElement"""
        for key, value in self.items():
            reset_dic = dic.get(key)
            if reset_dic is None:
                reset_dic = {}
            value.reset(reset_dic)

    def _flat(self):
        values = []
        spaces = []
        labels = []
        l, k = list(self.values()), list(self.keys())
        for n, item in enumerate(l):
            _values, _spaces, _labels = item._flat()
            values.extend(_values)
            spaces.extend(_spaces)
            labels.extend([k[n] + "|" + label for label in _labels])

        return values, spaces, labels

    def filter(self, mode, filterdict=None):
        """Retain only parts of the state.

        An example for filterdict's structure is as follows:

        ordereddict = OrderedDict(
        {"substate1": OrderedDict({"substate_x": 0, "substate_w": 0})}
            )
        will filter out every component but the first component (index 0) for substates x and w contained in substate_1.

        :param str mode: Wheter the filtering operates on the 'values' or on the 'spaces'
        :param collections.OrderedDict filterdict: The OrderedDict which specifies which substates to keep and which to leave out.
        :return: The filtered state
        :rtype: State

        """

        new_state = OrderedDict()
        if filterdict is None:
            filterdict = self
        for key, values in filterdict.items():
            if isinstance(self[key], State):
                new_state[key] = self[key].filter(mode, values)
            elif isinstance(self[key], StateElement):
                # to make S.filter("values", S) possible.
                # Warning: Contrary to what one would expect values != self[key]
                if isinstance(values, StateElement):
                    values = slice(0, len(values), 1)
                if mode == "spaces":
                    new_state[key] = flatten([self[key][mode][values]])
                else:
                    new_state[key] = self[key][mode][values]
            else:
                new_state[key] = self[key]

        return new_state

    def __content__(self):
        return list(self.keys())

    # Here we override copy and deepcopy simply because there seems to be some overhead in the default deepcopy implementation. It turns out the gain is almost None, but keep it here as a reminder that deepcopy needs speeding up.  Adapted from StateElement code
    def __copy__(self):
        cls = self.__class__
        copy_object = cls.__new__(cls)
        copy_object.__dict__.update(self.__dict__)
        copy_object.update(self)
        return copy_object

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        deepcopy_object = cls.__new__(cls)
        memodict[id(self)] = deepcopy_object
        deepcopy_object.__dict__.update(self.__dict__)
        for k, v in self.items():
            deepcopy_object[k] = copy.deepcopy(v, memodict)
        return deepcopy_object

    def serialize(self):
        """Serialize state --> JSON output.

        :return: JSON-like blob
        :rtype: dict

        """
        ret_dict = {}
        for key, value in dict(self).items():
            try:
                value_ = json.dumps(value)
            except TypeError:
                try:
                    value_ = value.serialize()
                except AttributeError:
                    print(
                        "warning: I don't know how to serialize {}. I'm sending the whole internal dictionnary of the object. Consider adding a serialize() method to your custom object".format(
                            value.__str__()
                        )
                    )
                    value_ = value.__dict__
            ret_dict[key] = value_
        return ret_dict

    def __str__(self):
        """Print out the game_state and the name of each substate with according indices."""

        table_header = ["Index", "Label", "Value", "Space", "Possible Value"]
        table_rows = []
        for i, (v, s, l) in enumerate(zip(*self._flat())):
            table_rows.append([str(i), l, str(v), str(s)])

        _str = tabulate(table_rows, table_header)

        return _str


# ================ Some Examples ==============
if __name__ == "__main__":

    # [start-space-def]
    continous_space = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )

    discrete_spaces = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    # [end-space-def]

    # [start-space-complex-def]
    h = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(4)], dtype=numpy.int16),
                    numpy.array([i for i in range(4)], dtype=numpy.int16),
                ]
                for j in range(3)
            ]
        )
    )

    none_space = Space([numpy.array([None], dtype=numpy.object)])
    # [end-space-complex-def]

    # [start-space-contains]
    space = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    x = numpy.array([2], dtype=numpy.int16)
    y = numpy.array([2], dtype=numpy.float32)
    yy = numpy.array([2])
    z = numpy.array([5])
    assert x in space
    assert y not in space
    assert yy in space
    assert z not in space

    space = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
    )
    x = numpy.array([[1, 1], [1, 1]], dtype=numpy.int16)
    y = numpy.array([[1, 1], [1, 1]], dtype=numpy.float32)
    yy = numpy.array([[1, 1], [1, 1]])
    yyy = numpy.array([[1.0, 1.0], [1.0, 1.0]])
    z = numpy.array([[5, 1], [1, 1]], dtype=numpy.float32)
    assert x in space
    assert y in space
    assert yy in space
    assert yyy in space
    assert z not in space
    # [end-space-contains]

    # [start-space-sample]
    f = Space(
        [
            numpy.array([[-2, -2], [-1, -1]], dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
    )
    g = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(31)], dtype=numpy.int16),
        ]
    )
    h = Space([numpy.array([i for i in range(10)], dtype=numpy.int16)])

    f.sample()
    g.sample()
    h.sample()
    # [end-space-sample]

    # [start-space-iter]
    g = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(31)], dtype=numpy.int16),
        ]
    )
    for _i in g:
        print(_i)

    h = Space(
        [
            -numpy.ones((3, 4), dtype=numpy.float32),
            numpy.ones((3, 4), dtype=numpy.float32),
        ]
    )

    for _h in h:
        print(_h)
        for __h in _h:
            print(__h)
    # [end-space-iter]

    # [start-state-example]
    # Continuous substate. Provide Space([low, high]). Value is optional
    x = StateElement(
        values=None,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    # Discrete substate. Provide Space([range]). Value is optional
    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], dtype=numpy.int)]))

    # Define a State, composed of two substates previously defined
    s1 = State(substate_x=x, substate_y=y)

    # Define a super-State that is composed of the State previously defined
    S = State()
    S["substate1"] = s1
    # [end-state-example]

    # -------------- StateElement------------

    # [start-stateelement-init]
    x = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )

    y = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    numpy.array([i for i in range(10)], dtype=numpy.int16),
                    numpy.array([i for i in range(10)], dtype=numpy.int16),
                ]
            )
            for j in range(3)
        ],
        clipping_mode="error",
    )
    # [end-stateelement-init]

    # [start-stateelement-reset]
    x.reset()
    y.reset()
    reset_dic = {"values": [-1 / 2, 2, -2]}
    x.reset(dic=reset_dic)
    reset_dic = {"values": [[0, 0], [10, 10], [5, 5]]}
    try:
        y.reset(dic=reset_dic)
    except StateNotContainedError:
        print("raised error as expected")
    # [end-stateelement-reset]

    # [start-stateelement-iter]
    for _x in x:
        print(_x)

    for _y in y:
        print(_y)
    # [end-stateelement-iter]

    # [start-stateelement-cp]
    x.reset()
    for n, _x in enumerate(x.cartesian_product()):
        # print(n, _x.values)
        print(n, _x)
    y.reset()
    # There are a million possible elements in y, so consider the first subspace only
    for n, _y in enumerate(y[0].cartesian_product()):
        print(n, _y.values)
    # [end-stateelement-cp]

    # [start-stateelement-comp]
    x.reset()
    a = x[0]
    print(x < numpy.array([2, -2, 4]))
    # [end-stateelement-comp]

    y.reset()
    targetdomain = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    -numpy.ones((2, 1), dtype=numpy.float32),
                    numpy.ones((2, 1), dtype=numpy.float32),
                ]
            )
            for j in range(3)
        ],
    )
    res = y.cast(targetdomain)

    # [start-stateelement-cast]
    b = StateElement(
        values=5,
        spaces=Space(
            [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
        ),
    )

    a = StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    import matplotlib.pyplot as plt

    # C2D
    continuous = []
    discrete = []
    for elem in numpy.linspace(-1, 1, 200):
        a["values"] = elem
        continuous.append(a["values"][0].squeeze().tolist())
        discrete.append(a.cast(b, mode="center")["values"][0].squeeze().tolist())

    plt.plot(continuous, discrete, "b*")
    plt.show()

    # D2C

    continuous = []
    discrete = []
    for elem in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        b["values"] = elem
        discrete.append(elem)
        continuous.append(b.cast(a, mode="edges")["values"][0].squeeze().tolist())

    plt.plot(discrete, continuous, "b*")
    plt.show()

    # C2C

    a = StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-2], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=3.5,
        spaces=Space(
            [
                numpy.array([3], dtype=numpy.float32),
                numpy.array([4], dtype=numpy.float32),
            ],
        ),
    )

    c1 = []
    c2 = []
    for elem in numpy.linspace(-2, 1, 100):
        a["values"] = elem
        c1.append(a["values"][0].squeeze().tolist())
        c2.append(a.cast(b)["values"][0].squeeze().tolist())

    plt.plot(c1, c2, "b*")
    plt.show()

    # D2D
    a = StateElement(
        values=5,
        spaces=Space([numpy.array([i for i in range(11)], dtype=numpy.int16)]),
    )
    b = StateElement(
        values=5,
        spaces=Space(
            [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
        ),
    )

    d1 = []
    d2 = []
    for i in range(11):
        a["values"] = i
        d1.append(i)
        d2.append(a.cast(b)["values"][0].squeeze().tolist())

    plt.plot(d1, d2, "b*")
    plt.show()
    # [end-stateelement-cast]

    # [start-stateelement-arithmetic]
    # Neg
    x = StateElement(
        values=numpy.array([[-0.237]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    print(-x)

    # Sum
    x = StateElement(
        values=numpy.array([[-0.237]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    y = StateElement(
        values=numpy.array([[-0.135]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    # adding two simple StateElements
    print(x + y)
    # add with a scalar
    z = -0.5
    print(x + z)

    a = StateElement(
        values=numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )

    print(a + b)
    print(b + a)
    print(a - b)
    print(b - a)
    c = numpy.array([0.12823329, -0.10512559])
    print(a + c)
    print(c + a)
    print(a - c)
    print(c - a)

    # Mul
    a = StateElement(
        values=numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )

    c = 1
    print(a * b)
    print(a * c)
    print(b * a)
    print(c * a)

    # Matmul
    a = StateElement(
        values=numpy.array([[-0.237, 0], [1, 1]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([[-1, -1], [-1, -1]], dtype=numpy.float32),
                numpy.array([[1, 1], [1, 1]], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32).reshape(-1, 1),
                numpy.array([1, 1], dtype=numpy.float32).reshape(-1, 1),
            ]
        ),
    )

    z = numpy.ones((2, 2))

    print(a @ b)
    print(z @ a)
    print(a @ z)
    # [end-stateelement-arithmetic]

    # -------------- State------------

    # [start-state-init]
    x = StateElement(
        values=1,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], dtype=numpy.int)]))

    z = StateElement(
        values=5,
        spaces=Space([numpy.array([i for i in range(10)], dtype=numpy.int)]),
    )

    s1 = State(substate_x=x, substate_y=y, substate_z=z)

    w = StateElement(
        values=numpy.zeros((3, 3)),
        spaces=Space([-3.5 * numpy.ones((3, 3)), 6 * numpy.ones((3, 3))]),
    )
    s1["substate_w"] = w

    xx = StateElement(
        values=numpy.ones((2, 2)),
        spaces=Space([-0.5 * numpy.ones((2, 2)), 0.5 * numpy.ones((2, 2))]),
        clipping_mode="clip",
    )

    yy = StateElement(
        values=None,
        spaces=Space(
            [numpy.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=numpy.int16)]
        ),
    )

    s2 = State(**{"substate_xx": xx, "substate_yy": yy})

    S = State()
    S["substate1"] = s1
    S["substate2"] = s2
    # [end-state-init]

    # [start-state-reset]
    print(S.reset())
    # [end-state-reset]

    # [start-state-filter]
    from collections import OrderedDict

    ordereddict = OrderedDict(
        {"substate1": OrderedDict({"substate_x": 0, "substate_w": 0})}
    )

    ns1 = S.filter("values", filterdict=ordereddict)
    ns2 = S.filter("spaces", filterdict=ordereddict)
    ns5 = S.filter("values")
    ns6 = S.filter("spaces")

    # [end-state-filter]

    # [start-state-serialize]
    print(S.serialize())
    # [end-state-serialize]
