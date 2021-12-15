import copy
import numpy
import json
import sys
import itertools
import warnings

from coopihc.helpers import flatten
from coopihc.space.Space import Space
from coopihc.space.utils import (
    SpaceLengthError,
    StateNotContainedError,
    StateNotContainedWarning,
    SpacesNotIdenticalError,
)


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
    :param str typing_priority: What to do when the type of the space does not match the type of the value. If 'space', convert the value to the dtype of the space, if 'value' the converse. Defaults to space.

    """

    __array_priority__ = 1  # to make __rmatmul__ possible with numpy arrays
    __precedence__ = ["error", "warning", "clip"]

    def __init__(
        self,
        *args,
        values=None,
        spaces=None,
        clipping_mode="warning",
        typing_priority="space",
        **kwargs
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
        # if key == "values":
        #     # setattr(self, key, self._preprocess_values(item))
        #     setattr(self, key, item)
        # elif key == "spaces":
        #     setattr(self, key, self._preprocess_spaces(item))
        # elif key == "clipping_mode":
        #     setattr(self, key, item)
        if key in ["values", "spaces", "clipping_mode"]:
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

    # Numpy switch, Not working, see Issue 42
    # def __getattr__(self, key):
    #     _np = sys.modules["numpy"]
    #     print(_np, key, getattr(_np, key))
    #     if hasattr(_np, key):
    #         # adapted from https://stackoverflow.com/questions/13776504/how-are-arguments-passed-to-a-function-through-getattr
    #         def wrapper(*args, **kwargs):
    #             return getattr(_np, key)(self.values, *args, **kwargs)

    #         return wrapper

    #     raise AttributeError(
    #         "StateElement does not have attribute {}. Tried to fall back to numpy but it also did not have this attribute".format(
    #             key
    #         )
    #     )

    # Comparison
    def _comp_preface(self, other):
        if isinstance(other, StateElement):
            other = other["values"]
        if not isinstance(other, list):
            other = flatten(other)
        return self, other

    def __eq__(self, other):
        if self.spaces != other.spaces:
            return False
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.equal(s, o) for s, o in zip(self["values"], other)]
        ).all()

    def equal(self, other, *args, **kwargs):
        self, other = self._comp_preface(other)
        return numpy.array(
            [numpy.equal(s, o, *args, **kwargs) for s, o in zip(self["values"], other)]
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
            except TypeError as msg:
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
                    warnings.warn(
                        StateNotContainedWarning(
                            "Warning: Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})".format(
                                str(v), type(v), str(s), str(s.low), str(s.high)
                            )
                        )
                    )
                elif self.clipping_mode == "clip":
                    v = self._clip(v, s)
                    # v = self._clip_1d(v, s)
                else:
                    pass
            # should not be needed, but hack for now. The problem was triggered when running stable_baselines' check_env on the env created by GymTrain. I believe the problem comes when creating vectorized envs, but could not point to it. Here we force the shape of the values, which seems to work for now, but at the cost of extra operations.
            # ===================
            if isinstance(v, numpy.ndarray):
                v = numpy.atleast_2d(v.squeeze())
            # =====================
            values[ni] = v
            # should not be needed, but hack for now, see above
            # ===================
            values = flatten(values)
            # =====================
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
