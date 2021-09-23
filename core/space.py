from collections import OrderedDict
import copy
import gym
import numpy
import json

numpy.set_printoptions(precision = 3, suppress = True)


from core.helpers import flatten, hard_flatten
import itertools
from tabulate import tabulate
import operator

def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix):]

class Box(gym.spaces.Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Discrete(gym.spaces.Discrete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def low(self):
        return 0

    @property
    def high(self):
        return self.n


from core.helpers import isdefined

class Space:
    def __init__(self, array, *args, **kwargs):

        self._cflag = None
        self.rng = numpy.random.default_rng()

        self._array = array
        self._range = None
        self._shape = None
        self._dtype = None

    def __repr__(self):
        if self.continuous:
            return 'Space[Continuous({}), {}]'.format(self.shape, self.dtype)
        else:
            return 'Space[Discrete({}), {}]'.format(max(self.range.shape), self.dtype)

    def __contains__(self, item):
        ## --- Maybe these checks are too strict---reshapes and recasts could be performed.
        if item.shape != self.shape:
            try:
                item.reshape(self.shape)
            except ValueError:
                return False
        if not numpy.can_cast(item.dtype, self.dtype, 'same_kind'):
            return False
        if self.continuous:
            return (numpy.all(item >= self.low) and numpy.all(item <= self.high))
        else:
            return (item in self.range)


    @property
    def range(self):
        if self._range is None:
            if self.continuous:
                if not self._array[0].shape == self._array[1].shape:
                    return AttributeError("The low {} and high {} ranges don't have matching shapes".format(self._array[0], slef._array[1]))
            self._range = self._array
        return self._range

    @property
    def low(self):
        if self.continuous:
            low = self.range[0]
        else:
            low = min(self.range)
        return low

    @property
    def high(self):
        if self.continuous:
            high = self.range[1]
        else:
            high = max(self.range)
        return high


    @property
    def shape(self):
        if self._shape is None:
            if not self.continuous:
                self._shape = (1,1)
            else:
                self._shape = numpy.atleast_2d(self.low).shape
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            if isinstance(self._array, list):
                self._dtype = numpy.find_common_type([self._array[0].dtype, self._array[1].dtype], [])
            else:
                self._dtype = self._array.dtype
        return self._dtype

    @property
    def continuous(self):
        if self._cflag is None:
            self._cflag = numpy.issubdtype(self.dtype, numpy.inexact)
        return self._cflag



    def sample(self):
        _l = []
        if self.continuous:
            return (self.high - self.low) * self.rng.random(self.shape, dtype = self.dtype) + self.low
        else:
            return self.rng.choice(self.range, size = self.shape, replace = True).astype(self.dtype)






class SpaceNotDefinedError(Exception):
    pass

class StateLengthError(Exception):
    pass

class BadSpaceError(Exception):
    pass

class StateNotContainedError(Exception):
    pass

class StateElement:

    __array_priority__ = 1 # to make __rmatmul__ possible with numpy arrays
    __precedence__ = ['error', 'warning', 'clip']

    def __init__(self, values = None, spaces = None, clipping_mode = 'warning'):
        self.clipping_mode = clipping_mode
        self.__values, self.__spaces = None, None

        self.spaces = spaces
        self.values = values


    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, values):
        self.__values = self.preprocess_values(values)

    @property
    def spaces(self):
        return self.__spaces

    @spaces.setter
    def spaces(self, values):
        self.__spaces = self.preprocess_spaces(values)

    @property
    def possible_values(self):
        return self.__possible_values


    def __iter__(self):
        self.n = 0
        self.max = len(self.spaces)
        return self

    def __next__(self):
        if self.n < self.max:
            result = StateElement(
                values = self.values[self.n],
                spaces = self.spaces[self.n],
                clipping_mode = self.clipping_mode  )
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.spaces)

    def __setitem__(self, key, item):
        if key == 'values':
            setattr(self, key, self.preprocess_values(item))
        elif key == 'spaces':
            setattr(self, key, self.preprocess_spaces(item))
        elif key == 'clipping_mode':
            setattr(self, key, item)
        else:
            raise ValueError('Key should belong to ["values", "spaces", "clipping_mode"]')

    def __getitem__(self, key):
        if key in ["values", "spaces", "clipping_mode"]:
            return getattr(self, key)
        elif isinstance(key, (int, numpy.int)):
            return StateElement(values = self.values[key],
                                spaces = self.spaces[key],
                                clipping_mode = self.clipping_mode )
        else:
            raise NotImplementedError('Indexing only works with keys ("values", "spaces", "clipping_mode") or integers')

    def __neg__(self):
        return StateElement(
            values = [-u for u in self['values']],
            spaces = self.spaces,
            clipping_mode = self.clipping_mode )


    def __add__(self, other):
        if isinstance(other, StateElement):
            other = other['values']

        _elem = StateElement(
            values = self.values,
            spaces = self.spaces,
            clipping_mode = self.mix_modes(other) )

        if not hasattr(other, '__len__'):
            other = [other]
        if len(_elem['values']) == len(other):
            out = [_elem['values'][k] + v for k,v in enumerate(other)]
        elif len(_elem['values']) == 1:
            out = _elem.values[0] + other
        elif len(_elem['values']) != 1 and len(other) == 1:
            out = [v + other[0] for k,v in enumerate(_elem['values'])]
        else:
            out = _elem['values'] + other
        _elem['values'] = out
        return _elem


    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__radd__(-other)

    def __mul__(self, other):
        _copy = copy.deepcopy(self)
        if not hasattr(other, '__len__'):
            other = [other]

        if len(_copy['values']) == 1:
            out = _copy.values[0] * other
        elif len(_copy['values']) == len(other):
            out = [_copy['values'][k] * v for k,v in enumerate(other)]
        elif len(_copy['values']) != 1 and len(other) == 1:
            out = [v * other[0] for k,v in enumerate(_copy['values'])]
        else:
            out = _copy['values'] * other
        # _copy['values'] = self.clip(out)
        _copy['values'] = out
        return _copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return "[StateElement - {}] - Value {} in {}".format(self.clipping_mode, self.values, self.spaces)

    def __repr__(self):
        return 'StateElement([{}] - {},...)'.format(self.clipping_mode, self.values.__repr__())

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


    def __matmul__(self, other):
        """
        Coarse implementation. Does not deal with all cases
        """
        se = copy.copy(self)
        if isinstance(other, StateElement):
            matA = self.values[0]
            matB = other.values[0]
        elif isinstance(other, numpy.ndarray):
            matA = self.values[0]
            matB = other
        else:
            raise TypeError('rhs of {}@{} should be either a StateElement containing a numpy.ndarray or a numpy.ndarray')

        values = matA @ matB
        low = se.spaces[0].low[:values.shape[0], :values.shape[1]]
        high = se.spaces[0].high[:values.shape[0], :values.shape[1]]

        se.spaces = [gym.spaces.Space(
                        range = [low, high],
                        shape = values.shape,
                        dtype = values.dtype
                        )]
        # se.spaces = [gym.spaces.Box(low, high, shape = values.shape)]
        se['values'] = [values]

        return se

    def __rmatmul__(self, other):
        se = copy.copy(self)
        if isinstance(other, StateElement):
            matA = self.values[0]
            matB = other.values[0]
        elif isinstance(other, numpy.ndarray):
            matA = self.values[0]
            matB = other
        else:
            raise TypeError('lhs of {}@{} should be either a StateElement containing a numpy.ndarray or a numpy.ndarray')
        values = matB @ matA
        if values.shape == se.spaces[0].shape:
            se['values'] = [values]
            return se

        else:
            low = -numpy.inf * numpy.ones(values.shape)
            high = numpy.inf * numpy.ones(values.shape)

            se.spaces = [gym.spaces.Space(
                            range = [low, high],
                            shape = values.shape,
                            dtype = values.dtype
                            )]
            se['values'] = [values]
            return se


    def serialize(self):
        v_list = []
        for v in self['values']:
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
        return {"values": v_list, "spaces" : str(self['spaces'])}



    def mix_modes(self, other):
        if hasattr(other, 'clipping_mode'):
            return self.__precedence__[min(self.__precedence__.index(self.clipping_mode), self.__precedence__.index(other.clipping_mode))]
        else:
            return self.clipping_mode



    def cartesian_product(self):
        lists = []
        for value, space in zip(self.values, self.spaces):
            if not space.continuous:
                lists.append(space.range)
            else:
                lists.append([value])
        return [StateElement(
            values = list(element),
            spaces = self.spaces,
            clipping_mode = self.clipping_mode ) for element in itertools.product(*lists) ]


    def reset(self, dic = None):
        if not dic:
            self['values'] = [space.sample() for space in self.spaces]
        else:
            self['values'] = dic.get('values')
            self['spaces'] = dic.get('spaces')

    def preprocess_spaces(self, spaces):
        # if spaces is None:
        #     if self.spaces is not None:
        #         return self.spaces
        spaces = flatten([spaces])
        return spaces

    def preprocess_values(self, values):
        # make sure spaces are defined
        # if flatten([self.spaces]) == None:
        #     raise SpaceNotDefinedError('Values of a state can not be instantiated before having defined the corresponding space')

        values = flatten([values])
        # Allow a single None syntax
        try:
            if values == [None]:
                values = [None for s in self.spaces]
        except ValueError:
            pass
        # Check for length match
        if len(values) != len(self.spaces):
            raise StateLengthError('The size of the values ({}) being instantiated does not match the size of the space ({})'.format(len(values), len(self.spaces)))
        # Make sure values are contained
        for ni, (v,s) in enumerate(zip(values, self.spaces)):
            if v is None:
                continue
            v = numpy.array(v).reshape(s.shape).astype(s.dtype)
            if v not in s:

                if (not s.continuous) or self.clipping_mode == 'error':
                    raise StateNotContainedError('Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})'.format(str(v), type(v), str(s), str(s.low), str(s.high)))
                elif self.clipping_mode == 'warn':
                    print('Warning: Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})'.format(str(v), type(v), str(s), str(s.low), str(s.high)))
                elif self.clipping_mode == 'clip':
                    v = self._clip(v, s)
                else:
                    pass
            values[ni] = v
        return values


    def flat(self):
        return self['values'], self['spaces'], [str(i) for i,v in enumerate(self['values'])]


    def clip(self, values):
        values = flatten([values])
        for n, (value, space) in enumerate(zip(values, self.spaces)):
            values[n] = self._clip(value, space)
        return values

    def _clip(self, value, space):
        if value not in space:
            if space.continuous:
                return numpy.clip( value, space.low, space.high)
            else:
                raise AttributeError('Clip can only work with continuous spaces')



    # works only for 1D
    def _n_to_box(self, other):
        _value = self['values'][0]
        ls = numpy.linspace(other['spaces'][0].low, other["spaces"][0].high, self["spaces"][0].n+1)
        _diff = ls[1]-ls[0]
        value = ls[_value] + _diff/2
        return value
        # works only for 1D
    def _box_to_box(self, other):
        _value = self['values'][0]
        _range_self = self['spaces'][0].high - self['spaces'][0].low
        _range_other = other['spaces'][0].high - other['spaces'][0].low
        value = (_value - _range_self/2)/_range_self*_range_other + _range_other/2
        return value

    def _box_to_n(self, other):
        _value = self['values'][0]
        _range = self['spaces'][0].high - self['spaces'][0].low
        value = int(numpy.floor(  (_value - self['spaces'][0].low)/_range*(other['spaces'][0].n)  ))
        return value

    # works only for 1D
    def cast(self, other):
        if not isinstance(other, StateElement):
            raise TypeError("other {} must be of type StateElement".format(str(other)))

        values = []
        for s,o in zip(self, other):
            if isinstance(s['spaces'][0], gym.spaces.Discrete) and isinstance(o['spaces'][0], gym.spaces.Box):
                value = s._n_to_box(o)
            elif isinstance(s['spaces'][0], gym.spaces.Box) and isinstance(o['spaces'][0], gym.spaces.Box):
                value = s._box_to_box(o)
            elif isinstance(s['spaces'][0], gym.spaces.Box) and isinstance(o['spaces'][0], gym.spaces.Discrete):
                value = s._box_to_n(o)
            elif isinstance(s['spaces'][0], gym.spaces.Discrete) and isinstance(o['spaces'][0], gym.spaces.Discrete):
                if s['spaces'][0].n == o['spaces'][0].n:
                    value = s['values']
                else:
                    raise ValueError('You are trying to match a discrete space to another discrete space of different size.')
            else:
                raise NotImplementedError
            values.append(value)

        _copy = copy.deepcopy(other)
        _copy['values'] = values
        _copy.clipping_mode = self.mix_modes(other)
        return _copy









class State(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, dic = {}):
        for key, value in self.items():
            reset_dic = dic.get(key)
            if reset_dic is None:
                reset_dic = {}
            value.reset(reset_dic)

    def flat(self):
        values = []
        spaces = []
        possible_values = []
        labels = []
        l,k = list(self.values()), list(self.keys())
        for n,item in enumerate(l):
            _values, _spaces, _labels = item.flat()
            values.extend(_values)
            spaces.extend(_spaces)
            labels.extend([k[n] + '|' + label for label in _labels])

        return values, spaces, labels

    def filter(self, mode, filterdict = None):
        new_state = OrderedDict()
        if filterdict is None:
            filterdict = self
        for key, values in filterdict.items():
            if isinstance(self[key], State):
                new_state[key] = self[key].filter(mode, values)
            elif isinstance(self[key], StateElement):
                # to make S.filter("values", S) possible. Warning: Contrary to what one would expect values != self[key]
                if isinstance(values, StateElement):
                    values = slice(0,len(values), 1)
                if mode == 'spaces':
                    new_state[key] = gym.spaces.Tuple(flatten([self[key][mode][values]]))
                else:
                    new_state[key] = self[key][mode][values]
            else:
                new_state[key] = self[key]
        if mode == 'spaces':
            return gym.spaces.Dict(new_state)
        return new_state


    def __content__(self):
        return list(self.keys())

    # Here we override copy and deepcopy simply because there seems to be some overhead in the default deepcopy implementation. Adapted from StateElement code
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
        ret_dict = {}
        for key, value in dict(self).items():
            try:
                value_ = json.dumps(value)
            except TypeError:
                try:
                    value_ = value.serialize()
                except AttributeError:
                    print("warning: I don't know how to serialize {}. I'm sending the whole internal dictionnary of the object. Consider adding a serialize() method to your custom object".format(value.__str__()))
                    value_ = value.__dict__
            ret_dict[key] = value_
        return ret_dict



    def __str__(self):
        """ Print out the game_state and the name of each substate with according indices.
        """

        table_header = ['Index', 'Label', 'Value','Space','Possible Value']
        table_rows = []
        for i, (v, s, p, l) in enumerate(zip(*self.flat())):
                table_rows.append([str(i), l, str(v), str(s), str(p)])

        _str = tabulate(table_rows, table_header)

        return _str
