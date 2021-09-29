from collections import OrderedDict
import copy
import gym
import numpy
import json
import sys

numpy.set_printoptions(precision = 3, suppress = True)


from core.helpers import flatten, hard_flatten
import itertools
from tabulate import tabulate
import operator

def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix):]


from core.helpers import isdefined

class Space:
    def __init__(self, array, *args, **kwargs):

        self._cflag = None
        self.rng = numpy.random.default_rng()

        # Deal with variable format input
        if isinstance(array, numpy.ndarray):
            pass
        else:
            for _array in array:
                if not isinstance(_array, numpy.ndarray):
                    raise AttributeError('Input argument array must be or must inherit from numpy.ndarray instance.')
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
            return item in self.range


    @property
    def range(self):
        if self._range is None:
            if self.continuous:
                if not self._array[0].shape == self._array[1].shape:
                    return AttributeError("The low {} and high {} ranges don't have matching shapes".format(self._array[0], self._array[1]))
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
    def N(self):
        if self.continuous:
            return None
        else:
            return len(self.range)


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
            if len(self._array) ==1:
                self._dtype = self._array[0].dtype
            else:
                self._dtype = numpy.find_common_type([ v.dtype for v in self._array], [])
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

    def __init__(self, values = None, spaces = None, clipping_mode = 'warning', typing_priority = 'space'):
        self.clipping_mode = clipping_mode
        self.typing_priority = typing_priority
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


    def __preface(self, other):
        if not isinstance(other, (StateElement, numpy.ndarray)):
            other = numpy.array(other)
        if hasattr(other, 'values'):
            other = other['values']

        _elem = StateElement(
            values = self.values,
            spaces = self.spaces,
            clipping_mode = self.mix_modes(other) )
        return _elem, other


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
        _elem, other = self.__preface(other)
        _elem['values'] = numpy.add(self['values'], other, casting = 'same_kind')
        return _elem

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)


    def __mul__(self, other):
        _elem, other = self.__preface(other)
        _elem['values'] = numpy.multiply(self['values'], other, casting = 'same_kind')

        return _elem

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
        if isinstance(other, StateElement):
            matA = self['values'][0]
            matB = other['values'][0]
        elif isinstance(other, numpy.ndarray):
            matA = self.values[0]
            matB = other
        else:
            raise TypeError('rhs of {}@{} should be either a StateElement containing a numpy.ndarray or a numpy.ndarray')

        values = matA @ matB
        low = self.spaces[0].low[:values.shape[0], :values.shape[1]]
        high = self.spaces[0].high[:values.shape[0], :values.shape[1]]

        return StateElement(    values = values,
                                spaces = Space([low, high]),
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
            values = dic.get('values')
            if values is not None:
                self['values'] = values
            spaces = dic.get('spaces')
            if spaces is not None:
                self['spaces'] = spaces

    def preprocess_spaces(self, spaces):

        spaces = flatten([spaces])
        return spaces

    def preprocess_values(self, values):

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

            if self.typing_priority == 'space':
                v = numpy.array(v).reshape(s.shape).astype(s.dtype)
            elif self.typing_priority == 'value':
                v = numpy.array(v).reshape(s.shape)
                s._dtype = v.dtype
            else:
                raise NotImplementedError

            if v not in s:
                if self.clipping_mode == 'error':
                    raise StateNotContainedError('Instantiated Value {}({}) is not contained in corresponding space {} (low = {}, high = {})'.format(str(v), type(v), str(s), str(s.low), str(s.high)))
                elif self.clipping_mode == 'warning':
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



    def _discrete2continuous(self, other, mode = 'center'):
        values = []
        for sv, ss, os in zip(self['values'], self['spaces'], other['spaces']):
            if not (not ss.continuous and os.continuous):
                raise AttributeError('Use this only to go from a discrete to a continuous space')

            if mode == 'edges':
                N = len(ss.range)
                ls = numpy.linspace(os.low, os.high, N)
                shift = 0
            elif mode == 'center':
                N = len(ss.range) +1
                ls = numpy.linspace(os.low, os.high, N)
                shift = (ls[1]-ls[0])/2

            values.append(ls[list(ss.range).index(sv)] + shift)
        return values

    def _continuous2discrete(self, other, mode = 'center'):
        values = []
        for sv, ss, os in zip(self['values'], self['spaces'], other['spaces']):
            if not (ss.continuous and not os.continuous):
                raise AttributeError('Use this only to go from a continuous to a discrete space')
            _range = ss.high - ss.low

            if mode == 'edges':

                _remainder = (sv-ss.low)%(_range/os.N)
                index = int((sv-ss.low-_remainder)/_range*os.N)
            elif mode == 'center':
                N = os.N -1
                _remainder = (sv-ss.low + (_range/2/N))%(_range/(N))

                index = int((sv-ss.low-_remainder + _range/2/N)/_range*N + 1e-5) # 1e-5 --> Hack to get around floating point arithmetic
            values.append(os.range[index])

        return values

    def _continuous2continuous(self, other):
        values = []
        for sv, ss, os in zip(self['values'], self['spaces'], other['spaces']):
            if not (ss.continuous and os.continuous):
                raise AttributeError('Use this only to go from a continuous to a continuous space')
            s_range = ss.high - ss.low
            o_range = os.high - os.low
            s_mid = (ss.high + ss.low)/2
            o_mid = (os.high + os.low)/2

            values.append((sv - s_mid)/s_range*o_range + o_mid)
            return values

    def _discrete2discrete(self, other):
        values = []
        for sv, ss, os in zip(self['values'], self['spaces'], other['spaces']):
            if ss.continuous or os.continuous:
                raise AttributeError('Use this only to go from a discrete to a discrete space')
            values.append(os.range[ss.range.squeeze().tolist().index(sv)])
            return values


    def cast(self, other, mode = 'center'):
        if not isinstance(other, StateElement):
            raise TypeError("input arg {} must be of type StateElement".format(str(other)))

        values = []
        for s, o in zip(self, other):
            for sv,ss,ov,os in zip(s['values'], s['spaces'], o['values'], o['spaces']):
                if (not ss.continuous and os.continuous):
                    value = s._discrete2continuous(o, mode = mode)
                elif (ss.continuous and os.continuous):
                    value = s._continuous2continuous(o)
                elif (ss.continuous and not os.continuous):
                    value = s._continuous2discrete(o, mode = mode)
                elif (not ss.continuous and not os.continuous):
                    if ss.N == os.N:
                        value = s._discrete2discrete(o)
                    else:
                        raise ValueError('You are trying to match a discrete space to another discrete space of different size.')
                else:
                    raise NotImplementedError
                values.append(value)

        return StateElement(
            values = values,
            spaces = other['spaces'],
            clipping_mode = self.mix_modes(other),
            typing_priority = self.typing_priority
        )



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
        for i, (v, s,  l) in enumerate(zip(*self.flat())):
                table_rows.append([str(i), l, str(v), str(s)])

        _str = tabulate(table_rows, table_header)

        return _str
