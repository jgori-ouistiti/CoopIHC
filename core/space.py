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
        super().__init__(self, *args, **kwargs)





class StateElement:

    __array_priority__ = 1 # to make __rmatmul__ possible with numpy arrays

    def __init__(self, values = None, spaces = None, possible_values = None, mode = 'error'):
        self.mode = mode
        self.values, self.spaces, self.possible_values = None, None, None
        self.spaces = self.check_spaces(spaces)
        self.possible_values = self.check_possible_values(possible_values)
        self.values = self.check_values(values)

    def __iter__(self):
        self.n = 0
        self.max = len(self.spaces)
        return self

    def __next__(self):
        if self.n < self.max:
            result = StateElement(values = self.values[self.n], spaces = self.spaces[self.n], possible_values = self.possible_values[self.n], mode = self.mode)
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.spaces)

    def __setitem__(self, key, item):
        if key == 'values':
            setattr(self, key, self.check_values(item))
        elif key == 'spaces':
            setattr(self, key, self.check_spaces(item))
        elif key == 'possible_values':
            setattr(self, key, self.check_possible_values(item))
        else:
            raise ValueError('Key should belong to ["values", "spaces", "possible_values"]')

    def __getitem__(self, key):
        if key in ["values", "spaces", "possible_values"]:
            return getattr(self, key)
        elif key == 'human_values':
            return self.get_human_values()
        elif isinstance(key, (int, numpy.int)):
            return StateElement(values = self.values[key],
                                spaces = self.spaces[key],
                                possible_values = self.possible_values[key], mode = self.mode)

    def __neg__(self):
        return StateElement(values = [-u for u in self['values']], spaces = self.spaces, possible_values = self.possible_values, mode = self.mode)

    def __add__(self, other):
        if isinstance(other, StateElement):
            other = other['values']
        _elem = StateElement(values = self.values, spaces = self.spaces, possible_values = self.possible_values, mode = self.mode)
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
        # _copy['values'] = self.clip(out)
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
        return "[StateElement - {}] - Value {} in {}, with possible_values {}".format(self.mode, self.values, self.spaces, self.possible_values)

    def __repr__(self):
        return 'StateElement([{}] - {},...)'.format(self.mode, self.values.__repr__())

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

        se.spaces = [gym.spaces.Box(low, high, shape = values.shape)]
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

            se.spaces = [gym.spaces.Box(low, high, shape = values.shape)]
            se['values'] = [values]
            return se


    def serialize(self):
        ret_list = []
        for v in self['values']:
            try:
                json.dumps(v)
                ret_list.append(v)
            except TypeError as msg:
                if isinstance(v, numpy.ndarray):
                    ret_list.extend(v.tolist())
                elif isinstance(v, numpy.generic):
                    ret_list.append(v.item())
                else:
                    print(v, type(v))
                    raise TypeError("".format(msg))
        return ret_list


    def cartesian_product(self):
        lists = []
        for value, space, possible_value in zip(self.values, self.spaces, self.possible_values):
            if isinstance(space, gym.spaces.Discrete):
                lists.append(list(range(space.n)))
            elif isinstance(space, gym.spaces.Box):
                lists.append([value])
        return [StateElement(values = list(element), spaces = self.spaces, possible_values = self.possible_values, mode = self.mode) for element in itertools.product(*lists)]


    def reset(self, dic = None):
        if not dic:
            self['values'] = [space.sample() for space in self.spaces]
        else:
            self['values'] = dic.get('values')
            self['spaces'] = dic.get('spaces')
            self['possible_values'] = dic.get('possible_values')


    def get_human_values(self):
        values = []
        for n, (v, hv) in enumerate(zip(self.values, self.possible_values)):
            if v is None:
                values.append(v)
            elif hv == [None]:
                values.append(v)
            else:
                if hv[v] is not None:
                    values.append(hv[v])
                else:
                    values.append(v)
        return values



    def check_values(self, values):
        values = flatten([values])

        try:
            if values == [None]:
                if not(self.values is None or self.values == [None]):
                    return self.values
                if self.values == [None] and not(self.spaces is None or self.spaces == [None]):
                    return [None for space in self.spaces]
                else:
                    return [None]
        except ValueError:
            pass

        if self.spaces is not None and self.spaces != [None]:
            # check whether values has the same number of elements as spaces
            if len(values) != len(self.spaces):
                raise ValueError('You are assigning a value of length {:d}, which mismatches the length {:d} of the space'.format(len(values), len(self.spaces)))
            # check whether values conform to the spaces
            for n, (value, space) in enumerate(zip(values, self.spaces)):
                if isinstance(space, gym.spaces.Discrete):
                    low = 0
                    high = space.n
                elif isinstance(space, gym.spaces.Box):
                    low = space.low
                    high = space.high
                try:
                    if value is None:
                        pass
                    else:
                        if value not in space:
                            if self.mode == 'error':
                                raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {} (low: {}, high: {})'.format(str(n), str(value), str(space), str(low), str(high)))
                            elif self.mode == 'warn':
                                print('Warning: You are assigning an invalid value: value number {} ({}) is not contained in {} (low: {}, high: {})'.format(str(n), str(value), str(space), str(low), str(high)))
                            elif self.mode == 'clip':
                                values[n] = self._clip(value, space)
                            else:
                                pass

                except AttributeError:
                    if numpy.array(value).reshape(space.shape) in space:
                        values[n] = numpy.array(value).reshape(space.shape)
                    else:
                        raise ValueError('AttributeError triggered: You are assigning an invalid value: value number {} ({}) is not contained in {}'.format(str(n), str(value), str(space)))
        if self.possible_values is not None and len(values) != len(self.possible_values):
            raise ValueError('You are assigning a value of length {}, which mismatches the length {} of the possible values'.format(len(value), len(self.possible_values)))
        return values

    def check_spaces(self, spaces):
        if spaces is None:
            if self.spaces is None:
                return [None]
            else:
                return self.spaces
        spaces = flatten([spaces])
        if (self.values is not None) and (self.values != [None]):
            if len(self.values) != len(spaces):
                raise ValueError('You are assigning a space of length {:d}, which mismatches the length {:d} of the value'.format(len(spaces), len(self.values)))
            for n, (value, space) in enumerate(zip(self.values, spaces)):
                try:
                    if value not in space:
                        raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {}'.format(str(n), str(value), str(space)))
                except AttributeError:
                    raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {}'.format(str(n), str(value), str(space)))

        if self.possible_values is not None and len(spaces) != len(self.possible_values):
            raise ValueError('You are assigning a space of length {:d}, which mismatches the length {:d} of the possible values'.format(len(space), len(self.possible_values)))
        return spaces

    def check_possible_values(self, possible_values):
        if possible_values is None:
            if self.possible_values is not None:
                return self.possible_values
            elif self.possible_values is None and self.spaces is not None:
                return [[None] for space in self.spaces]
            else:
                return [None]

        if not isinstance(possible_values, list):
            raise TypeError('Possible values of type {} should be of type list'.format(type(possible_values)))

        if self.spaces is not None and len(self.spaces) != len(possible_values):
            if len(possible_values) != 1 and len(self.spaces) == 1:
                possible_values = [possible_values]
                self.check_possible_values(possible_values)
            else:
                raise ValueError('You are assigning possible values of length {:d}, which mismatches the length {:d} of the space'.format(len(possible_values), len(self.spaces)))

        if self.values is not None and len(self.values)!= len(possible_values):
            raise ValueError('You are assigning possible values of length {:d}, which mismatches the length {:d} of the values'.format(len(possible_values), len(self.spaces)))
        return possible_values


    def flat(self):
        return self['values'], self['spaces'], self['possible_values'], [str(i) for i,v in enumerate(self['values'])]

    def clip(self, values):
        values = flatten([values])
        for n, (value, space) in enumerate(zip(values, self.spaces)):
            values[n] = self._clip(value, space)
        return values

    def _clip(self, value, space):
        if value not in space:
            if isinstance(space, gym.spaces.Box):
                return numpy.clip( value, space.low, space.high)
            elif isinstance(space, gym.spaces.Discrete):
                return max(0,min(space.n -1, value))
            else:
                raise NotImplementedError



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
            _values, _spaces, _possible_values, _labels = item.flat()
            values.extend(_values)
            spaces.extend(_spaces)
            possible_values.extend(_possible_values)
            labels.extend([k[n] + '|' + label for label in _labels])
        return values, spaces, possible_values, labels
        # return hard_flatten(values), spaces, possible_values, labels

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
            print(key)
            try:
                value_ = json.dumps(value)
            except TypeError:
                try:
                    value_ = value.serialize()
                except AttributeError:
                    print("warning: I don't know how to serialize {}. I'm sending the whole internal dictionnary of the object. Consider adding a _serialize() method to your custom object".format(value.__str__()))
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
