from collections import OrderedDict
import copy
import gym
import numpy

numpy.set_printoptions(precision = 3, suppress = True)

from core.helpers import flatten, hard_flatten
import itertools
from tabulate import tabulate


def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix):]


class StateElement:
    def __init__(self, values = None, spaces = None, possible_values = None):
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
            result = StateElement(values = self.values[self.n], spaces = self.spaces[self.n], possible_values = self.possible_values[self.n])
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.spaces)

    def cartesian_product(self):
        lists = []
        for value, space, possible_value in zip(self.values, self.spaces, self.possible_values):
            if isinstance(space, gym.spaces.Discrete):
                lists.append(list(range(space.n)))
            elif isinstance(space, gym.spaces.Box):
                lists.append([value])
        return [StateElement(values = list(element), spaces = self.spaces, possible_values = self.possible_values) for element in itertools.product(*lists)]


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
            # print('warning: you might want to check whether the values {} have been initialized correctly in {}'.format(values.__str__(), self.__repr__() ) )

        if self.spaces is not None and self.spaces != [None]:
            # check whether values has the same number of elements as spaces
            if len(values) != len(self.spaces):
                raise ValueError('You are assigning a value of length {:d}, which mismatches the length {:d} of the space'.format(len(values), len(self.spaces)))
            # check whether values conform to the spaces
            for n, (value, space) in enumerate(zip(values, self.spaces)):
                try:
                    if value is None:
                        pass
                    else:
                        if value not in space:# and value is not None:
                            raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {} (low: {}, high: {})'.format(str(n), str(value), str(space), str(space.low), str(space.high)))
                except AttributeError:
                    if numpy.array(value).reshape(space.shape) in space:
                        values[n] = numpy.array(value).reshape(space.shape)
                    else:
                        raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {}'.format(str(n), str(value), str(space)))
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
            if value not in space:
                if isinstance(space, gym.spaces.Box):
                    values[n] = numpy.clip( value, space.low, space.high)
                elif isinstance(space, gym.spaces.Discrete):
                    values[n] = max(0,min(space.n -1, value))
        return values



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
    def cast(self, other, inplace = False):
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
        if inplace:
            other['values'] = values
            return
        else:
            _copy = copy.deepcopy(other)
            _copy['values'] = values
            return _copy





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
                                possible_values = self.possible_values[key])

    def __add__(self, other):
        _copy = copy.deepcopy(self)
        if not hasattr(other, '__len__'):
            other = [other]

        if len(_copy['values']) == 1:
            out = _copy.values[0] + other
        elif len(_copy['values']) == len(other):
            out = [_copy['values'][k] + v for k,v in enumerate(other)]
        elif len(_copy['values']) != 1 and len(other) == 1:
            out = [v + other[0] for k,v in enumerate(_copy['values'])]
        else:
            out = _copy['values'] + other
        _copy['values'] = self.clip(out)
        return _copy

    def __radd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        raise NotImplementedError

    def __str__(self):
        _str = '\n'
        _str += 'value:\t'
        if self.values:
            _str += str(self.values) + '\n'
        else:
            _str += 'None\n'

        _str += 'spaces:\t'
        if self.spaces:
            _str += str(self.spaces) + '\n'
        else:
            _str += 'None\n'

        _str += 'possible values:\t'
        if self.possible_values:
            _str += str(self.possible_values) + '\n'
        else:
            _str += 'None\n'

        return _str

    def __repr__(self):
        return 'StateElement({},...)'.format(self.values.__repr__())

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

    def filter(self, mode, ordereddict):
        new_state = OrderedDict()
        if ordereddict is None:
            ordereddict = self
        for key, values in ordereddict.items():
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


    def __str__(self):
        """ Print out the game_state and the name of each substate with according indices.
        """

        table_header = ['Index', 'Label', 'Value','Space','Possible Value']
        table_rows = []
        for i, (v, s, p, l) in enumerate(zip(*self.flat())):
                table_rows.append([str(i), l, str(v), str(s), str(p)])

        _str = tabulate(table_rows, table_header)


        return _str
