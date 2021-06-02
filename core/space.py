from collections import OrderedDict
import copy
import gym
import numpy
from core.helpers import flatten
import itertools

def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix):]

class Array:
    def __init__(self, array, low = None, high = None):
        self.array_size = array.shape

        if low is None:
            self.low = numpy.full(array.shape, -numpy.inf)
        elif self.low.shape == array.shape:
            self.low = low
        else:
            raise ValueError("Shape mismatch between array and lower bound with shapes {}, {}".format(array.shape.__str__(), low.shape.__str__()))

        if high is None:
            self.high = numpy.full(array.shape, numpy.inf)
        elif self.high.shape == array.shape:
            self.high = high
        else:
            raise ValueError("Shape mismatch between array and lower bound with shapes {}, {}".format(array.shape.__str__(), high.shape.__str__()))


    def sample(self):
        return 2*numpy.random.sample(self.array_size) -1

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
            if hv == [None]:
                values.append(v)
            else:
                if hv[v] is not None:
                    values.append(hv[v])
                else:
                    values.append(v)
        return values



    def check_values(self, values):
        values = flatten([values])

        if values == [None]:
            if not(self.values is None or self.values == [None]):
                return self.values
            if self.values == [None] and not(self.spaces is None or self.spaces == [None]):
                return [None for space in self.spaces]
            else:
                return [None]
        # values = flatten([values])

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
                            raise ValueError('You are assigning an invalid value: value number {} ({}) is not contained in {}'.format(str(n), str(value), str(space)))
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

    def __repr__(self):
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



class State(OrderedDict):
    def reset(self, dic = {}):
        for key, value in self.items():
            reset_dic = dic.get(key)
            if reset_dic is None:
                reset_dic = {}
            value.reset(reset_dic)

    # get around the huge overhead of the regular deepcopy method. This is probably more  like copy than deepcopy, but it is enough for what is needed for now.
    def __deepcopy__(self, memodict={}):
        copy_object = State()
        copy_object.update(self)
        return copy_object


    def __repr__(self):
        _str = 'State 0x{} of type {}'.format(id(self), self.__class__)
        for key, value in self.items():
            _str += "\n{}:   {}".format(key, value.__repr__())

        return _str

    # def __repr__(self):
    #     """ Redefine the default print behavior to have a pretty print format of the game state. Can only be called after the bundle has been reset.
    #
    #
    #     :meta private:
    #     """
    #     _str = "{}  {:>20}  {:<10} {:<10}\n".format("index", "substate", "value", 'real_value')
    #     l = 0
    #     for key, value in self.items():
    #         for skey, svalue in value.items():
    #             for nitem, item in enumerate(svalue[0]):
    #                 try:
    #                     real_value = svalue[2][nitem][item]
    #                 except (TypeError, IndexError):
    #                     real_value = ""
    #                 if isinstance(item, float):
    #                     _str += "{}  {:>30}  {:<10.3f} {:<10}\n".format(l, key + '|' + str(skey) + '|' + str(nitem),  item, real_value)
    #                 else:
    #                     _str += "{}  {:>30}  {:<10} {:<10}\n".format(l, key + '|' + str(skey) + '|' + str(nitem),  item.__str__(), real_value)
    #                 l+=1
    #     return "{}".format(_str)


class _State(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, item):
        if isinstance(item, State):
            super().__setitem__(key, item)
            return

        if '_value_' in key:
            key = remove_prefix(key, '_value_')
            item = (item, None, None)


        if len(item) == 3:
            value, space, possible_values = item
        else:
            raise TypeError('Assigning an item in State requires you to pass a list [value, spacetype, possible_values], where spacetype and possible_values may be None. Alternatively you can set the value of State with by indexing using the special syntax _value_ e.g. State["_value_x"] = [0] will set the value of substate x to [0].')
        # elif len(item) == 1:
        #     value, space, possible_values = item, None, None
        values = self.get(key)
        if values is not None:
            if value is None:
                value = values[0]
            # wrap the action in a list if needed
            if isinstance(value, (int, float, numpy.integer, numpy.float, type(None))):
                value = [value]
            # if the state value list is nested in a list, unwrap one level
            if isinstance(value[0], list):
                value = value[0]
            # If the state value is an array nested in a list, unwrap one level
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], numpy.ndarray):
                    value = value[0]
            if space is None:
                space = values[1]
            if possible_values is None:
                possible_values = values[2]
        else:
            if isinstance(value, (int, float, numpy.integer, numpy.float, type(None))):
                value = [value]
        super().__setitem__(key, [value, space, possible_values])

    def __getitem__(self, key):
        if isinstance(key, str):
            if '_value_' in key:
                return self.__getitem__(remove_prefix(key, '_value_'))[0]
            return super().__getitem__(key)
        else:
            str_key = list(self.keys())[key]
            return super().__getitem__(str_key)

    def reset(self, **kwargs):
        if not kwargs:
            for key, value in self.items():
                self['_value_{}'.format(key)] = [v.sample() for v in value[1]]
        else:
            raise NotImplementedError

    def __str__(self):
        """ Redefine the default print behavior to have a pretty print format of the game state. Can only be called after the bundle has been reset.
        only works for bundle for the moment


        :meta private:
        """
        _str = "{}  {:>20}  {:<10} {:<10}\n".format("index", "substate", "value", 'real_value')
        l = 0
        for key, value in self.items():
            for skey, svalue in value.items():
                for nitem, item in enumerate(svalue[0]):
                    try:
                        real_value = svalue[2][nitem][item]
                    except (TypeError, IndexError):
                        real_value = ""
                    if isinstance(item, float):
                        _str += "{}  {:>30}  {:<10.3f} {:<10}\n".format(l, key + '|' + str(skey) + '|' + str(nitem),  item, real_value)
                    else:
                        _str += "{}  {:>30}  {:<10} {:<10}\n".format(l, key + '|' + str(skey) + '|' + str(nitem),  item.__str__(), real_value)
                    l+=1
        return "{}".format(_str)

# task = State()
# task['x'] = [[None], [gym.spaces.Box(-1,1, shape = (1,))],  [None]]
# operator = State()
# operator['xhat'] = [[None], [gym.spaces.Box(-2,2, shape = (1,))], [None]]
#
# u  = State()
# u['y'] = [[None], [gym.spaces.Box(-1,1, shape = (1,))],  [None]]
#
#
# bundle = State()
# bundle['task_state'] = task
# bundle['operator_state'] = operator
# bundle['u_state'] = u
#
#
# packagedoperator = State()
# packagedoperator['operator_state'] =
