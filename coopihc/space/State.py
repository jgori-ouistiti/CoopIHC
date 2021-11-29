from collections import OrderedDict
import copy
import json
from tabulate import tabulate
from coopihc.helpers import flatten
from .StateElement import StateElement


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
