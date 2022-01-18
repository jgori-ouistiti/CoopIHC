import copy
import json
from tabulate import tabulate
import numpy
import warnings
import itertools

from coopihc.helpers import flatten
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import NotKnownSerializationWarning


class State(dict):
    """State

    The container class for States. State subclasses dictionnary and adds a few methods:

        * reset(dic = reset_dic), which passes reset values to the StateElements it holds and triggers their reset method
        * filter(mode=mode, filterdict=filterdict), which filters out the state to extract some information.
        * serialize(), which transforms the state into a format that can be serializable, e.g. to send as a JSON format.

    Initializing a State is straightforward:

    .. code:block:: python

        state = State()
        substate = State()

        substate["x1"] = StateElement(1, discrete_space([1, 2, 3]))
        substate["x2"] = StateElement(
            [1, 2, 3],
            multidiscrete_space(
                [
                    [0, 1, 2],
                    [1, 2, 3],
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                ]
            ),
        )
        substate["x3"] = StateElement(
            1.5 * numpy.ones((3, 3)),
            continuous_space(numpy.ones((3, 3)), 2 * numpy.ones((3, 3))),
        )

        substate2 = State()
        substate2["y1"] = StateElement(1, discrete_space([1, 2, 3]))
        substate2["y2"] = StateElement(
            [1, 2, 3],
            multidiscrete_space(
                [
                    [0, 1, 2],
                    [1, 2, 3],
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                ]
            ),
        )
        state["sub1"] = substate
        state["sub2"] = substate2

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        for (key, value), (okey, ovalue) in itertools.zip_longest(
            self.items(), other.items()
        ):
            cond = value == ovalue
            if not isinstance(cond, bool):
                try:
                    cond = cond.all()
                except:
                    cond = all(cond)
            if not (key == okey and cond):
                return False
        return True

    def reset(self, dic={}):
        """Initialize the state. See StateElement

        Example usage:

        .. code-block:: python

        s = State()
        s["x"] = StateElement(1, autospace([1, 2, 3]))
        s["y"] = StateElement(
            0 * numpy.ones((2, 2)), autospace(-numpy.ones((2, 2)), numpy.ones((2, 2)))
        )

        # Normal reset
        s.reset()
        assert s["x"] in autospace([1, 2, 3])
        assert s["y"] in autospace(-numpy.ones((2, 2)), numpy.ones((2, 2)))

        # Forced reset
        reset_dic = {"x": 3, "y": numpy.zeros((2, 2))}
        s.reset(reset_dic)
        assert s["x"] == 3
        assert (s["y"] == numpy.zeros((2, 2))).all()


        """
        for key, value in self.items():
            reset_dic = dic.get(key)
            value.reset(reset_dic)

    def filter(self, mode="array", filterdict=None):
        """Extract some part of the state information

        An example for filterdict's structure is as follows:

        .. code-block:: python

            filterdict = dict(
                {
                    "sub1": dict({"x1": 0, "x2": slice(0, 2)}),
                    "sub2": dict({"y2": 2}),
                }
            )

        This will filter out

            * the first component (index 0) for subsubstate x1 in substate sub1,
            * the first and second components for subsubstate x2 in substate sub1,
            * the third component for subsubstate y2 in substate sub2.


        Example usage:

        .. code-block:: python

            # Filter out the spaces

            f_state = state.filter(mode="spaces", filterdict=filterdict)
            assert f_state == {
                "sub1": {
                    "x1": Space(numpy.array([1, 2, 3]), "discrete", contains="soft"),
                    "x2": Space(
                        [numpy.array([0, 1, 2]), numpy.array([1, 2, 3])],
                        "multidiscrete",
                        contains="soft",
                    ),
                },
                "sub2": {"y2": Space(numpy.array([0, 1, 2, 3]), "discrete", contains="soft")},
            }

            # Filter out as arrays

            f_state = state.filter(mode="array", filterdict=filterdict)

            # Filter out as StateElements

            f_state = state.filter(mode="stateelement", filterdict=filterdict)

            # Extract spaces for all components
            f_state = state.filter(mode="spaces")

            # Extract arrays for all components
            f_state = state.filter(mode="array")


        :param mode: "array" or "spaces" or "stateelement", defaults to "array". If "stateelement", returns a dictionnary with the selected stateelements. If "spaces", returns the same dictionnary, but with only the spaces (no array information). If "array", returns the same dictionnary, but with only the value arrays (no space information).
        :type mode: str, optional
        :param filterdict: the dictionnary that indicates which components to filter out, defaults to None
        :type filterdict: dictionnary, optional
        :return: The dictionnary with the filtered state
        :rtype: dictionnary
        """

        new_state = {}
        if filterdict is None:
            filterdict = self
        for key, values in filterdict.items():
            if isinstance(self[key], State):
                new_state[key] = self[key].filter(mode, values)
            elif isinstance(self[key], StateElement):
                # to make S.filter("values", S) possible.
                # Warning: values == filterdict[key] != self[key]
                if isinstance(values, StateElement):
                    values = slice(0, len(values), 1)
                if mode == "spaces":
                    _SEspace = self[key].spaces
                    if _SEspace.space_type == "discrete":
                        new_state[key] = _SEspace
                    else:
                        new_state[key] = _SEspace[values]
                elif mode == "array":
                    new_state[key] = (self[key][values]).view(numpy.ndarray)
                elif mode == "stateelement":
                    new_state[key] = self[key][values, {"spaces": True}]
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
        """Makes the state serializable.

        .. code-block:: python

            assert state.serialize() == {
                "sub1": {
                    "x1": {
                        "values": [1],
                        "spaces": {
                            "array_list": [1, 2, 3],
                            "space_type": "discrete",
                            "seed": None,
                            "contains": "soft",
                        },
                    },
                    "x2": {
                        "values": [[1], [2], [3]],
                        "spaces": {
                            "array_list": [[0, 1, 2], [1, 2, 3], [0, 1, 2, 3]],
                            "space_type": "multidiscrete",
                            "seed": None,
                            "contains": "soft",
                        },
                    },
                    "x3": {
                        "values": [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
                        "spaces": {
                            "array_list": [
                                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                            ],
                            "space_type": "continuous",
                            "seed": None,
                            "contains": "soft",
                        },
                    },
                },
                "sub2": {
                    "y1": {
                        "values": [1],
                        "spaces": {
                            "array_list": [1, 2, 3],
                            "space_type": "discrete",
                            "seed": None,
                            "contains": "soft",
                        },
                    },
                    "y2": {
                        "values": [[1], [2], [3]],
                        "spaces": {
                            "array_list": [[0, 1, 2], [1, 2, 3], [0, 1, 2, 3]],
                            "space_type": "multidiscrete",
                            "seed": None,
                            "contains": "soft",
                        },
                    },
                },
            }

        :return: serializable dictionnary
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
                    warnings.warns(
                        NotKnownSerializationWarning(
                            "warning: I don't know how to serialize {}. I'm sending the whole internal dictionnary of the object. Consider adding a serialize() method to your custom object".format(
                                value.__str__()
                            )
                        )
                    )
                    value_ = value.__dict__
            ret_dict[key] = value_
        return ret_dict

    def _tabulate(self):
        """_tabulate

        See __str__ for usage

        """
        table = []
        line_no = 0
        for n, (key, value) in enumerate(self.items()):
            tab, tablines = value._tabulate()
            for nline, line in enumerate(tab):
                if isinstance(value, State) and nline != 0:
                    key = " "
                line.insert(0, key)
            table.extend(tab)
            line_no += (n + 1) * (nline + 1)
        return (table, line_no)

    def __str__(self):
        return tabulate(self._tabulate()[0])
