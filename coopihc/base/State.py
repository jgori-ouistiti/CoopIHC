import copy
import json
from tabulate import tabulate
import numpy
import warnings
import itertools

from coopihc.base.StateElement import StateElement
from coopihc.base.utils import (
    NotKnownSerializationWarning,
    StateElementAssignmentWarning,
)
from coopihc.base.Space import CatSet


class State(dict):
    """State

    The container class for States. State subclasses dictionnary and adds a few methods:

        * reset(dic = reset_dic), which passes reset values to the StateElements it holds and triggers their reset method
        * filter(mode=mode, filterdict=filterdict), which filters out the state to extract some information.
        * serialize(), which transforms the state into a format that can be serializable, e.g. to send as a JSON format.

    Initializing a State is straightforward:

    .. code-block:: python

        state = State()
        substate = State()
        substate["x1"] = discrete_array_element(init=1, low=1, high=3)
        substate["x3"] = array_element(
            init=1.5 * numpy.ones((2, 2)), low=numpy.ones((2, 2)), high=2 * numpy.ones((2, 2))
        )

        substate2 = State()
        substate2["y1"] = discrete_array_element(init=1, low=1, high=3)

        state["sub1"] = substate
        state["sub2"] = substate2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        """__eq__

        equality checks on arrays (soft) à la Numpy

        .. code-block:: python

            _example_state = example_game_state()
            obs = {
                "game_info": {"turn_index": numpy.array(0), "round_index": numpy.array(0)},
                "task_state": {"position": numpy.array(2), "targets": numpy.array([0, 1])},
                "user_action": {"action": numpy.array(0)},
                "assistant_action": {"action": numpy.array(2)},
            }
            del _example_state["user_state"]
            del _example_state["assistant_state"]
            assert _example_state == obs
            assert _example_state.equals(obs, mode="soft")

        """
        return self.equals(other, mode="soft")

    def equals(self, other, mode="hard"):
        """equals

        equality checks that also checks for spaces (hard).

        .. code-block:: python

            _example_state = example_game_state()
            obs = {
                "game_info": {"turn_index": numpy.array(0), "round_index": numpy.array(0)},
                "task_state": {"position": numpy.array(2), "targets": numpy.array([0, 1])},
                "user_action": {"action": numpy.array(0)},
                "assistant_action": {"action": numpy.array(2)},
            }
            del _example_state["user_state"]
            del _example_state["assistant_state"]
            assert not _example_state.equals(obs, mode="hard")

        """
        for (key, value), (okey, ovalue) in itertools.zip_longest(
            self.items(), other.items()
        ):
            cond = value.equals(ovalue, mode=mode)
            if not isinstance(cond, bool):
                try:
                    cond = cond.all()
                except:
                    cond = all(cond)
            if not (key == okey and cond):
                return False
        return True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setitem__(self, key, value):
        if isinstance(value, (State, StateElement)):
            if isinstance(value, StateElement):
                try:
                    if self[key].space != value.space:
                        warnings.warn(
                            StateElementAssignmentWarning(
                                f"You are trying to assign StateElement {value} with space {value.space} to a state which has previous StateElement {self[key]} with space {self[key].space}. To suppress this warning, either make sure your assignment is not of type StateElement, or delete the old statelement beforehand if you want it to be replaced"
                            )
                        )
                        self[key][...] = value[...]
                        return
                except KeyError:
                    return super().__setitem__(key, value)
            return super().__setitem__(key, value)
        try:
            self[key][...] = value
            return
        except KeyError:
            return super().__setitem__(key, value)

    def reset(self, dic={}):
        """Initialize the state. See StateElement

        Example usage:

        .. code-block:: python

            # Normal reset
            state.reset()


            # Forced reset
            reset_dic = {
                "sub1": {"x1": 3},
                "sub2": {"y1": 3},
            }
            state.reset(dic=reset_dic)


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
                    "sub1": dict({"x1": 0, "x3": slice(0, 1)}),
                    "sub2": dict({"y1": 0}),
                }
            )

        This will filter out

            * the first component (index 0) for subsubstate x1 in substate sub1,
            * the first and second components for subsubstate x3 in substate sub1,
            * the first component for subsubstate y1 in substate sub2.


        Example usage:

        .. code-block:: python

            # Filter out spaces
            f_state = state.filter(mode="space", filterdict=filterdict)

            # Filter out as arrays
            f_state = state.filter(mode="array", filterdict=filterdict)

            # Filter out as StateElement
            f_state = state.filter(mode="stateelement", filterdict=filterdict)

            # Get spaces
            f_state = state.filter(mode="space")

            # Get arrays
            f_state = state.filter(mode="array")

            # Get Gym Compatible arrays
            f_state = state.filter(mode="array-Gym")


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
        for key, value in filterdict.items():
            if isinstance(self[key], State):
                new_state[key] = self[key].filter(mode=mode, filterdict=value)
            elif isinstance(self[key], StateElement):
                # to make S.filter("values", S) possible.
                # Warning: values == filterdict[key] != self[key]
                if isinstance(value, StateElement):
                    try:
                        value = slice(0, len(value), 1)
                    except TypeError:  # Deal with 0-D arrays
                        value = slice(0, 1, 1)  # Ellipsis  # slice(0, 1, 1)
                if mode == "space":
                    _SEspace = self[key].space
                    new_state[key] = _SEspace
                elif mode == "array":
                    new_state[key] = (self[key][value]).view(numpy.ndarray)
                elif mode == "array-Gym":
                    v = (self[key][value]).view(numpy.ndarray)
                    if isinstance(self[key].space, CatSet):
                        new_state[key] = int(v)
                    elif self[key].space.shape == ():
                        new_state[key] = numpy.atleast_1d(v)
                    else:
                        new_state[key] = v

                elif mode == "stateelement":
                    new_state[key] = self[key][value, {"space": True}]
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

            state.serialize()

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

            nline = 1  # deal with empty substates

            for nline, line in enumerate(tab):
                if isinstance(value, State) and nline != 0:
                    key = " "
                line.insert(0, key)
            table.extend(tab)
            line_no += (n + 1) * (nline + 1)
        return (table, line_no)

    def __str__(self):
        return tabulate(self._tabulate()[0])
