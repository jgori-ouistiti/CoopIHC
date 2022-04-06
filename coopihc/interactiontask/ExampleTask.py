"""
This module provides access to an example subclass of
the InteractionTask class.
"""


import numpy
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.elements import discrete_array_element
from coopihc.interactiontask.InteractionTask import InteractionTask


class ExampleTask(InteractionTask):
    """ExampleTask

    An example algebraic task which a single task state 'x', which finishes when x = 4.

    """

    def __init__(self, *args, **kwargs):

        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task

        super().__init__(*args, **kwargs)

        # Describe the state. Here it is a single item which takes value in [-4, -3, ..., 3, 4]. The StateElement has out_of_bounds_mode = clip, which means that values outside the range will automatically be clipped to fit the space.
        self.state["x"] = discrete_array_element(
            init=0, low=-1, high=4, out_of_bounds_mode="clip"
        )

    def reset(self, dic=None):
        # Always start with state 'x' at 0
        self.state["x"] = 0
        return

    def on_user_action(self, *args, **kwargs):
        # Modify the state in place, adding the user action
        is_done = False
        # self.state["x"] = self.state["x"] + self.user_action
        self.state["x"] += self.user_action

        # Stopping condition, return is_done boolean floag
        if self.state["x"] == 4:
            is_done = True

        reward = -1
        return self.state, reward, is_done

    def on_assistant_action(self, *args, **kwargs):
        is_done = False
        # Modify the state in place, adding the assistant action
        self.state["x"] += self.assistant_action
        # Stopping condition, return is_done boolean floag
        if self.state["x"] == 4:
            is_done = True

        reward = -1
        return self.state, reward, is_done


class CoordinatedTask(InteractionTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state["x"] = discrete_array_element(init=0, low=0, high=9)

    def reset(self, dic=None):
        self.state["x"] = 0
        return

    def on_user_action(self, *args, **kwargs):
        is_done = False

        if self.state["x"] == 9:
            is_done = True

        if self.round_number == 100:
            is_done = True

        reward = -1
        return self.state, reward, is_done

    def on_assistant_action(self, *args, **kwargs):
        is_done = False

        if self.user_action == self.assistant_action:
            self.state["x"] += 1

        reward = -1
        return self.state, reward, is_done
