"""
This module provides access to an example subclass of
the InteractionTask class.
"""


import numpy
from coopihc.space.Space import Space
from coopihc.space.utils import discrete_space
from coopihc.space.StateElement import StateElement
from coopihc.interactiontask.InteractionTask import InteractionTask


class ExampleTask(InteractionTask):
    """ExampleTask

    An example algebraic task which a single task state 'x', which finishes when x = 4.

    """

    def __init__(self, *args, **kwargs):

        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task

        super().__init__(*args, **kwargs)

        # Describe the state. Here it is a single item which takes value in [-4, -3, ..., 3, 4]. The StateElement has out_of_bounds_mode = clip, which means that values outside the range will automatically be clipped to fit the space.
        self.state["x"] = StateElement(
            0,
            discrete_space(
                numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)
            ),
            out_of_bounds_mode="clip",
        )

    def reset(self, dic=None):
        # Always start with state 'x' at 0
        self.state["x"][:] = 0
        return

    def user_step(self, *args, **kwargs):
        # Modify the state in place, adding the user action
        is_done = False
        self.state["x"][:] = self.state["x"] + self.user_action

        # Stopping condition, return is_done boolean floag
        if self.state["x"] == 4:
            is_done = True

        reward = -1
        return self.state, reward, is_done

    def assistant_step(self, *args, **kwargs):
        is_done = False
        # Modify the state in place, adding the assistant action
        self.state["x"][:] = self.state["x"] + self.assistant_action
        # Stopping condition, return is_done boolean floag
        if self.state["x"] == 4:
            is_done = True

        reward = -1
        return self.state, reward, is_done

    def render(self, *args, mode="text"):

        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")
