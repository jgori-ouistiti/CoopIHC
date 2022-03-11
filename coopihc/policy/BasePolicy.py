import numpy
import copy

from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space


# ============== General Policies ===============


class BasePolicy:
    """BasePolicy

    Base Policy class. Randomly samples from the action state. You have can provide an action state as an argument (args[0]). If no action state is provided, the policy is initialized with an action state with a single 'None' action.
    """

    def __init__(self, *args, action_state=None, **kwargs):

        # If a state is provided, use it; else create one (important not to lose the reference w/r the game_state)

        if action_state is None:
            action_state = State()
            action_state["action"] = StateElement(
                0, Space(numpy.array([0, 1], dtype=numpy.int16), "discrete")
            )

        self.action_state = action_state
        self.host = None

    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    def _bind(self, func, as_name=None):
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(self, self.__class__)
        setattr(self, as_name, bound_method)
        return bound_method

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        """observation

        Return the last observation.

        :return: last observation
        :rtype: `State<coopihc.space.State.State>`
        """
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        """action

        Return the last action.

        :return: last action
        :rtype: `State<coopihc.space.StateElement.StateElement>`
        """
        return self.action_state["action"]

    @property
    def new_action(self):
        """new action (copy)

        Return a copy of the last action.

        :return: last action
        :rtype: `StateElement<coopihc.space.StateElement.StateElement>`
        """
        return copy.copy(self.action_state["action"])

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.action.reset()

    def sample(self, observation=None):
        """sample

        (Randomly) Sample from the policy

        :return: (action, action reward)
        :rtype: (StateElement<coopihc.space.StateElement.StateElement>, float)
        """
        self.action.reset()
        return self.action, 0

    def __repr__(self):
        try:
            return self.action_state.__str__()
        except AttributeError:
            return "Policy--unreadable"
