import numpy
import copy

from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space



# ============== General Policies ===============


class BasePolicy:
    """Policy to subclass. Provide either an action state used for initialization, or specify action_spaces and action_sets"""

    def __init__(self, *args, **kwargs):
        # If a state is provided, use it; else create one (important not to lose the reference w/r the game_state)
        if args:
            self.action_state = args[0]
        else:
            action_state = State()
            action_state["action"] = StateElement(
                values=None,
                spaces=Space([numpy.array([None], dtype=numpy.object)]),
            )
            self.action_state = action_state
        # if kwargs:
        #     spaces = kwargs.get('action_space')
        #     if spaces is not None:
        #         self.action_state['action']['spaces'] = spaces
        #     set = kwargs.get('action_set')
        #     if set is not None:
        #         self.action_state['action']['possible_values'] = set
        #     values = kwargs.get('action_values')
        #     if values is not None:
        #         self.action_state['action']['values'] = values
        #     clipping_mode = kwargs.get('clipping_mode')
        #     if clipping_mode is not None:
        #         self.action_state['action']['clipping_mode'] = clipping_mode

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
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        return self.action_state["action"]

    @property
    def new_action(self):
        return copy.copy(self.action_state["action"])

    @property
    def unwrapped(self):
        return self

    def reset(self):
        pass

    def sample(self):
        self.action.reset()
        return self.action, 0

    def __repr__(self):
        try:
            return self.action_state.__str__()
        except AttributeError:
            return "Policy--unreadable"
