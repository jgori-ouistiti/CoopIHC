import copy
from xml.dom.minidom import Attr
import numpy

from coopihc.base.State import State


class BaseObservationEngine:
    """Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    All Observation Engines are subclassed from this main class, but you are really not inheriting much... This is mostly here for potential future changes.

    """

    def __init__(self, *args, seed=None, **kwargs):
        self.rng = numpy.random.default_rng(seed)

    def __content__(self):
        """__content__

        Custom class representation

        :return: custom repr
        :rtype: string
        """
        return self.__class__.__name__

    @property
    def observation(self):
        """observation

        returns the last observation

        :return: last observation
        :rtype: :py:class:`State <coopihc.base.State.State>`
        """
        try:
            return self.host.inference_engine.buffer[-1]
        except AttributeError:
            return None

    @property
    def bundle(self):
        try:
            return self.host.bundle
        except AttributeError:
            raise AttributeError(
                "You haven't connected the observation to a user that is connected to a bundle yet."
            )

    @property
    def action(self):
        """action

        returns the last action

        :return: last action
        :rtype: :py:class:`State<coopihc.base.State.State>`
        """
        try:
            return self.host.action
        except AttributeError:
            return None

    @property
    def unwrapped(self):
        return self

    def observe_from_substates(
        self,
        game_info={},
        task_state={},
        user_state={},
        assistant_state={},
        user_action={},
        assistant_action={},
    ):
        game_state = State(
            **{
                "game_info": game_info,
                "task_state": task_state,
                "user_state": user_state,
                "assistant_state": assistant_state,
                "user_action": user_action,
                "assistant_action": assistant_action,
            }
        )
        return self.observe(game_state=game_state)

    def default_value(func):
        """Apply this decorator to use bundle.game_state as default value to observe if game_state = None"""

        def wrapper_default_value(self, game_state=None):
            if game_state is None:
                game_state = self.host.bundle.game_state
            return func(self, game_state=game_state)

        return wrapper_default_value

    @default_value
    def observe(self, game_state=None):
        """observe

        Redefine this

        .. warning::

            deepcopy mechanisms is extremely slow

        :param game_state: game state
        :type game_state: :py:class:`State<coopihc.base.State.State>`
        :return: observation, obs reward
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float)
        """
        return copy.deepcopy(game_state), 0

    def reset(self, random=True):
        """reset _summary_

        Empty by default.

        :param random: whether states internal to the observation engine are reset randomly, defaults to True. Useful in case of subclassing the Observation Engine.
        :type random: bool, optional
        """
        return

    # To be able to inherit these decorators
    # get_params = staticmethod(get_params)
    default_value = staticmethod(default_value)
