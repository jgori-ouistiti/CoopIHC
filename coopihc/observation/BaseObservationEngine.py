import copy
import numpy


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
        :rtype: :py:class:`State <coopihc.space.State.State>`
        """
        try:
            return self.host.inference_engine.buffer[-1]
        except AttributeError:
            return None

    @property
    def action(self):
        """action

        returns the last action

        :return: last action
        :rtype: :py:class:`State<coopihc.space.State.State>`
        """
        try:
            return self.host.policy.action_state["action"]
        except AttributeError:
            return None

    @property
    def unwrapped(self):
        return self

    def observe(self, game_state=None):
        """observe

        Redefine this

        .. warning::

            deepcopy mechanisms is extremely slow

        :param game_state: game state
        :type game_state: :py:class:`State<coopihc.space.State.State>`
        :return: observation, obs reward
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        if game_state is None:
            game_state = self.host.bundle.game_state
        return copy.deepcopy(game_state), 0

    def reset(self):
        """reset

        Empty by default.
        """
        return
