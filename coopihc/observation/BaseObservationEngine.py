import copy


class BaseObservationEngine:
    """Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    All Observation Engines are subclassed from this main class, but you are really not inheriting much... This is mostly here for potential future changes.

    """

    def __init__(self):
        pass
        # self.type = "base"

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
        :rtype: `State<coopihc.space.State.State`
        """
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        """action

        returns the last action

        :return: last action
        :rtype: `State<coopihc.space.State.State`
        """
        return self.host.policy.action_state["action"]

    @property
    def unwrapped(self):
        return self

    def observe(self, game_state):
        """observe

        Redefine this

        .. warning::

            deepcopy mechanisms is extremely slow

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State`
        :return: observation, obs reward
        :rtype: tuple(`State<coopihc.space.State.State`, float)
        """
        return copy.deepcopy(game_state), 0

    def reset(self):
        """reset

        Empty by default.
        """
        return
