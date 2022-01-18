from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.space.State import State

# [start-obseng-subclass]
class ExampleObservationEngine(BaseObservationEngine):
    """ExampleObservationEngine

    A simple example where the engine is only able to see a particular state (observable_state).

    :param observable_state: only state that can be observed
    :type observable_state: string
    """

    def __init__(self, observable_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable_state = observable_state

    def observe(self, game_state=None):
        """observe

        Select only state observable_state.

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State`
        :return: (observation, obs reward)
        :rtype: tuple(`State<coopihc.space.State.State`, float)
        """
        game_state = super().observe(game_state=game_state)
        return State(**{self.observable_state: game_state[self.observable_state]}), 0


# [end-obseng-subclass]
