from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.base.State import State

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

    # @BaseObservationEngine.get_params
    @BaseObservationEngine.default_value
    def observe(self, game_state=None):
        """observe

        Select only state observable_state.

        :param game_state: game state
        :type game_state: `State<coopihc.base.State.State`
        :return: (observation, obs reward)
        :rtype: tuple(`State<coopihc.base.State.State`, float)
        """
        return (
            State(**{self.observable_state: game_state[self.observable_state]}),
            0,
        )


# [end-obseng-subclass]
