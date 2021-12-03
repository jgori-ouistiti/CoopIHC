from coopihc.observation.BaseObservationEngine import BaseObservationEngine


class ExampleObservationEngine(BaseObservationEngine):
    def __init__(self, observable_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable_state = observable_state

    def observe(self, game_state):
        return game_state[self.observable_state], 0
