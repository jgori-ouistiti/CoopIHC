import copy


class BaseObservationEngine:
    """Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    The only requirement for an Observation Engine is that it has a type (either base, rule, process) and that it has a function called observe with the signature below.

    All Observation Engines are subclassed from this main class, but you are really not inheriting much... This is mostly here for potential future changes.

    :meta public:
    """

    def __init__(self):
        pass
        # self.type = "base"

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        return self.host.policy.action_state["action"]

    @property
    def unwrapped(self):
        return self

    def observe(self, game_state):
        return copy.deepcopy(game_state), 0

    def reset(self):
        return
