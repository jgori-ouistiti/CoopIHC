from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class NewInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):
        self.state = #
        reward = #
        return self.state, reward