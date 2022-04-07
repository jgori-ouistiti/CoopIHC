from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine

# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)
class DualInferenceEngine(BaseInferenceEngine):
    """BaseInferenceEngine

    The base Inference Engine from which other engines can be defined. This engine does nothing but return the same state. Any new inference method can subclass ``InferenceEngine`` to have a buffer and ``add_observation`` method (required by the bundle)

    :param buffer_depth: number of observations that are stored, defaults to 1
    :type buffer_depth: int, optional
    """

    """"""

    def __init__(
        self,
        primary_inference_engine,
        dual_inference_engine,
        primary_kwargs={},
        dual_kwargs={},
        order="primary-first",
        **kwargs
    ):
        self.order = order
        self._mode = "primary"
        if type(primary_inference_engine).__name__ == "type":
            self.primary_engine = primary_inference_engine(**primary_kwargs)
        else:
            self.primary_engine = primary_inference_engine

        if type(dual_inference_engine).__name__ == "type":
            self.dual_engine = dual_inference_engine(**dual_kwargs)
        else:
            self.dual_engine = dual_inference_engine

        super().__init__(**kwargs)

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        self._host = value
        self.primary_engine.host = value
        self.dual_engine.host = value

    # Set mode to read-only
    @property
    def mode(self):
        return self._mode

    @property
    def buffer(self):
        if self.mode == "primary":
            return self.primary_engine.buffer
        else:
            return self.dual_engine.buffer

    @buffer.setter
    def buffer(self, value):
        if self.mode == "primary":
            self.primary_engine.buffer = value
        else:
            self.dual_engine.buffer = value

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):
        if self._mode == "primary":
            state, primary_reward = self.primary_engine.infer(
                agent_observation=agent_observation
            )
            return state, primary_reward
        else:
            state, dual_reward = self.dual_engine.infer(
                agent_observation=agent_observation
            )
            return state, dual_reward
