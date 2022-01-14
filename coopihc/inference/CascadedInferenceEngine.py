from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class CascadedInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Combine two or more inference engines serially. Example code:

    .. code-block::

        first_inference_engine = ProvideLikelihoodInferenceEngine(perceptualnoise)
        second_inference_engine = LinearGaussianContinuous()
        inference_engine = CascadedInferenceEngine(
            [first_inference_engine, second_inference_engine]
        )

    """

    def __init__(self, engine_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_list = engine_list
        self.render_tag = ["text", "plot"]
        self._host = None

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        self._host = value
        for eng in self.engine_list:
            eng.host = value

    def add_observation(self, observation):
        """add observation

        Add an observation to a buffer. If the buffer does not exist, create a naive buffer. The buffer has a size given by buffer length

        :param observation: observation produced by an engine
        :type observation: :py:class:`State<coopihc.space.State.State>`
        """

        if self.buffer is None:
            self.buffer = []
        if len(self.buffer) < self.buffer_depth:
            self.buffer.append(observation)
        else:
            self.buffer = self.buffer[1:] + [observation]

        # Broadcast observations to contained inference engines
        for eng in self.engine_list:
            eng.add_observation(observation)

    def __content__(self):
        return {
            self.__class__.__name__: {
                "Engine{}".format(ni): i.__content__()
                for ni, i in enumerate(self.engine_list)
            }
        }

    def infer(self, *args, user_state=None, **kwargs):

        if user_state is None:
            user_state = self.state
        rewards = 0
        for engine in self.engine_list:
            new_state, new_reward = engine.infer(user_state=user_state)
            rewards += new_reward
            user_state.update(new_state)

        return user_state, rewards

    def render(self, *args, **kwargs):
        for eng in self.engine_list:
            eng.render(*args, **kwargs)
