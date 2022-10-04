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

    @property
    def observation(self):
        return self.engine_list[-1].observation

    def add_observation(self, observation):
        """add observation

        Add an observation to a buffer. If the buffer does not exist, create a naive buffer. The buffer has a size given by buffer length

        :param observation: observation produced by an engine
        :type observation: :py:class:`State<coopihc.base.State.State>`
        """

        # if self.buffer is None:
        #     self.buffer = []
        # if len(self.buffer) < self.buffer_depth:
        #     self.buffer.append(observation)
        # else:
        #     self.buffer = self.buffer[1:] + [observation]

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

    # Don't put the default value decorator here, it will cause an error + not needed because the engine's inference engines are already decorated
    def infer(self, agent_observation=None):

        user_state = (
            agent_observation.get("user_state", {})
            if agent_observation is not None
            else {}
        )
        rewards = 0
        for engine in self.engine_list:
            new_state, new_reward = engine.infer(agent_observation=agent_observation)
            rewards += new_reward
            user_state.update(new_state)
            # agent_observation[f"{self.host.role}_state"].update(new_state)

        return user_state, rewards

    def render(
        self, mode=None, ax_user=None, ax_assistant=None, ax_task=None, **kwargs
    ):

        for eng in self.engine_list:
            eng.render(
                ax_task=ax_task,
                ax_user=ax_user,
                ax_assistant=ax_assistant,
                mode=mode,
                **kwargs,
            )
