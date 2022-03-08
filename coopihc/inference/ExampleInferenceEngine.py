from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine

# [start-infeng-subclass]
class ExampleInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Example class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, agent_state=None):
        """infer

        Do nothing. Same behavior as parent ``BaseInferenceEngine``

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        if agent_state is None:
            agent_state = self.state

        reward = 0
        # Do something
        # agent_state = ..
        # reward = ...

        return agent_state, reward


ExampleInferenceEngine(buffer_depth=5)
# [end-infeng-subclass]


class CoordinatedInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def simulation_bundle(self):
        return self.host.simulation_bundle

    def infer(self, agent_state=None):
        if agent_state is None:
            agent_state = self.state

        # Parameter Inference is naive on purpose here
        while True:
            # Prediction using user model (can sample directly from the policy in this case, because it already does a single-shot prediction)
            usermodel_action, _ = self.host.policy.sample(observation=self.observation)

            # actual observation
            user_action = self.observation.user_action.action

            # Compare prediction with observation
            if user_action != usermodel_action:
                # If different, increment parameter by 1 and apply modulo 10. This works because we assumed we knew everything except the value of this parameter.
                agent_state["user_p0"][:] = (agent_state["user_p0"] + 1) % 10
            else:
                break

        reward = 0

        return self.state, reward
