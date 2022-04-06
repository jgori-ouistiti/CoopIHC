import numpy
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, cat_element
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.policy.DualPolicy import DualPolicy
from coopihc.policy.ExamplePolicy import (
    CoordinatedPolicy,
    CoordinatedPolicyWithParams,
    PseudoRandomPolicy,
)
from coopihc.inference.DualInferenceEngine import DualInferenceEngine
from coopihc.inference.ExampleInferenceEngine import (
    CoordinatedInferenceEngine,
    RolloutCoordinatedInferenceEngine,
)
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine
from coopihc.bundle.Bundle import Bundle
import copy


class ExampleAssistant(BaseAgent):
    """An Example of an Assistant.

    An agent that handles the ExamplePolicy, has a single 1d state, and has the default observation and inference engines.
    See the documentation of the :py:mod:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class for more details.

    :meta public:
    """

    def __init__(self, *args, **kwargs):

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=1, init=0)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "assistant",
            *args,
            agent_policy=BasePolicy,
            policy_kwargs={"action_state": action_state},
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            **kwargs
        )


class CoordinatedAssistant(BaseAgent):
    def __init__(self, user_model=None, *args, **kwargs):

        self.user_model = user_model

        # Call the policy defined above
        action_state = State()
        action_state["action"] = cat_element(N=10, init=0)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "assistant",
            *args,
            agent_policy=CoordinatedPolicy(action_state=action_state),
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            **kwargs
        )

    def finit(self):
        copy_task = copy.deepcopy(self.task)
        self.simulation_bundle = Bundle(task=copy_task, user=self.user_model)


class CoordinatedAssistantWithInference(BaseAgent):
    def __init__(self, user_model=None, *args, **kwargs):

        self.user_model = user_model
        state = State()
        state["user_p0"] = copy.deepcopy(user_model.state.p0)
        # Call the policy defined above
        action_state = State()
        action_state["action"] = discrete_array_element(init=0, low=0, high=9)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = CoordinatedInferenceEngine()

        super().__init__(
            "assistant",
            *args,
            agent_state=state,
            agent_policy=CoordinatedPolicyWithParams(action_state=action_state),
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            **kwargs
        )

    def finit(self):
        copy_task = copy.deepcopy(self.task)
        self.simulation_bundle = Bundle(task=copy_task, user=self.user_model)


class CoordinatedAssistantWithRollout(BaseAgent):
    def __init__(self, task_model, user_model, p, **kwargs):
        state = State()
        state["p0"] = discrete_array_element(init=0, low=0, high=9)
        state["p1"] = discrete_array_element(init=p[0], low=0, high=9)
        state["p2"] = discrete_array_element(init=p[1], low=0, high=9)

        # Use default observation engine
        inference_engine = DualInferenceEngine(
            primary_inference_engine=RolloutCoordinatedInferenceEngine(
                task_model, user_model, self
            ),
            dual_inference_engine=BaseInferenceEngine(),
            primary_kwargs={},
            dual_kwargs={},
        )

        policy = PseudoRandomPolicy(
            action_state=State(
                **{"action": discrete_array_element(init=0, low=0, high=9)}
            )
        )

        super().__init__(
            "assistant",
            agent_state=state,
            agent_policy=policy,
            agent_observation_engine=None,
            agent_inference_engine=inference_engine,
            **kwargs
        )
