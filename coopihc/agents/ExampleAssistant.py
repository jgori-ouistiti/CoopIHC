import numpy
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.policy import ExamplePolicy
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ExampleAssistant(BaseAgent):
    """An Example of an Assistant.

    An agent that handles the ExamplePolicy, has a single 1d state, and has the default observation and inference engines.
    See the documentation of the :py:mod:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class for more details.

    :meta public:
    """

    def __init__(self, *args, **kwargs):

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0, Space(numpy.array([0], dtype=numpy.int16), "discrete")
        )

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
