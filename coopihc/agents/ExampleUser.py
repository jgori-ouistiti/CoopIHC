import numpy
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.policy.ExamplePolicy import ExamplePolicy
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ExampleUser(BaseAgent):
    """An Example of a User.

    An agent that handles the ExamplePolicy, has a single 1d state, and has the default observation and inference engines.
    See the documentation of the :py:mod:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class for more details.

    :meta public:
    """

    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = StateElement(
            4,
            Space(
                numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16),
                "discrete",
            ),
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            0, Space(numpy.array([-1, 0, 1], dtype=numpy.int16), "discrete")
        )
        # agent_policy = ExamplePolicy(action_state=action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            *args,
            agent_policy=ExamplePolicy,
            policy_kwargs={"action_state": action_state},
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """
        self.state["goal"][:] = 4
