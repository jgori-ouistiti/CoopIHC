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


class ExampleUser(BaseAgent):
    """An agent that handles the ExamplePolicy."""

    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = StateElement(
            values=numpy.array([4]),
            spaces=[
                Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])
            ],
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            values=None,
            spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
        )
        agent_policy = ExamplePolicy(action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    # Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4
    def reset(self, dic=None):
        self.state["goal"]["values"] = 4

    # [start-baseagent-init]

    # Define a state
    state = State()
    state["goalstate"] = StateElement(
        values=numpy.array([4]),
        spaces=[
            Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])
        ],
    )

    # Define a policy (random policy)
    action_state = State()
    action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )
    agent_policy = BasePolicy(action_state)

    # Explicitly use default observation and inference engines (default behavior is triggered when keyword argument is not provided or keyword value is None)
    observation_engine = RuleObservationEngine(
        deterministic_specification=base_user_engine_specification
    )
    inference_engine = BaseInferenceEngine(buffer_depth=0)

    BaseAgent(
        "user",
        agent_policy=agent_policy,
        agent_observation_engine=observation_engine,
        agent_inference_engine=inference_engine,
        agent_state=state,
    )
    # [end-baseagent-init]


if __name__ == "__main__":
    exampleuser = ExampleUser()
