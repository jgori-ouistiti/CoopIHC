import numpy

from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


# [start-baseagent-init]

# Define a state
state = State()
state["goalstate"] = StateElement(
    4,
    Space(numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16), "discrete"),
)

# Define a policy (random policy)
action_state = State()
action_state["action"] = StateElement(
    0, Space(numpy.array([-1, 0, 1], dtype=numpy.int16), "discrete")
)
agent_policy = BasePolicy(action_state)

# Explicitly use default observation and inference engines (default behavior is triggered when keyword argument is not provided or keyword value is None)
observation_engine = RuleObservationEngine(
    deterministic_specification=base_user_engine_specification
)
inference_engine = BaseInferenceEngine(buffer_depth=0)

new_agent = BaseAgent(
    "user",
    agent_policy=agent_policy,
    agent_observation_engine=observation_engine,
    agent_inference_engine=inference_engine,
    agent_state=state,
)
# [end-baseagent-init]
