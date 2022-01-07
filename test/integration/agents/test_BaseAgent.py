import pytest
import numpy
import copy

from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.space.utils import autospace
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


# Define a state
state = State()
state["goalstate"] = StateElement(
    4, autospace([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)
)

# Define a policy (random policy)
action_state = State()
action_state["action"] = StateElement(0, autospace([-1, 0, 1], dtype=numpy.int16))
agent_policy = BasePolicy(action_state=action_state)

# Explicitly use default observation and inference engines (default behavior is triggered when keyword argument is not provided or keyword value is None)
observation_engine = RuleObservationEngine(
    deterministic_specification=base_user_engine_specification
)
inference_engine = BaseInferenceEngine(buffer_depth=0)

new_agent = BaseAgent(
    "user",
    agent_policy=BasePolicy,
    agent_observation_engine=observation_engine,
    agent_inference_engine=inference_engine,
    agent_state=state,
)


def test_init_args():
    # Check args influence
    new_agent = BaseAgent("user")
    assert isinstance(new_agent, BaseAgent)
    new_agent = BaseAgent("assistant")
    assert isinstance(new_agent, BaseAgent)
    with pytest.raises(TypeError):
        new_agent = BaseAgent()

    # ======== Checking defaults ==========
    new_agent = BaseAgent("user")
    # ----------- Check state default
    state = new_agent.state
    assert isinstance(new_agent.state, State)
    assert bool(state) == False
    # ----------- Check policy default
    policy = new_agent.policy
    assert policy.host is new_agent
    action_state = policy.action_state
    assert bool(action_state) == True
    assert action_state.get("action") is not None
    assert isinstance(action_state["action"], StateElement)
    assert action_state["action"] == 0
    space = action_state["action"].spaces
    assert isinstance(space, Space)
    assert space.dtype == numpy.int16
    # ----------- Check observation engine default
    obseng = new_agent.observation_engine
    assert obseng.host is new_agent
    se_tx = StateElement(
        numpy.array(1).reshape(1, 1),
        Space(
            [
                numpy.array([-1], dtype=numpy.float32).reshape(1, 1),
                numpy.array([1], dtype=numpy.float32).reshape(1, 1),
            ],
            "continuous",
        ),
    )
    se_ty = StateElement(
        numpy.array(1).reshape(1),
        Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete"),
    )
    se_tz = StateElement(
        numpy.array(-4).reshape(1),
        Space(numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16), "discrete"),
    )

    gamestate = State(
        **{
            "task_state": State(**{"tx": se_tx, "ty": se_ty, "tz": se_tz}),
            "user_state": State(**{"ux": copy.copy(se_tx)}),
            "assistant_state": State(**{"ax": copy.copy(se_ty)}),
            "user_action": State(),
            "assistant_action": State(),
        }
    )
    observed_gamestate, reward = obseng.observe(gamestate)
    assert reward == 0
    assert observed_gamestate == State(
        **{
            "task_state": gamestate["task_state"],
            "user_state": gamestate["user_state"],
        }
    )
    # ----------- Check inference engine default
    infeng = new_agent.inference_engine
    assert infeng.host is new_agent
    assert infeng.buffer_depth == 1
    infeng.add_observation(observed_gamestate)
    new_state, reward = infeng.infer()
    assert reward == 0
    assert new_state == observed_gamestate["user_state"]


def test_init_agent_state():
    se = StateElement(
        4,
        Space(
            numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16), "discrete"
        ),
    )

    state = State()
    state["goalstate"] = se
    new_agent = BaseAgent("user", agent_state=state)
    assert new_agent.state is state
    other_new_agent = BaseAgent("user", state_kwargs={"goalstate": copy.copy(se)})
    assert other_new_agent.state == state


def test_init_agent_policy():
    action_state = State()
    action_state["action"] = StateElement(
        1, Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    )
    policy = BasePolicy(action_state=action_state)
    new_agent = BaseAgent("user", agent_policy=policy)
    assert policy.host is new_agent
    assert new_agent.policy.action_state is action_state


def test_init_agent_inference_engine():
    inference_engine = BaseInferenceEngine(buffer_depth=10)
    new_agent = BaseAgent("user", agent_inference_engine=inference_engine)
    assert inference_engine.host is new_agent
    assert new_agent.inference_engine is inference_engine
    assert new_agent.inference_engine.buffer_depth == 10


def test_init_observation_engine():
    observation_engine = BaseObservationEngine()
    new_agent = BaseAgent("user", agent_observation_engine=observation_engine)
    assert observation_engine.host is new_agent
    assert new_agent.observation_engine is observation_engine

    se_tx = StateElement(
        numpy.array(1).reshape(1, 1),
        Space(
            [
                numpy.array([-1], dtype=numpy.float32).reshape(1, 1),
                numpy.array([1], dtype=numpy.float32).reshape(1, 1),
            ],
            "continuous",
        ),
    )
    se_ty = StateElement(
        numpy.array(1).reshape(1),
        Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete"),
    )
    se_tz = StateElement(
        numpy.array(-4).reshape(1),
        Space(numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16), "discrete"),
    )

    gamestate = State(
        **{
            "task_state": State(**{"tx": se_tx, "ty": se_ty, "tz": se_tz}),
            "user_state": State(**{"ux": copy.copy(se_tx)}),
            "assistant_state": State(**{"ax": copy.copy(se_ty)}),
            "user_action": State(),
            "assistant_action": State(),
        }
    )

    obs, reward = new_agent._observe(gamestate)
    assert reward == 0
    assert obs == gamestate


if __name__ == "__main__":
    test_init_args()
    test_init_agent_state()
    test_init_agent_policy()
    test_init_agent_inference_engine()
    test_init_observation_engine()
