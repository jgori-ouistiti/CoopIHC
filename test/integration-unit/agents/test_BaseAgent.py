import pytest
import numpy
import copy

from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.Space import Space
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.StateElement import StateElement

from coopihc.policy.BasePolicy import BasePolicy
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


# Define a state
state = State()
state["goalstate"] = discrete_array_element(init=-4, low=-4, high=4)

# Define a policy (random policy)
action_state = State()
action_state["action"] = discrete_array_element(init=0, low=-1, high=1)
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
    space = action_state["action"].space
    # ----------- Check observation engine default
    obseng = new_agent.observation_engine
    assert obseng.host is new_agent
    se_tx = array_element(init=1, low=numpy.array([-1]), high=numpy.array([1]))
    se_ty = cat_element(
        N=3,
        init=1,
    )
    se_tz = discrete_array_element(init=-4, low=-6, high=-1)

    turn_index = cat_element(N=4, init=0, out_of_bounds_mode="raw")

    round_index = discrete_array_element(N=None, init=0, out_of_bounds_mode="raw")

    gamestate = State(
        **{
            "game_info": State(
                **{"turn_index": turn_index, "round_index": round_index}
            ),
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
            "game_info": gamestate["game_info"],
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
    se = discrete_array_element(init=4, low=-4, high=4)
    state = State()
    state["goalstate"] = se
    new_agent = BaseAgent("user", agent_state=state)
    assert new_agent.state is state
    other_new_agent = BaseAgent("user", state_kwargs={"goalstate": copy.copy(se)})
    assert other_new_agent.state == state


def test_init_agent_policy():
    action_state = State()
    action_state["action"] = discrete_array_element(init=1, low=1, high=3)
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

    se_tx = array_element(init=1, low=numpy.array([-1]), high=numpy.array([1]))

    se_ty = cat_element(
        N=3,
        init=1,
    )
    se_tz = discrete_array_element(init=-4, low=-6, high=-1)

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
