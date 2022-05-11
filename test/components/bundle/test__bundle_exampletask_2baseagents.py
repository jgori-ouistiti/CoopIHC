import numpy
from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.base.State import State
from coopihc.base.elements import (
    discrete_array_element,
    array_element,
    cat_element,
    integer_space,
    integer_set,
)
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.agents.ExampleUser import ExampleUser
from coopihc.agents.ExampleAssistant import ExampleAssistant
import copy

user_action_state = State()
user_action_state["action"] = discrete_array_element(low=-1, high=1)

assistant_action_state = State()
assistant_action_state["action"] = cat_element(N=2)

bundle = Bundle(
    task=ExampleTask(),
    # Here policy = None, will call BasePolicy of BaseAgent with kwargs policy_kwargs
    user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
    # Here, we use the override mechanism directly. Both are equivalent
    assistant=BaseAgent(
        "assistant",
        override_policy=(BasePolicy, {"action_state": assistant_action_state}),
    ),
)

# ============== Helpers =============
def assert_action_states(bundle):
    user_flag = bundle.user.policy.action_state["action"].space == integer_space(
        N=3, start=-1
    )

    assistant_flag = bundle.assistant.policy.action_state[
        "action"
    ].space == integer_set(2)
    return user_flag and assistant_flag


def assert_agent_states(bundle):
    user_flag = bundle.user.state == State()
    assistant_flag = bundle.assistant.state == State()
    return user_flag and assistant_flag


# ============= initialize Bundle ==========
def test_init():
    global bundle
    assert isinstance(bundle.task, ExampleTask)
    assert isinstance(bundle.user, BaseAgent)
    assert isinstance(bundle.user.policy, BasePolicy)
    assert isinstance(bundle.assistant, BaseAgent)
    assert isinstance(bundle.assistant.policy, BasePolicy)
    assert assert_action_states(bundle)


# =============== Reset with various turns ===========
def test_reset_turn0():
    global bundle

    bundle.reset(go_to=0)

    assert bundle.round_number == 0
    assert bundle.turn_number == 0
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert bundle.user.inference_engine.buffer.is_empty()
    assert bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_turn1():
    global bundle

    bundle.reset(go_to=1)

    assert bundle.round_number == 0
    assert bundle.turn_number == 1
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert not bundle.user.inference_engine.buffer.is_empty()
    assert bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_turn2():
    global bundle

    bundle.reset(go_to=2)

    assert bundle.round_number == 0
    assert bundle.turn_number == 2
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert not bundle.user.inference_engine.buffer.is_empty()
    assert bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_turn3():
    global bundle

    bundle.reset(go_to=3)

    assert bundle.round_number == 0
    assert bundle.turn_number == 3
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert not bundle.user.inference_engine.buffer.is_empty()
    assert not bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_turn2_to_3():
    global bundle

    bundle.reset(start_after=2, go_to=3)

    assert bundle.round_number == 0
    assert bundle.turn_number == 3
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert bundle.user.inference_engine.buffer.is_empty()
    assert not bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_string():
    global bundle

    bundle.reset(start_after="after_user_action", go_to="before_assistant_action")

    assert bundle.round_number == 0
    assert bundle.turn_number == 3
    assert assert_action_states(bundle)
    assert assert_agent_states(bundle)
    assert bundle.user.inference_engine.buffer.is_empty()
    assert not bundle.assistant.inference_engine.buffer.is_empty()


def test_reset_turn():
    test_reset_turn0()
    test_reset_turn1()
    test_reset_turn2()
    test_reset_turn3()
    test_reset_turn2_to_3()
    test_reset_string()


def test_multi_reset():
    global bundle
    for i in range(10):
        state = bundle.reset(go_to=0)
        assert bundle.round_number == 0
        assert bundle.turn_number == 0
        assert bundle.task.state["x"] == 0


def test_reset():
    test_reset_turn()
    test_multi_reset()


# ============ Helpers ================


def ensure_reward_dict(reward_dict):
    assert isinstance(reward_dict, dict)
    assert sorted(list(reward_dict.keys())) == sorted(
        [
            "user_observation_reward",
            "user_inference_reward",
            "user_policy_reward",
            "first_task_reward",
            "assistant_observation_reward",
            "assistant_inference_reward",
            "assistant_policy_reward",
            "second_task_reward",
        ]
    )
    assert all(
        [isinstance(v, (int, float, complex)) for v in list(reward_dict.values())]
    )

    return True


def ensure_state(bundle, state):
    assert isinstance(state, State)
    assert bundle.game_state == state
    return True


# =========== Check step(user_action, assistant_action)


def null_step(bundle, task_state_value):
    x = task_state_value
    state, rewards, is_done = bundle.step(user_action=0, assistant_action=0)
    assert bundle.game_state["task_state"]["x"] == x
    assert bundle.turn_number == 1
    assert bundle.round_number == 1
    assert is_done == False
    assert ensure_reward_dict(rewards)
    assert ensure_state(bundle, state)
    return True


def on_user_action(bundle, task_state_value):
    x = task_state_value
    state, rewards, is_done = bundle.step(user_action=1, assistant_action=0)
    assert bundle.game_state["task_state"]["x"] == x + 1
    assert bundle.turn_number == 1
    assert bundle.round_number == 1
    if x + 1 == 4:
        assert is_done == True
    else:
        assert is_done == False

    assert ensure_reward_dict(rewards)
    assert ensure_state(bundle, state)
    return True


def on_assistant_action(bundle, task_state_value):
    x = task_state_value
    state, rewards, is_done = bundle.step(user_action=0, assistant_action=1)
    assert bundle.game_state["task_state"]["x"] == x + 1
    assert bundle.turn_number == 1
    assert bundle.round_number == 1
    if x + 1 == 4:
        assert is_done == True
    else:
        assert is_done == False

    assert ensure_reward_dict(rewards)
    assert ensure_state(bundle, state)
    return True


def test_step_both():
    global bundle
    bundle.reset(go_to=1)
    x = bundle.game_state["task_state"]["x"]
    assert null_step(bundle, x)
    bundle.reset(go_to=1)
    x = copy.copy(bundle.game_state["task_state"]["x"])
    assert on_user_action(bundle, x)
    bundle.reset(go_to=1)
    x = copy.copy(bundle.game_state["task_state"]["x"])
    assert on_assistant_action(bundle, x)


# =============== Check step(user_action)


def null_step_useronly(bundle, task_state_value):
    x = task_state_value
    state, rewards, is_done = bundle.step(user_action=0)
    assistant_action = state["assistant_action"]["action"]
    assert bundle.game_state["task_state"]["x"] == x + assistant_action
    assert bundle.turn_number == 1
    assert bundle.round_number == 1
    assert is_done == False
    assert ensure_reward_dict(rewards)
    assert ensure_state(bundle, state)
    return True


def on_user_action_useronly(bundle, task_state_value):
    x = task_state_value
    state, rewards, is_done = bundle.step(user_action=1)
    assistant_action = state["assistant_action"]["action"]

    assert bundle.game_state["task_state"]["x"] == numpy.clip(
        x + 1 + assistant_action, -1, 4
    )
    assert bundle.round_number == 1
    assert bundle.turn_number == 1
    if x + 1 == 4:
        assert is_done == True
    else:
        assert is_done == False

    assert ensure_reward_dict(rewards)
    assert ensure_state(bundle, state)
    return True


def test_step_useronly():
    global bundle
    bundle.reset(go_to=1)
    x = copy.deepcopy(bundle.game_state["task_state"]["x"])
    assert null_step_useronly(bundle, x)
    bundle.reset(go_to=1)
    x = copy.deepcopy(bundle.game_state["task_state"]["x"])
    assert on_user_action_useronly(bundle, x)


def test_multistep_both():
    global bundle
    bundle.reset(go_to=1)
    x = copy.deepcopy(bundle.game_state["task_state"]["x"])
    while True:
        state, rewards, is_done = bundle.step(user_action=1, assistant_action=0)
        if x + 1 == 4:
            assert is_done == True
            return
        else:
            assert is_done == False

        assert state["task_state"]["x"] == numpy.clip(x + 1, -1, 4)
        x = copy.deepcopy(state["task_state"]["x"])


def test_multistep_single():
    global bundle
    bundle.reset(go_to=1)
    x = copy.deepcopy(bundle.game_state["task_state"]["x"])
    while True:
        state, rewards, is_done = bundle.step(user_action=1)
        assistant_action = state["assistant_action"]["action"]
        new_value = numpy.clip(x + 1 + assistant_action, -1, 4)
        if new_value >= 4:
            assert is_done == True
            return
        else:
            assert is_done == False

        assert state["task_state"]["x"] == new_value
        x = copy.deepcopy(state["task_state"]["x"])


def test_multistep_none():
    global bundle

    bundle.reset(go_to=1)
    x = copy.deepcopy(bundle.game_state["task_state"]["x"])
    init = copy.deepcopy(x)
    while True:
        state, rewards, is_done = bundle.step()
        user_action = state["user_action"]["action"]
        assistant_action = state["assistant_action"]["action"]
        new_value = numpy.clip(x + user_action + assistant_action, -1, 4)
        if new_value >= 4:
            assert is_done == True
            assert state["task_state"]["x"] == 4
            break
        else:
            assert is_done == False

        try:
            assert state["task_state"]["x"] == new_value
        except AssertionError:
            print(init)
            print(state["task_state"]["x"])
            print(new_value)
            exit()

        x = copy.deepcopy(state["task_state"]["x"])


def test_multistep():
    test_multistep_both()
    test_multistep_single()
    test_multistep_none()


def test_partial_round():
    global bundle
    for i in [1, 2, 3]:
        bundle.reset(go_to=0)
        state, rewards, is_done = bundle.step(go_to=i)
        assert bundle.turn_number == i
        assert bundle.round_number == 0

    for i in [3, 0, 1]:
        bundle.reset(go_to=2)
        state, rewards, is_done = bundle.step(go_to=i)
        assert bundle.turn_number == i
        if i >= 2:
            assert bundle.round_number == 0
        else:
            assert bundle.round_number == 1


def quarter_step():
    global bundle
    bundle.reset(go_to=0)
    for i in range(4):
        state, rewards, is_done = bundle.quarter_step()
        assert bundle.turn_number == (i + 1) % 4
        if i < 3:
            assert bundle.round_number == 0
        else:
            assert bundle.round_number == 1


def agent_step():
    global bundle
    bundle.reset(go_to=0)
    bundle.user.prepare_action()
    assert bundle.turn_number == 1
    assert bundle.round_number == 0
    bundle.user.take_action()
    assert bundle.turn_number == 2
    assert bundle.round_number == 0
    bundle.assistant.prepare_action()
    assert bundle.turn_number == 3
    assert bundle.round_number == 0
    bundle.assistant.take_action()
    assert bundle.turn_number == 0
    assert bundle.round_number == 1


def test_step():
    test_step_both()
    test_step_useronly()
    test_multistep()
    test_partial_round()
    quarter_step()
    agent_step()


if __name__ == "__main__":
    test_init()
    test_reset()
    test_step()
