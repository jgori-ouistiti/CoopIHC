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
    user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
    assistant=BaseAgent(
        "assistant",
        override_policy=(BasePolicy, {"action_state": assistant_action_state}),
    ),
    seed=222,
)


def test_init():
    assert bundle.user.seedsequence.entropy == 222
    assert bundle.assistant.seedsequence.entropy == 222
    assert bundle.user.inference_engine.seedsequence.entropy == 222
    assert bundle.user.observation_engine.seedsequence.entropy == 222
    assert bundle.user.policy.seedsequence.entropy == 222
    assert bundle.assistant.inference_engine.seedsequence.entropy == 222
    assert bundle.assistant.observation_engine.seedsequence.entropy == 222
    assert bundle.assistant.policy.seedsequence.entropy == 222

    copied_game_state = copy.deepcopy(bundle.game_state)
    del copied_game_state["game_info"]
    for i in copied_game_state.filter(mode="space", flat=True).values():
        assert i.seed.entropy == 222


def test_samples():
    copied_game_state = copy.deepcopy(bundle.game_state)
    del copied_game_state["game_info"]
    sequence = [3, 1]
    for n, i in enumerate(copied_game_state.filter(mode="space", flat=True).values()):
        assert i.sample() == sequence[n]


if __name__ == "__main__":
    test_init()
    test_samples()
