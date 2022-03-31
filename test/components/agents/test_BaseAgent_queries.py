import pytest
import numpy
import copy

from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.elements import array_element, discrete_array_element
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.base.elements import example_game_state
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


user_action_state = State()
user_action_state["action"] = array_element(low=-1, high=1)


class TestInferenceEngine(BaseInferenceEngine):
    def infer(self, agent_observation=None):
        old_state = agent_observation["user_state"]
        new_goal_value = old_state["goal"] + 1
        old_state["goal"][...] = new_goal_value
        return old_state, 0


class TestPolicy(BasePolicy):
    def sample(self, agent_observation=None, agent_state=None):
        action = agent_observation["assistant_state"]["beliefs"][2]
        return action, 0


user = BaseAgent(
    "user",
    agent_state=State(**{"goal": discrete_array_element(low=0, high=10)}),
    agent_inference_engine=TestInferenceEngine(),
    agent_policy=TestPolicy(action_state=user_action_state),
)


def test_obs_game_state():
    user.reset_all(random=False)
    game_state = example_game_state()
    obs, reward = user.observe(game_state=game_state, affect_bundle=False)
    assert user.inference_engine.buffer is None
    obs, reward = user.observe(game_state=game_state, affect_bundle=True)
    del game_state["assistant_state"]
    assert user.inference_engine.buffer[-1] == game_state


def test_obs_substates():
    game_state = example_game_state()
    user_state = game_state["user_state"]
    assistant_state = game_state["assistant_state"]
    user_action = game_state["user_action"]
    obs, reward = user.observe(
        user_state=user_state,
        assistant_state=assistant_state,
        user_action=user_action,
        affect_bundle=False,
    )


def test_obs():
    test_obs_game_state()
    test_obs_substates()


def test_infer():
    agent_observation = example_game_state()
    new_state, reward = user.infer(
        agent_observation=agent_observation, affect_bundle=False
    )
    assert new_state["goal"] == 1
    assert user.state["goal"] == 0

    agent_observation = example_game_state()
    new_state, reward = user.infer(
        agent_observation=agent_observation, affect_bundle=True
    )
    assert new_state["goal"] == 1
    assert user.state["goal"] == 1


def test_take_action():
    agent_observation = example_game_state()
    act, rew = user.take_action(agent_observation=agent_observation, agent_state=None)


if __name__ == "__main__":
    test_obs()
    test_infer()
    test_take_action()
