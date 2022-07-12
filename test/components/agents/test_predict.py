from curses import wrapper
from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.assistants import BIGGain

from coopihc import Bundle, WrapperReferencer

from gym import ActionWrapper
from gym.spaces import Box
from gym.wrappers import FilterObservation, FlattenObservation
from stable_baselines3.common.env_checker import check_env
import numpy as np


def make_simple_pointing_env():
    config_task = dict(gridsize=20, number_of_targets=8, mode="position")
    config_user = dict(error_rate=0.01)

    obs_keys = (
        "assistant_state__beliefs",
        "task_state__position",
        "task_state__targets",
        "user_action__action",
    )

    class AssistantActionWrapper(ActionWrapper, WrapperReferencer):
        def __init__(self, env):
            ActionWrapper.__init__(self, env)
            WrapperReferencer.__init__(self, env)
            _as = env.action_space["assistant_action__action"]
            self.action_space = Box(low=-1, high=1, shape=_as.shape, dtype=np.float32)
            self.low, self.high = _as.low, _as.high
            self.half_amp = (self.high - self.low) / 2
            self.mean = (self.high + self.low) / 2

        def action(self, action):
            return {"assistant_action__action": int(action * self.half_amp + self.mean)}

        def reverse_action(self, action):
            raw = action["assistant_action__action"]
            return (raw - self.mean) / self.half_amp

    task = SimplePointingTask(**config_task)
    user = CarefulPointer(**config_user)
    assistant = BIGGain()
    bundle = Bundle(
        seed=12345,
        task=task,
        user=user,
        assistant=assistant,
        random_reset=True,
        start_after=3,
        reset_skip_user_step=False,
    )

    env = bundle.convert_to_gym_env(train_user=False, train_assistant=True)
    env = FlattenObservation(FilterObservation(env, obs_keys))
    env = AssistantActionWrapper(env)

    # Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
    check_env(env)
    return env


def test_predict_one_step():
    env = make_simple_pointing_env()
    for i in range(100):
        env.reset()
        action, reward = env.bundle.assistant.take_action(increment_turn=False)
        _action, _ = env.bundle.assistant.predict(
            None, increment_turn=False, wrappers=True
        )
        assert (_action) + 1 * 19 / 2


def test_predict_increment_False():
    env = make_simple_pointing_env()
    env.reset()
    action, reward = env.bundle.assistant.take_action(increment_turn=False)
    for i in range(100):
        assert action == env.bundle.assistant.take_action(increment_turn=False)[0]


def test_predict_with_step():
    import copy

    env = make_simple_pointing_env()
    env.reset()
    copied_env = copy.deepcopy(env)

    state_just_after_initial_reset = copy.deepcopy(env.unwrapped.bundle.state)
    action, reward = env.unwrapped.bundle.assistant.take_action(increment_turn=False, update_action_state = False)
    state_after_action_increment_turn__false = env.unwrapped.bundle.state
    assert state_just_after_initial_reset == state_after_action_increment_turn__false

    # before trying out things make sure we have the same object:
    assert env.unwrapped.bundle.state == copied_env.unwrapped.bundle.state
    # play action with increment = True
    _action, _ = copied_env.unwrapped.bundle.assistant.predict(
        None, increment_turn=True, wrapper=True, update_action_state = True
    )
    # Play the bundle forward so that we can compare with step
    # copied_env.unwrapped.bundle.step(go_to=3)
    env.step(_action)
    assert env.unwrapped.bundle.state == copied_env.unwrapped.bundle.state


if __name__ == "__main__":
    test_predict_one_step()
    test_predict_increment_False()
    test_predict_with_step()
