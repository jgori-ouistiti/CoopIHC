from typing import OrderedDict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from coopihc import State, BasePolicy, Bundle, discrete_array_element
from coopihc.bundle.wrappers.Train import TrainGym, TrainGym2SB3ActionWrapper

import numpy
import gym
import pytest


from coopihc.examples.simplepointing.envs import SimplePointingTask
from coopihc.examples.simplepointing.users import CarefulPointer
from coopihc.examples.simplepointing.assistants import ConstantCDGain


task = SimplePointingTask(gridsize=31, number_of_targets=8)
unitcdgain = ConstantCDGain(1)

# The policy to be trained has the simple action set [-5,-4,-3,-2,-1,0,1,2,3,,4,5]
action_state = State()
action_state["action"] = discrete_array_element(low=-5, high=5)

user = CarefulPointer(override_policy=(BasePolicy, {"action_state": action_state}))
bundle = Bundle(task=task, user=user, assistant=unitcdgain, reset_go_to=1)
observation = bundle.reset()
env = TrainGym(
    bundle,
    train_user=True,
    train_assistant=False,
)


def test_init_spaces():
    assert env.action_space == gym.spaces.Dict(
        OrderedDict(
            {
                "user_action__action": gym.spaces.Box(
                    low=-5, high=5, shape=(1,), dtype=numpy.int64
                )
            }
        )
    )


def test_check_env():
    from stable_baselines3.common.env_checker import check_env

    check_env(env, warn=False)


if __name__ == "__main__":
    test_init_spaces()
    test_check_env()
