from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from coopihc import State, BasePolicy, Bundle, discrete_array_element
from coopihc.bundle.wrappers.Train import TrainGym, TrainGym2SB3ActionWrapper

import numpy
import gym
import pytest


# [start-define-bundle]
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


# >>> print(observation)
# ----------------  -----------  -------------------------  ------------------------------------------
# game_info         turn_index   1                          Discr(4)
#                   round_index  0                          Discr(1)
# task_state        position     24                         Discr(31)
#                   targets      [ 4 12 13 16 20 21 23 25]  MultiDiscr[31, 31, 31, 31, 31, 31, 31, 31]
# user_state        goal         4                          Discr(31)
# user_action       action       -4                         Discr(11)
# assistant_action  action       [[1.]]                     Cont(1, 1)
# ----------------  -----------  -------------------------  ------------------------------------------

# [end-define-bundle]

# [start-define-traingym]

env = TrainGym(
    bundle,
    train_user=True,
    train_assistant=False,
)
obs = env.reset()
# >>> print(env.action_space)
# Dict(user_action_0:Box(-5, 5, (), int64))
# >>> print(env.observation_space)
# Dict(turn_index:Discrete(4), round_index:Box(0, 9223372036854775807, (), int64), position:Box(0, 30, (), int64), targets:Box(0, 31, (8,), int64), goal:Box(0, 30, (), int64), user_action:Box(-5, 5, (), int64), assistant_action:Box(1, 1, (1, 1), int64))

obs, reward, is_done, inf = env.step({"user_action": 1})

# Use env_checker from stable_baselines3 to verify that the env adheres to the Gym API
from stable_baselines3.common.env_checker import check_env

check_env(env, warn=False)
# [end-define-traingym]


# [start-define-mywrappers]

TEN_EPSILON32 = 10 * numpy.finfo(numpy.float32).eps


class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=numpy.float32,
        )

    def action(self, action):
        return {
            "user_action": int(
                numpy.around(action * 11 / 2 - TEN_EPSILON32, decimals=0)
            )
        }

    def reverse_action(self, action):
        return numpy.array((action["user_action"] - 5.0) / 11.0 * 2).astype(
            numpy.float32
        )


from gym.wrappers import FilterObservation, FlattenObservation

# Apply Observation Wrapper
modified_env = FlattenObservation(FilterObservation(env, ("position", "goal")))
# Normalize actions with a custom wrapper
modified_env = NormalizeActionWrapper(modified_env)
# >>> print(modified_env.action_space)
# Box(-0.9999997615814209, 0.9999997615814209, (1,), float32)

# >>> print(modified_env.observation_space)
# Box(0.0, 30.0, (2,), float32)

check_env(modified_env, warn=True)
# >>> modified_env.reset()
# array([ 4., 23.], dtype=float32)

# Check that modified_env and the bundle game state concord
# >>> print(modified_env.unwrapped.bundle.game_state)
# ----------------  -----------  -------------------------  -------------
# game_info         turn_index   1                          CatSet(4)
#                   round_index  0                          Numeric()
# task_state        position     4                          Numeric()
#                   targets      [ 2  7  8 19 20 21 23 25]  Numeric(8,)
# user_state        goal         23                         Numeric()
# user_action       action       -5                         Numeric()
# assistant_action  action       [[1]]                      Numeric(1, 1)
# ----------------  -----------  -------------------------  -------------


modified_env.step(
    0.99
)  # 0.99 is cast to +5, multiplied by CD gain of 1 = + 5 increment

# >>> modified_env.step(
# ...     0.99
# ... )
# (array([ 9., 23.], dtype=float32), -1.0, False, \\infodict\\

# >>> print(modified_env.unwrapped.bundle.game_state)
# ----------------  -----------  -------------------------  -------------
# game_info         turn_index   1                          CatSet(4)
#                   round_index  1                          Numeric()
# task_state        position     9                          Numeric()
#                   targets      [ 2  7  8 19 20 21 23 25]  Numeric(8,)
# user_state        goal         23                         Numeric()
# user_action       action       5                          Numeric()
# assistant_action  action       [[1]]                      Numeric(1, 1)
# ----------------  -----------  -------------------------  -------------


# [end-define-mywrappers]

# As an Alternative to MyActionWrapper, you can use this generic wrapper which will one-hot encode discrete spaces to continuous spaces. SB3 handles dict spaces fine, but will one-hot encode discrete spaces and the like to a box.
# out of date
# # [start-define-SB3wrapper]
# sb3env = TrainGym2SB3ActionWrapper(env)
# check_env(sb3env, warn=True)
# # [end-define-SB3wrapper]


# ============= function to make env

# [start-make-env]
def make_env():
    def _init():

        task = SimplePointingTask(gridsize=31, number_of_targets=8)
        unitcdgain = ConstantCDGain(1)

        action_state = State()
        action_state["action"] = discrete_array_element(low=-5, high=5)

        user = CarefulPointer(
            override_policy=(BasePolicy, {"action_state": action_state})
        )
        bundle = Bundle(task=task, user=user, assistant=unitcdgain)
        observation = bundle.reset(go_to=1)
        env = TrainGym(
            bundle,
            train_user=True,
            train_assistant=False,
        )

        modified_env = FlattenObservation(FilterObservation(env, ("position", "goal")))
        modified_env = NormalizeActionWrapper(modified_env)

        return modified_env

    return _init


# [end-make-env]
# =============
pytest.skip("not testing the learning", allow_module_level=True)

# [start-train]
if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for i in range(4)])
    # to track rewards on tensorboard
    from stable_baselines3.common.vec_env import VecMonitor

    env = VecMonitor(env, filename="tmp/log")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
    print("start training")
    model.learn(total_timesteps=1e6)
    model.save("saved_model")
# [end-train]
