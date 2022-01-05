from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.space.utils import continuous_space

from coopihc.interactiontask.ExampleTask import ExampleTask

from coopihc.policy.BasePolicy import BasePolicy

from coopihc.agents.ExampleUser import ExampleUser

from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.wrappers.Train import TrainGym

import gym
from collections import OrderedDict
import numpy
import pytest


# def test_init():
#     global bundle
#     task = ExampleTask(gridsize=31, number_of_targets=8)
#     assistant = ExampleUser()  # action_space = [-1,0,1]
#     user = ExampleUser()  # action_space = [-1,0,1]
#     bundle = Bundle(task=task, user=user, assistant=assistant)
#     env = TrainGym(bundle)
#     assert isinstance(env, gym.Env)


# def test_action_space():
#     env = TrainGym(bundle)
#     assert env.action_space == gym.spaces.MultiDiscrete([3, 3])
#     assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# def test_action_space_user():
#     env = TrainGym(bundle, train_user=True, train_assistant=False)
#     assert env.action_space == gym.spaces.Discrete(3)
#     assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# # def test_action_space_continuous():


# def test_action_space_assistant():
#     env = TrainGym(bundle, train_user=False, train_assistant=True)
#     assert env.action_space == gym.spaces.Discrete(3)
#     assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# def test_observation_space():
#     env = TrainGym(bundle)
#     assert env.observation_space == gym.spaces.MultiDiscrete([4, 9, 9, 9, 3, 3])
#     assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)

#     global filterdict
#     filterdict = OrderedDict(
#         {
#             "user_state": OrderedDict({"goal": 0}),
#             "task_state": OrderedDict({"x": 0}),
#         }
#     )
#     env = TrainGym(bundle, observation_dict=filterdict)
#     print(env.observation_space)
#     assert env.observation_space == gym.spaces.MultiDiscrete([9, 9])
#     assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)


# def test_reset():
#     env = TrainGym(bundle)
#     obs = env.reset()
#     assert (obs[:4] == numpy.array([0, 0, 4, 4])).all()
#     assert obs[4] in [-1, 0, 1]
#     assert obs[5] in [-1, 0, 1]

#     filterdict = OrderedDict(
#         {
#             "user_state": OrderedDict({"goal": 0}),
#             "task_state": OrderedDict({"x": 0}),
#         }
#     )
#     env = TrainGym(bundle, observation_dict=filterdict)
#     obs = env.reset()
#     assert (obs == numpy.array([4, 0])).all()


# def test_step():
#     env = TrainGym(bundle)
#     obs = env.reset()

#     action = env.action_space.sample()
#     obs, rewards, is_done, _dic = env.step(action)
#     assert isinstance(is_done, bool)
#     assert isinstance(rewards, float)
#     assert isinstance(obs, numpy.ndarray)
#     assert isinstance(_dic, dict)
#     assert obs in env.observation_space

#     env = TrainGym(bundle, observation_dict=filterdict)
#     obs = env.reset()

#     action = env.action_space.sample()
#     obs, rewards, is_done, _dic = env.step(action)
#     assert isinstance(is_done, bool)
#     assert isinstance(rewards, float)
#     assert isinstance(obs, numpy.ndarray)
#     assert isinstance(_dic, dict)
#     assert obs in env.observation_space


# def test_sb3():
#     from stable_baselines3.common.env_checker import check_env

#     env = TrainGym(bundle, observation_dict=filterdict)
#     check_env(env, warn=True, skip_render_check=True)


# # incomplete test --- should check that the conversion works as intended.
# def test_wrapper():
#     env = TrainGym(bundle)
#     action_wrappers = env.action_wrappers
#     env = env.action_wrappers(env)
#     env.step(env.action_space.sample())


# # =================== Test with Continuous Space ======================


# def test_c_init():
#     global bundle
#     task = ExampleTask(gridsize=31, number_of_targets=8)
#     assistant = ExampleUser()  # action_space = [-1,0,1]
#     action_state = State(
#         **{
#             "action": StateElement(
#                 values=1.0,
#                 spaces=continuous_space(2 * numpy.array([[-1]]), numpy.array([[1]])),
#             )
#         }
#     )

#     policy = BasePolicy(action_state=action_state)
#     user = ExampleUser(override_policy=(policy, {}))  # action_space = [-1,0,1]
#     bundle = Bundle(task=task, user=user, assistant=assistant)
#     env = TrainGym(bundle, force=True)


# def test_c_action_space():
#     env = TrainGym(bundle, force=True)
#     assert env.action_space == gym.spaces.Box(
#         low=numpy.array([-2, -1]),
#         high=numpy.array([1, 1]),
#         shape=(2,),
#         dtype=numpy.float32,
#     )
#     action_wrapper = env.action_wrappers(env)
#     assert isinstance(action_wrapper, gym.ActionWrapper)
#     action = [-1.58, -0.75]
#     assert action_wrapper.action(action) == [-1.58, -1]


# def test_c_action_space_user():
#     env = TrainGym(bundle, train_user=True, train_assistant=False, force=True)
#     assert env.action_space == gym.spaces.Box(-2, 1, shape=(1,), dtype=numpy.float32)
#     assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# def test_c_action_space_assistant():
#     env = TrainGym(bundle, train_user=False, train_assistant=True, force=True)
#     assert env.action_space == gym.spaces.Box(-1, 1, shape=(1,), dtype=numpy.float32)
#     assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# def test_c_observation_space():
#     env = TrainGym(bundle, force=True)
#     assert env.observation_space == gym.spaces.Box(
#         low=numpy.array([0, -4, -4, -4, -2, -1]),
#         high=numpy.array([3, 4, 4, 4, 1, 1]),
#         dtype=numpy.float32,
#     )
#     assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)

#     global filterdict
#     filterdict = OrderedDict(
#         {
#             "user_state": OrderedDict({"goal": 0}),
#             "task_state": OrderedDict({"x": 0}),
#         }
#     )
#     env = TrainGym(bundle, observation_dict=filterdict, force=True)
#     assert env.observation_space == gym.spaces.Box(
#         low=numpy.array([-4, -4]),
#         high=numpy.array([4, 4]),
#         dtype=numpy.float32,
#     )
#     assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)


# def test_c_reset():
#     env = TrainGym(bundle, force=True)
#     obs = env.reset()
#     assert (obs[:4] == numpy.array([0.0, 0.0, 4.0, 4.0])).all()
#     assert -2 <= obs[4] and obs[4] <= 1
#     assert obs[5] in [-1.0, 0.0, 1.0]

#     filterdict = OrderedDict(
#         {
#             "user_state": OrderedDict({"goal": 0}),
#             "task_state": OrderedDict({"x": 0}),
#         }
#     )
#     env = TrainGym(bundle, observation_dict=filterdict, force=True)
#     obs = env.reset()
#     assert (obs == numpy.array([4.0, 0.0])).all()


# def test_c_step():
#     env = TrainGym(bundle, force=True)
#     obs = env.reset()

#     action = env.action_space.sample()
#     obs, rewards, is_done, _dic = env.step(action)
#     assert isinstance(is_done, bool)
#     assert isinstance(rewards, float)
#     assert isinstance(obs, numpy.ndarray)
#     assert isinstance(_dic, dict)
#     assert obs in env.observation_space

#     env = TrainGym(bundle, observation_dict=filterdict, force=True)
#     obs = env.reset()

#     action = env.action_space.sample()
#     obs, rewards, is_done, _dic = env.step(action)
#     assert isinstance(is_done, bool)
#     assert isinstance(rewards, float)
#     assert isinstance(obs, numpy.ndarray)
#     assert isinstance(_dic, dict)
#     assert obs in env.observation_space


# def test_c_sb3():
#     from stable_baselines3.common.env_checker import check_env

#     env = TrainGym(bundle, observation_dict=filterdict, force=True)
#     check_env(env, warn=True, skip_render_check=True)


# # incomplete test --- should check that the conversion works as intended.
# def test_c_wrapper():
#     env = TrainGym(bundle, force=True)
#     action_wrappers = env.action_wrappers
#     env = env.action_wrappers(env)
#     env.step(env.action_space.sample())


# if __name__ == "__main__":
#     test_init()
#     test_action_space()
#     test_action_space_user()
#     test_action_space_assistant()
#     test_observation_space()
#     test_reset()
#     test_step()
#     test_sb3()
#     test_wrapper()

#     test_c_init()
#     test_c_action_space()
#     test_c_action_space_user()
#     test_c_action_space_assistant()
#     test_c_observation_space()
#     test_c_reset()
#     test_c_step()
#     test_c_sb3()
#     test_c_wrapper()
