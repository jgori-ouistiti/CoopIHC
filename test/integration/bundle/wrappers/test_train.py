from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space

from coopihc.interactiontask.ExampleTask import ExampleTask

from coopihc.policy.BasePolicy import BasePolicy

from coopihc.agents.ExampleUser import ExampleUser

from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.wrappers.Train import TrainGym

import gym
from collections import OrderedDict
import numpy


def test_ExampleBundle():
    task = ExampleTask(gridsize=31, number_of_targets=8)
    assistant = ExampleUser()  # action_space = [-1,0,1]
    user = ExampleUser()  # action_space = [-1,0,1]
    global bundle
    bundle = Bundle(task=task, user=user, assistant=assistant)

    test_init()
    test_action_space()
    test_observation_space()
    test_reset()
    test_step()
    test_sb3()


def test_init():
    env = TrainGym(bundle)
    assert isinstance(env, gym.Env)


def test_action_space():
    env = TrainGym(bundle)
    assert env.action_space == gym.spaces.MultiDiscrete([3, 3])
    assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


def test_observation_space():
    env = TrainGym(bundle)
    assert env.observation_space == gym.spaces.MultiDiscrete([4, 9, 9, 9, 3, 3])
    assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)

    global filterdict
    filterdict = OrderedDict(
        {
            "user_state": OrderedDict({"goal": 0}),
            "task_state": OrderedDict({"x": 0}),
        }
    )
    env = TrainGym(bundle, observation_dict=filterdict)
    assert env.observation_space == gym.spaces.MultiDiscrete([9, 9])
    assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)


def test_reset():
    env = TrainGym(bundle)
    obs = env.reset()
    assert (obs[:4] == numpy.array([0, 0, 4, 4])).all()
    assert obs[4] in [-1, 0, 1]
    assert obs[5] in [-1, 0, 1]

    filterdict = OrderedDict(
        {
            "user_state": OrderedDict({"goal": 0}),
            "task_state": OrderedDict({"x": 0}),
        }
    )
    env = TrainGym(bundle, observation_dict=filterdict)
    obs = env.reset()
    assert (obs == numpy.array([4, 0])).all()


def test_step():
    env = TrainGym(bundle)
    obs = env.reset()

    action = env.action_space.sample()
    obs, rewards, is_done, _dic = env.step(action)
    assert isinstance(is_done, bool)
    assert isinstance(rewards, float)
    assert isinstance(obs, numpy.ndarray)
    assert isinstance(_dic, dict)
    assert obs in env.observation_space

    env = TrainGym(bundle, observation_dict=filterdict)
    obs = env.reset()

    action = env.action_space.sample()
    obs, rewards, is_done, _dic = env.step(action)
    assert isinstance(is_done, bool)
    assert isinstance(rewards, float)
    assert isinstance(obs, numpy.ndarray)
    assert isinstance(_dic, dict)
    assert obs in env.observation_space


def test_sb3():
    from stable_baselines3.common.env_checker import check_env

    env = TrainGym(bundle, observation_dict=filterdict)
    check_env(env, warn=True, skip_render_check=True)


# incomplete test --- should check that the conversion works as intended.
def test_wrapper():
    env = TrainGym(bundle)
    action_wrappers = env.action_wrappers
    env = env.action_wrappers(env)
    env.step(env.action_space.sample())


if __name__ == "__main__":
    test_ExampleBundle()
