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

task = ExampleTask(gridsize=31, number_of_targets=8)
assistant = ExampleUser()  # action_space = [-1,0,1]
user = ExampleUser()  # action_space = [-1,0,1]
bundle = Bundle(task=task, user=user, assistant=assistant)


def test_init():
    env = TrainGym(bundle)
    assert isinstance(env, gym.Env)
    test_action_space()
    test_observation_space()


def test_action_space():
    env = TrainGym(bundle)
    assert env.action_space == gym.spaces.MultiDiscrete([3, 3])
    assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


def test_observation_space():
    env = TrainGym(bundle)
    assert env.observation_space == gym.spaces.MultiDiscrete([4, 9, 9, 9, 3, 3])
    assert isinstance(env.observation_wrappers(env), gym.ObservationWrapper)

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
    assert obs[:4] == [0, 0, 4, 4]
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
    assert obs == [4, 0]


if __name__ == "__main__":
    test_init()
    test_reset()
