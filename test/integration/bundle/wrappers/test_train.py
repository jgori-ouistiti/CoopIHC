from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space

from coopihc.interactiontask.ExampleTask import ExampleTask

from coopihc.policy.BasePolicy import BasePolicy

from coopihc.agents.ExampleUser import ExampleUser

from coopihc.bundle.Bundle import Bundle
from coopihc.bundle.wrappers.Train import Train

import gym
import numpy

task = ExampleTask(gridsize=31, number_of_targets=8)
assistant = ExampleUser()  # action_space = [-1,0,1]
user = ExampleUser()  # action_space = [-1,0,1]
bundle = Bundle(task=task, user=user, assistant=assistant)
env = Train(bundle)


def test_init():
    test_action_space()


#     test_observation_space()
#     test_reset()


def test_action_space():
    assert env.action_space == gym.spaces.MultiDiscrete([3, 3])
    assert isinstance(env.action_wrappers(env), gym.ActionWrapper)


# def test_observation_space():
#     pass


# def test_reset():
#     pass


# if __name__ == "__main__":
#     test_init()
