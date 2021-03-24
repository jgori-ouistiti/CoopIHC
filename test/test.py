import gym
import numpy

from core.bundle import PlayNone, PlayOperator, PlayAssistant, PlayBoth, Train
from pointing.envs import SimplePointingTask
from pointing.operators import CarefulPointer
from pointing.assistants import ConstantCDGain, BIGGain

from core.helpers import flatten

import sys
_str = sys.argv[1]

from check_env import check_env



if _str == 'basic-PlayNone' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')
    while True:
        sum_rewards, is_done, rewards = bundle.step()
        bundle.render('plotext')
        if is_done:
            bundle.close()
            break

if _str == 'biggain-PlayNone' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = BIGGain(operator.operator_model)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext-permanent')
    while True:
        sum_rewards, is_done, rewards = bundle.step()
        if is_done:
            bundle.render('plotext')
            bundle.close()
            break

if _str == 'basic-PlayOperator' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayOperator(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext-permanent')
    observation, sum_rewards, is_done, rewards = bundle.step([-1])
    # while True:
    #     observation, sum_rewards, is_done, rewards = bundle.step(1)
    #     if is_done:
    #         bundle.render('plotext')
    #         bundle.close()
    #         break


if _str == 'basic-PlayAssistant' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayAssistant(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext-permanent')
    observation, sum_rewards, is_done, rewards = bundle.step([1])
    # while True:
    #     observation, sum_rewards, is_done, rewards = bundle.step(1)
    #     if is_done:
    #         bundle.render('plotext')
    #         bundle.close()
    #         break

if _str == 'basic-PlayBoth' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayBoth(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext-permanent')
    observation, sum_rewards, is_done, rewards = bundle.step([[-1], [2]])
    # while True:
    #     observation, sum_rewards, is_done, rewards = bundle.step(1)
    #     if is_done:
    #         bundle.render('plotext')
    #         bundle.close()
    #         break


if _str == 'basic-TrainOperator' or _str == 'all':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayOperator(task, operator, assistant)
    env = Train(bundle)
    check_env(env)

if _str == 'basic-TrainAssistant' or _str == 'all':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayAssistant(task, operator, assistant)
    env = Train(bundle)
    check_env(env)

if _str == 'basic-TrainBoth' or _str == 'all':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayBoth(task, operator, assistant)
    env = Train(bundle)
    check_env(env)
