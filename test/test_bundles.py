import gym
import numpy

from core.bundle import PlayNone, PlayOperator, PlayAssistant, PlayBoth, Train, SinglePlayOperator, SinglePlayOperatorAuto, _DevelopTask
from pointing.envs import SimplePointingTask, Screen_v0
from pointing.operators import CarefulPointer
from pointing.assistants import ConstantCDGain, BIGGain
from eye.envs import ChenEyePointingTask
from eye.operators import ChenEye

from collections import OrderedDict
from core.models import LinearEstimatedFeedback
from core.helpers import flatten
from core.agents import BaseAgent, FHDT_LQRController, IHDT_LQRController, IHCT_LQGController
from core.observation import base_task_engine_specification, base_operator_engine_specification, RuleObservationEngine
from core.interactiontask import ClassicControlTask
from core.policy import LLDiscretePolicy, Policy
from core.space import State

import sys
_str = sys.argv[1]

from check_env import check_env


if _str == 'basic-plot':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer(task)
    assistant = ConstantCDGain(1)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')

if _str == 'basic-PlayNone' or _str == 'all':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    binary_operator = CarefulPointer(task)

    # ====================  Implementing Assistant  ===============
    # We are going to implement an assistant which has a single action (always 1). This mimics a unit constant CD Gain. No internal state is needed, and the policy is a single action. While possible, there is no need to use a LLDiscretePolicy here, and we will directly use the baseclass Policy. No observation and inference engines are needed either. However, putting a blind observation_engine instead of nothing will be more efficient (likely, not tested).

    #
    # ------- Policy -------
    action_space = [gym.spaces.Discrete(1)]
    action_set = [[1]]
    agent_policy = Policy(action_space, action_set = action_set)

    base_blind_engine_specification  =    [ ('bundle_state', 'all'),
                                        ('task_state', None),
                                        ('operator_state', None),
                                        ('assistant_state', None),
                                        ('operator_action', 'all'),
                                        ('assistant_action', 'all')
                                        ]

    observation_engine = RuleObservationEngine(deterministic_specification = base_blind_engine_specification)

    unitcdgain = BaseAgent( 'assistant',
                            policy = agent_policy,
                            observation_engine = observation_engine
                            )

    bundle = PlayNone(task, binary_operator, unitcdgain)
    game_state = bundle.reset()
    bundle.render('plotext')
    k = 0
    while True:
        k += 1
        sum_rewards, is_done, rewards = bundle.step()
        bundle.render('plotext')
        # bundle.fig.savefig("/home/jgori/Documents/img_tmp/{}.pdf".format(k))
        if is_done:
            bundle.close()
            break

if _str == 'biggain-PlayNone' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 10)
    operator = CarefulPointer()
    assistant = BIGGain(operator.operator_model)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')
    k = 0
    while True:
        bundle.fig.savefig("/home/jgori/Documents/img_tmp/{}.png".format(k))
        sum_rewards, is_done, rewards = bundle.step()
        k+= 1
        bundle.render('plotext')
        if is_done:
            bundle.fig.savefig("/home/jgori/Documents/img_tmp/{}.png".format(k))

            break

if _str == 'basic-PlayOperator' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayOperator(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')
    # observation, sum_rewards, is_done, rewards = bundle.step([-1])
    while True:
        observation, sum_rewards, is_done, rewards = bundle.step([1])
        bundle.render('plotext')
        if is_done:
            bundle.close()
            break


if _str == 'basic-PlayAssistant-0' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayAssistant(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')

    while True:
        observation, sum_rewards, is_done, rewards = bundle.step([1])
        bundle.render('plotext')
        if is_done:
            bundle.close()
            break

if _str == 'basic-PlayAssistant-1' or _str == 'all':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayAssistant(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')


    gain = 4

    sign_flag = game_state["assistant_state"]['OperatorAction'][0]
    observation = game_state
    k = 0
    while True:
        k+=1
        sign_flag = sign_flag * observation["assistant_state"]['OperatorAction'][0]
        if sign_flag == -1:
            gain = max(1,gain/2)
        observation, sum_rewards, is_done, rewards = bundle.step([gain])
        bundle.render('plotext')
        bundle.fig.savefig("/home/jgori/Documents/img_tmp/{}.pdf".format(k))

        if is_done:
            bundle.close()
            break

if _str == 'basic-PlayBoth' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayBoth(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')
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

if _str == 'chen-play':
    fitts_W = 4e-2
    fitts_D = 0.8
    ocular_std = 0.09
    swapping_std = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D, ocular_std)
    operator = ChenEye(swapping_std)
    bundle = SinglePlayOperator(task, operator)
    obs = bundle.reset()
    bundle.render('plotext')
    while True:
        action = obs['operator_state']['MuBelief']
        obs, reward, is_done, _ = bundle.step(action)
        bundle.render('plotext')
        if is_done:
            break

if _str == 'chen-play-auto':
    fitts_W = 4e-2
    fitts_D = 0.8
    ocular_std = 0
    swapping_std = 0.1
    task = ChenEyePointingTask(fitts_W, fitts_D, ocular_std)
    operator = ChenEye(swapping_std)
    bundle = SinglePlayOperatorAuto(task, operator)
    bundle.reset()
    bundle.render('plotext')
    # bundle.fig.savefig('/home/jgori/Documents/img_tmp/{}.jpg'.format(str(bundle.task.round)))
    while True:
        obs, reward, is_done, _ = bundle.step()
        bundle.render('plotext')
        # bundle.fig.savefig('/home/jgori/Documents/img_tmp/{}.jpg'.format(str(bundle.task.round)))
        if is_done:
            break

if _str == 'chen-train':
    fitts_W = 4e-2
    fitts_D = 0.5
    ocular_std = 0.09
    swapping_std = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D, ocular_std)
    operator = ChenEye(swapping_std)
    bundle = SinglePlayOperator(task, operator)
    bundle
    env = Train(bundle)
    env.squeeze_output(slice(3,5,1))
    check_env(env)

    # if __name__ == "__main__":

if _str == 'screen':
    task = Screen_v0([500,500], 10)
    bundle = _DevelopTask(task)
    bundle.reset()
    bundle.render('plot')
    bundle.step([[100,100],[1,1]])

if _str == 'LQR':
    m, d, k = 1, 1.2, 3
    Q = numpy.array([ [1,0], [0,0] ])
    R = 1e-4*numpy.array([[1]])

    F = numpy.array([   [0, 1],
                        [-k/m, -d/m]    ])

    G = numpy.array([[ 0, 1]]).reshape((-1,1))
    task = ClassicControlTask(2, 0.002, F, G, discrete_dynamics = False)
    operator = IHDT_LQRController("operator", Q, R)
    bundle = SinglePlayOperatorAuto(task, operator)
    bundle.reset()
    bundle.playspeed = 0.01
    bundle.render('plot')
    for i in range(1500):
        bundle.step()
        if not i%10:
            bundle.render("plot")

if _str == 'LQRbis':
    m, d, k = 1, 1.2, 3
    Q = numpy.diag([1, 0.01, 0, 0])
    R = numpy.array([[1e-3]])

    I = 0.25
    b = 0.2
    ta = 0.03
    te = 0.04

    a1 = b/(ta*te*I)
    a2 = 1/(ta*te) + (1/ta + 1/te)*b/I
    a3 = b/I + 1/ta + 1/te
    bu = 1/(ta*te*I)

    timestep = 0.01
    # Task dynamics
    F = numpy.array([   [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, -a1, -a2, -a3]    ])
    G = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

    task = ClassicControlTask(0.002, F, G, discrete_dynamics = False)
    operator = IHDT_LQRController("operator", Q, R)
    bundle = SinglePlayOperatorAuto(task, operator)
    bundle.reset()
    bundle.playspeed = 0.01
    bundle.render('plot')
    for i in range(1500):
        bundle.step()
        if not i%10:
            bundle.render("plot")

if _str == 'LQG':
    I = 0.25
    b = 0.2
    ta = 0.03
    te = 0.04

    a1 = b/(ta*te*I)
    a2 = 1/(ta*te) + (1/ta + 1/te)*b/I
    a3 = b/I + 1/ta + 1/te
    bu = 1/(ta*te*I)

    timestep = 0.01
    # Task dynamics
    A = numpy.array([   [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, -a1, -a2, -a3]    ])
    B = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

    # Task noise
    F = numpy.diag([0, 0, 0, 0.001])
    G = 0.03*numpy.diag([1,1,0,0])


    # Determinstic Observation Filter
    C = numpy.array([   [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]
                            ])

    # Motor and observation noise
    Gamma = numpy.array(0.08)
    D = numpy.array([   [0.01, 0, 0],
                        [0, 0.01, 0],
                        [0, 0, 0.05]
                        ])


    # Cost matrices
    Q = numpy.diag([1, 0.01, 0, 0])
    R = numpy.array([[1e-3]])
    U = numpy.diag([1, 0.1, 0.01, 0])

    task = ClassicControlTask(timestep, A, B, F = F, G = G, discrete_dynamics = False, noise = 'off')
    operator = IHCT_LQGController('operator', timestep, Q, R, U, C, Gamma, D, noise = 'on')
    bundle = SinglePlayOperator(task, operator, onreset_deterministic_first_half_step = True)
    bundle.reset( {
            'task_state': {'x':  numpy.array([[-0.5],[0],[0],[0]]) }
                    } )
    bundle.playspeed = 0.001
    bundle.render('plot')
    for i in range(1):
        bundle.step([1])
        if not i%10:
            bundle.render("plot")
        bundle.render('plot')


if _str == 'test':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
