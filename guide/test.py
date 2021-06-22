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
from core.agents import BaseAgent, FHDT_LQRController, IHDT_LQRController, IHCT_LQGController, PackageToOperator, DummyAssistant
from core.observation import base_task_engine_specification, base_operator_engine_specification, RuleObservationEngine, ObservationEngine, CascadedObservationEngine
from core.interactiontask import ClassicControlTask,  InteractionTask, TaskWrapper
from core.policy import ELLDiscretePolicy, Policy, BIGDiscretePolicy, RLPolicy
from core.space import State
import matplotlib.pyplot as plt
import copy

import sys
_str = sys.argv[1]

from loguru import logger
logger.remove()
try:
    import os
    os.remove('logs/{}.log'.format(_str))
except FileNotFoundError:
    pass
logger.add('logs/{}.log'.format(_str), format = "{time} {level} {message}")

if _str == 'basic-plot':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayNone(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')

if _str == 'basic-PlayNone' or _str == 'all':

    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)


    binary_operator = CarefulPointer()
    unitcdgain = ConstantCDGain(1)
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
    task = SimplePointingTask(gridsize = 31, number_of_targets = 10, mode = 'position')
    binary_operator = CarefulPointer()

    BIGpointer = BIGGain()

    bundle = PlayNone(task, binary_operator, BIGpointer)

    game_state = bundle.reset()
    bundle.render('plotext')
    plt.tight_layout()
    # k = 0
    # plt.savefig('/home/jgori/Documents/img_tmp/biggain_{}.png'.format(k))
    #
    while True:
        sum_rewards, is_done, rewards = bundle.step()
        bundle.render('plotext')
        # k+=1

        # plt.savefig('/home/jgori/Documents/img_tmp/biggain_{}.png'.format(k))

        if is_done:
            break


if _str == 'basic-TrainOperator' or _str == 'all':
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    unitcdgain = ConstantCDGain(1)

    policy = Policy(    action_space = [gym.spaces.Discrete(10)],
                        action_set = [-5 + i for i in range (5)] + [i+1 for i in range(5)],
                        action_values = None
    )

    operator = CarefulPointer(agent_policy = policy)
    bundle = PlayOperator(task, operator, unitcdgain)

    observation_dict = OrderedDict({'task_state': OrderedDict({'Position': 0}), 'operator_state': OrderedDict({'Goal': 0})})
    md_env = ThisActionWrapper( Train(
            bundle,
            observation_mode = 'multidiscrete',
            observation_dict = observation_dict
            ))



    from stable_baselines3.common.env_checker import check_env
    check_env(md_env)
    num_cpu = 3
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=100000)



if _str == 'loadNNpolicy' or _str == 'all':



    # Two cases:
    #     1. The NN policy was obtained via bundles, in which case you should just recreated the training environment and pass it to the policy constructor
    #     2. The NN policy was obtained via an other medium, in which case you should write wrappers to ensure that observations by the environment are compatible with the NN you are using. It should be easy to write one or two functions to help with this, but for now, nothing is done.



    # ----------------- Recreate the training environment

    class ThisActionWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.N = env.action_space[0].n
            self.action_space = gym.spaces.Box(low  = -1, high = 1, shape = (1,))


        def action(self, action):
            return int(numpy.round((action *(self.N-1e-3)/2) + (self.N-1)/2)[0])

    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    unitcdgain = ConstantCDGain(1)

    policy = Policy(    action_space = [gym.spaces.Discrete(10)],
                        action_set = [-5 + i for i in range (5)] + [i+1 for i in range(5)],
                        action_values = None
    )

    operator = CarefulPointer(agent_policy = policy)
    bundle = PlayOperator(task, operator, unitcdgain)

    observation_dict = OrderedDict({'task_state': OrderedDict({'Position': 0}), 'operator_state': OrderedDict({'Goal': 0})})
    training_env = ThisActionWrapper( Train(
            bundle,
            observation_mode = 'multidiscrete',
            observation_dict = observation_dict
            ))

    # ----------------- Training environment creatin finished


    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    unitcdgain = ConstantCDGain(1)

    # specifying operator policy
    observation_dict = OrderedDict({'task_state': OrderedDict({'Position': 0}), 'operator_state': OrderedDict({'Goal': 0})})
    action_wrappers = OrderedDict()
    action_wrappers['ThisActionWrapper'] = (ThisActionWrapper, ())
    policy = RLPolicy(
            'operator',
            model_path =  'guide/models/basic_pointer_ppo',
            learning_algorithm = 'PPO',
            library = 'stable_baselines3',
            training_env = training_env,
            wrappers = {'actionwrappers': action_wrappers, 'observation_wrappers': {}}
              )
    operator = CarefulPointer(agent_policy = policy)

    bundle = PlayAssistant(task, operator, unitcdgain)
    game_state = bundle.reset()
    bundle.render('plotext')
    plt.tight_layout()
    while True:
        observation, sum_reward, is_done, rewards = bundle.step([0])
        bundle.render('plotext')
        if is_done:
            break

if _str == 'chen-play-1D':
    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.2
    oculomotornoise = 0.2
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye(perceptualnoise, oculomotornoise, dimension = 1)
    bundle = SinglePlayOperator(task, operator)


    bundle.render('plotext')
    while True:
        _action = copy.deepcopy(obs['operator_state']['belief'])
        action = _action[0]
        noise_obs = State()
        noise_obs['task_state'] = State()
        noise_obs['task_state']['Targets'] = action
        noise_obs['task_state']['Fixation'] = obs['task_state']['Fixation']
        noise = operator.eccentric_noise_gen(noise_obs, ocular_std)[0]
        noisy_action = action + noise
        obs, reward, is_done, _ = bundle.step(noisy_action)
        bundle.render('plotext')
        print(is_done)
        if is_done:
            break

if _str == 'chen-play':
    fitts_W = 4e-2
    fitts_D = 0.8
    ocular_std = 0.09
    swapping_std = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D)
    operator = ChenEye(swapping_std, ocular_std)
    bundle = SinglePlayOperator(task, operator)
    obs = bundle.reset()
    bundle.render('plotext')
    while True:
        _action = copy.deepcopy(obs['operator_state']['belief'])
        action = _action[0]
        noise_obs = State()
        noise_obs['task_state'] = State()
        noise_obs['task_state']['Targets'] = action
        noise_obs['task_state']['Fixation'] = obs['task_state']['Fixation']
        noise = operator.eccentric_noise_gen(noise_obs, ocular_std)[0]
        noisy_action = action + noise
        obs, reward, is_done, _ = bundle.step(noisy_action)
        bundle.render('plotext')
        print(is_done)
        if is_done:
            break



if _str == 'chen-play-auto':
    fitts_W = 4e-2
    fitts_D = 0.8
    ocular_std = 0.09
    swapping_std = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D)
    operator = ChenEye(swapping_std, ocular_std)
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



if _str == 'careful-with-obs':

    # Add a state to the SimplePointingTask to memorize the old position
    class OldPositionMemorizedSimplePointingTask(SimplePointingTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memorized = None

        def reset(self, dic = {}):
            super().reset(dic = dic)
            self.state['OldPosition'] = copy.deepcopy(self.state['Position'])

        def operator_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['Position'])
            obs, rewards, is_done, _doc = super().operator_step(*args, **kwargs)
            obs['OldPosition'] = self.memorized
            return obs, rewards, is_done, _doc

        def assistant_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['Position'])
            obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
            obs['OldPosition'] = self.memorized
            return obs, rewards, is_done, _doc

    logger.info('Pointing task definition')
    pointing_task = OldPositionMemorizedSimplePointingTask(gridsize = 31, number_of_targets = 8, mode = 'position')
    # bundle = _DevelopTask(pointing_task)
    # bundle.reset()

    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.09
    oculomotornoise = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye( perceptualnoise, oculomotornoise, dimension = 1)
    obs_bundle = SinglePlayOperatorAuto(task, operator, start_at_action = True)

    # reset_dic = {'task_state':
    #                 {   'Targets': .5,
    #                     'Fixation': -.5    }
    #             }
    # obs_bundle.reset(reset_dic)

    class ChenEyeObservationEngineWrapper(ObservationEngine):
        """ Not impleted yet.
        """
        def __init__(self, obs_bundle):
            super().__init__()
            self.type = 'process'
            self.obs_bundle = obs_bundle
            self.obs_bundle.reset()

        def observe(self, game_state):
            # Cast to the box of the obs bundle
            target = game_state['task_state']['Position'].cast(self.obs_bundle.game_state['task_state']['Targets'], inplace = False)
            fixation = game_state['task_state']['OldPosition'].cast(self.obs_bundle.game_state['task_state']['Fixation'], inplace = False)
            reset_dic = {'task_state':
                            {   'Targets': target,
                                'Fixation': fixation    }
                        }

            self.obs_bundle.reset(dic = reset_dic)
            is_done = False
            rewards = 0
            while True:
                obs, reward, is_done, _doc = self.obs_bundle.step()
                rewards += reward
                if is_done:
                    break
            obs['task_state']['Fixation'].cast(game_state['task_state']['OldPosition'], inplace = True)
            obs['task_state']['Targets'].cast(game_state['task_state']['Position'], inplace = True)
            return game_state, rewards





    cursor_tracker = ChenEyeObservationEngineWrapper(obs_bundle)
    base_operator_engine_specification  =    [ ('turn_index', 'all'),
                                        ('task_state', 'all'),
                                        ('operator_state', 'all'),
                                        ('assistant_state', None),
                                        ('operator_action', 'all'),
                                        ('assistant_action', 'all')
                                        ]
    default_observation_engine = RuleObservationEngine(
            deterministic_specification = base_operator_engine_specification,
            )

    observation_engine = CascadedObservationEngine([cursor_tracker, default_observation_engine])
    binary_operator = CarefulPointer(observation_engine = observation_engine)
    BIGpointer = BIGGain()
    bundle = PlayAssistant(pointing_task, binary_operator, BIGpointer)


    bundle = PlayNone(pointing_task, binary_operator, BIGpointer)
    game_state = bundle.reset()
    bundle.render('plotext')
    rewards = []
    while True:
        reward, is_done, reward_list = bundle.step()
        rewards.append(reward_list)
        bundle.render('plotext')
        if is_done:
            break

    # exit()
    #
    # bundle.render('plotext')
    # operator_obs_reward, operator_infer_reward, first_task_reward, is_done = bundle._operator_step()
    # print(is_done)
    # assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = bundle._assistant_step()
    # print(is_done)
    # bundle.render('plotext')



    # rewards = 0
    # self = binary_operator.observation_engine
    # engine = self.engine_list[0]
    # game_state = bundle.game_state
    # obs, reward = engine.observe(game_state)
    # game_state.update(obs)
    # engine = self.engine_list[1]
    # obs, reward = engine.observe(game_state)
    # game_state.update(obs)
    # exit()

    # target = game_state['task_state']['Position'].cast(obs_bundle.game_state['task_state']['Targets'], inplace = False)
    # fixation = game_state['task_state']['OldPosition'].cast(obs_bundle.game_state['task_state']['Fixation'], inplace = False)
    #
    # reset_dic = {'task_state':
    #                 {   'Targets': target,
    #                     'Fixation': fixation    }
    #             }
    #
    #
    # obs_bundle.reset(reset_dic)
    # obs_bundle.render('plotext')
    #
    # obs_bundle.task.reset(reset_dic['task_state'])
    # obs_bundle.operator.reset(None)
    # print(obs_bundle.game_state)
    # agent_observation, agent_reward = obs_bundle.operator.observe(obs_bundle.game_state)
    # obs_bundle.operator.inference_engine.add_observation(agent_observation)
    # agent_state, agent_infer_reward = obs_bundle.operator.inference_engine.infer()
    # exit()



if _str == 'screen':
    task = Screen_v0([800,500], 10,
        target_radius = 1e-1        )
    bundle = _DevelopTask(task)
    bundle.reset()
    bundle.render('plot')
    bundle.step([ numpy.array([-0.2,-0.2]), numpy.array([1,1]) ])


















# ============ Outdated




if _str == 'basic-PlayOperator' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    operator = CarefulPointer()
    assistant = ConstantCDGain(1)

    bundle = PlayOperator(task, operator, assistant)
    game_state = bundle.reset()
    bundle.render('plotext')
    # observation, sum_rewards, is_done, rewards = bundle.step([-1])
    # while True:
    #     observation, sum_rewards, is_done, rewards = bundle.step([1])
    #     bundle.render('plotext')
    #     if is_done:
    #         bundle.close()
    #         break


if _str == 'basic-PlayAssistant-0' or _str == 'all':
    # Checking Evaluate with CarefulPointer, ConstantCDGain and SimplePointingTask
    task = SimplePointingTask(gridsize = 31, number_of_targets = 10)
    binary_operator = CarefulPointer()

    BIGpointer = BIGGain()

    bundle = PlayAssistant(task, binary_operator, BIGpointer)

    game_state = bundle.reset()
    bundle.render('plotext')
    assistant_obs_reward, assistant_infer_reward = bundle._assistant_first_half_step()
    print(bundle.game_state)
    # assistant_action = bundle.assistant.policy.sample()
    exit()

    while True:
        sum_rewards, is_done, rewards = bundle.step([1])
        bundle.render('plotext')
        if is_done:
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



if _str == 'LQR':
    m, d, k = 1, 1.2, 3
    Q = numpy.array([ [1,0], [0,0] ])
    R = 1e-4*numpy.array([[1]])

    Ac = numpy.array([   [0, 1],
                        [-k/m, -d/m]    ])

    Bc = numpy.array([[ 0, 1]]).reshape((-1,1))
    task = ClassicControlTask(0.002, Ac, Bc, discrete_dynamics = False)
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
    Ac = numpy.array([   [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, -a1, -a2, -a3]    ])
    Bc = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

    task = ClassicControlTask(0.002, Ac, Bc, discrete_dynamics = False)
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
    Ac = numpy.array([   [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, -a1, -a2, -a3]    ])
    Bc = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

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

    task = ClassicControlTask(timestep, Ac, Bc, F = F, G = G, discrete_dynamics = False, noise = 'off')
    operator = IHCT_LQGController('operator', timestep, Q, R, U, C, Gamma, D, noise = 'on')
    bundle = SinglePlayOperatorAuto(task, operator, onreset_deterministic_first_half_step = True)
    bundle.reset( {
            'task_state': {'x':  numpy.array([[-0.5],[0],[0],[0]]) }
                    } )
    bundle.playspeed = 0.001
    bundle.render('plot')
    for i in range(1500):
        bundle.step()
        if not i%10:
            bundle.render("plot")
        bundle.render('plot')



if _str == 'test':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
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
    Ac = numpy.array([   [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, -a1, -a2, -a3]    ])
    Bc = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

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

    classic_cc_task = ClassicControlTask(timestep, Ac, Bc, F = F, G = G, discrete_dynamics = False, noise = 'off')
    LQG_operator = IHCT_LQGController('operator', timestep, Q, R, U, C, Gamma, D, noise = 'on')
    op_bundle = SinglePlayOperatorAuto(classic_cc_task, LQG_operator, onreset_deterministic_first_half_step = True)



    task = Screen_v0([800,600], 10)
    bundle = _DevelopTask(task)
    #
    #
    #
    #
    # packaged_operator = PackageToOperator(op_bundle)
    # assistant = ConstantCDGain(1)
    #
    # bundle = PLayNone(task, )
