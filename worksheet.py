import gym
import numpy

from core.bundle import PlayNone, PlayOperator, PlayAssistant, PlayBoth, Train, SinglePlayOperator, SinglePlayOperatorAuto, _DevelopTask
from pointing.envs import SimplePointingTask, Screen_v0
from pointing.operators import CarefulPointer, LQGPointer
from pointing.assistants import ConstantCDGain, BIGGain
from eye.envs import ChenEyePointingTask
from eye.operators import ChenEye
import core
from collections import OrderedDict
from core.models import LinearEstimatedFeedback
from core.helpers import flatten
from core.agents import BaseAgent, FHDT_LQRController, IHDT_LQRController, IHCT_LQGController, DummyAssistant, DummyOperator
from core.observation import base_task_engine_specification, base_operator_engine_specification, RuleObservationEngine, BaseObservationEngine, CascadedObservationEngine, WrapAsObservationEngine
from core.interactiontask import ClassicControlTask,  InteractionTask, TaskWrapper
from core.policy import ELLDiscretePolicy, BasePolicy, BIGDiscretePolicy, RLPolicy, WrapAsPolicy
from core.space import State, StateElement
from core.wsbundle import Server
import matplotlib.pyplot as plt
import copy
import asyncio


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
        game_state, sum_rewards, is_done, rewards = bundle.step()
        bundle.render('plotext')
        # bundle.fig.savefig("/home/jgori/Documents/img_tmp/{}.pdf".format(k))
        if is_done:
            bundle.close()
            break

if _str == 'biggain-PlayNone' or _str == 'all':
    task = SimplePointingTask(gridsize = 15, number_of_targets = 8, mode = 'position')
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
        game_state, sum_rewards, is_done, rewards = bundle.step()
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

    policy = BasePolicy(    action_space = [core.space.Discrete(10)],
                        action_set = [[-5 + i for i in range (5)] + [i+1 for i in range(5)]],
                        action_values = None
    )

    operator = CarefulPointer(agent_policy = policy)
    bundle = PlayOperator(task, operator, unitcdgain)
    observation = bundle.reset()

    observation_dict = OrderedDict({'task_state': OrderedDict({'position': 0}), 'operator_state': OrderedDict({'goal': 0})})


    class ThisActionWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.N = env.action_space[0].n
            self.action_space = gym.spaces.Box(low  = -1, high = 1, shape = (1,))

        def action(self, action):
            return int(numpy.round((action *(self.N-1e-3)/2) + (self.N-1)/2)[0])


    md_env = ThisActionWrapper( Train(
            bundle,
            observation_mode = 'multidiscrete',
            observation_dict = observation_dict
            ))



    from stable_baselines3.common.env_checker import check_env
    check_env(md_env)

    obs = md_env.reset()
    print(obs)
    print(md_env.bundle.game_state)
    # =============
    def make_env(rank, seed = 0):
        def _init():
            task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
            unitcdgain = ConstantCDGain(1)

            policy = BasePolicy(    action_space = [core.space.Discrete(10)],
                                action_set = [[-5 + i for i in range (5)] + [i+1 for i in range(5)]],
                                action_values = None
            )

            operator = CarefulPointer(agent_policy = policy)
            bundle = PlayOperator(task, operator, unitcdgain)

            observation_dict = OrderedDict({'task_state': OrderedDict({'position': 0}), 'operator_state': OrderedDict({'goal': 0})})
            env = ThisActionWrapper( Train(
                    bundle,
                    observation_mode = 'multidiscrete',
                    observation_dict = observation_dict
                    ))

            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init
    # =============

    if __name__ == '__main__':
        print('inside main')
        num_cpu = 3
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

        model = PPO('MlpPolicy', env, verbose=1)
        print('start training')
        model.learn(total_timesteps=10000)



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

    policy = BasePolicy(    action_space = [core.space.Discrete(10)],
                        action_set = [[-5 + i for i in range (5)] + [i+1 for i in range(5)]],
                        action_values = None
    )

    operator = CarefulPointer(agent_policy = policy)
    bundle = PlayOperator(task, operator, unitcdgain)

    observation_dict = OrderedDict({'task_state': OrderedDict({'position': 0}), 'operator_state': OrderedDict({'goal': 0})})
    training_env = ThisActionWrapper( Train(
            bundle,
            observation_mode = 'multidiscrete',
            observation_dict = observation_dict
            ))

    # ----------------- Training environment creation finished


    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    unitcdgain = ConstantCDGain(1)

    # specifying operator policy
    observation_dict = OrderedDict({'task_state': OrderedDict({'position': 0}), 'operator_state': OrderedDict({'goal': 0})})
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
    perceptualnoise = 0.15
    oculomotornoise = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye(perceptualnoise, oculomotornoise, dimension = 1)
    bundle = SinglePlayOperator(task, operator)
    obs = bundle.reset()
    bundle.render('plotext')
    while True:
        _action = copy.deepcopy(obs['operator_state']['belief'])
        action = _action[0]
        noise_obs = State()
        noise_obs['task_state'] = State()
        noise_obs['task_state']['targets'] = action
        noise_obs['task_state']['fixation'] = obs['task_state']['fixation']
        noise = operator.eccentric_noise_gen(noise_obs, oculomotornoise)[0]

        noisy_action = action + noise
        obs, reward, is_done, _ = bundle.step(noisy_action)
        bundle.render('plotext')
        print(is_done)
        if is_done:
            break

if _str == 'chen-play':
    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.15
    oculomotornoise = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D)
    operator = ChenEye(perceptualnoise, oculomotornoise)
    bundle = SinglePlayOperator(task, operator)
    obs = bundle.reset()
    bundle.render('plotext')
    while True:
        _action = copy.deepcopy(obs['operator_state']['belief'])
        action = _action[0]
        noise_obs = State()
        noise_obs['task_state'] = State()
        noise_obs['task_state']['targets'] = action
        noise_obs['task_state']['fixation'] = obs['task_state']['fixation']
        noise = operator.eccentric_noise_gen(noise_obs, oculomotornoise)[0]
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
    class oldpositionMemorizedSimplePointingTask(SimplePointingTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memorized = None

        def reset(self, dic = {}):
            super().reset(dic = dic)
            self.state['oldposition'] = copy.deepcopy(self.state['position'])

        def operator_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['position'])
            obs, rewards, is_done, _doc = super().operator_step(*args, **kwargs)
            obs['oldposition'] = self.memorized
            return obs, rewards, is_done, _doc

        def assistant_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['position'])
            obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
            obs['oldposition'] = self.memorized
            return obs, rewards, is_done, _doc

    logger.info('Pointing task definition')
    pointing_task = oldpositionMemorizedSimplePointingTask(gridsize = 31, number_of_targets = 8, mode = 'position')
    # bundle = _DevelopTask(pointing_task)
    # bundle.reset()

    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.2
    oculomotornoise = 0.2
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye( perceptualnoise, oculomotornoise, dimension = 1)
    obs_bundle = SinglePlayOperatorAuto(task, operator, start_at_action = True)



    class ChenEyeObservationEngineWrapper(WrapAsObservationEngine):
        def __init__(self, obs_bundle):
            super().__init__(obs_bundle)

        def observe(self, game_state):
            # set observation bundle to the right state and cast it to the right space
            target = game_state['task_state']['position'].cast(self.game_state['task_state']['targets'])
            fixation = game_state['task_state']['oldposition'].cast(self.game_state['task_state']['fixation'])
            reset_dic = {'task_state':
                            {   'targets': target,
                                'fixation': fixation    }
                        }
            self.reset(dic = reset_dic)

            # perform the run
            is_done = False
            rewards = 0
            while True:
                obs, reward, is_done, _doc = self.step()
                rewards += reward
                if is_done:
                    break


            # cast back to initial space and return
            obs['task_state']['fixation'].cast(game_state['task_state']['oldposition'])
            obs['task_state']['targets'].cast(game_state['task_state']['position'])

            return game_state, rewards

    # Define cascaded observation engine
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
        obs, reward, is_done, reward_list = bundle.step()
        print(reward)
        rewards.append(reward_list)
        bundle.render('plotext')
        if is_done:
            break


if _str == 'LQR':
    m, d, k = 1, 1.2, 3
    Q = numpy.array([ [1,0], [0,0] ])
    R = 1e-4*numpy.array([[1]])

    Ac = numpy.array([   [0, 1],
                        [-k/m, -d/m]    ])

    Bc = numpy.array([[ 0, 1]]).reshape((-1,1))
    task = ClassicControlTask(0.002, Ac, Bc, discrete_dynamics = False)
    operator = IHDT_LQRController("operator", Q, R, None)
    bundle = SinglePlayOperatorAuto(task, operator)
    bundle.reset()
    bundle.playspeed = 0.01
    # bundle.render('plot')
    for i in range(1500):
        bundle.step()
        print(i)
        if not i%20:
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
    operator = IHDT_LQRController("operator", Q, R, None)
    bundle = SinglePlayOperatorAuto(task, operator)
    bundle.reset()
    bundle.playspeed = 0.01
    bundle.render('plot')
    for i in range(1500):
        bundle.step()
        if not i%20:
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
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]
                            ])

    # Motor and observation noise
    Gamma = numpy.array(0.08)
    D = numpy.array([   [0.01, 0, 0, 0],
                        [0, 0.01, 0, 0],
                        [0, 0, 0.05, 0],
                        [0, 0, 0, 0]
                        ])


    # Cost matrices
    Q = numpy.diag([1, 0.01, 0, 0])
    R = numpy.array([[1e-3]])
    U = numpy.diag([1, 0.1, 0.01, 0])

    task = ClassicControlTask(timestep, Ac, Bc, F = F, G = G, discrete_dynamics = False, noise = 'on')
    operator = IHCT_LQGController('operator', timestep, Q, R, U, C, Gamma, D, noise = 'on')
    bundle = SinglePlayOperatorAuto(task, operator, onreset_deterministic_first_half_step = True,
    start_at_action = True)
    obs = bundle.reset( dic = {
            'task_state': {'x':  numpy.array([[-0.5],[0],[0],[0]]) },
            'operator_state': {'xhat':  numpy.array([[-0.5],[0],[0],[0]]) }
                    } )
    bundle.playspeed = 0.001
    bundle.render('plot')
    for i in range(250):
        bundle.step()
        print(i)
        if not i%3:
            bundle.render("plot")

if _str == 'LQGpointer':
    # Malfunctions (pointer can not go left)
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
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]
                            ])

    # Motor and observation noise
    Gamma = numpy.array(0.08)
    D = numpy.array([   [0.01, 0, 0, 0],
                        [0, 0.01, 0, 0],
                        [0, 0, 0.05, 0],
                        [0, 0, 0, 0]
                        ])


    # Cost matrices
    Q = numpy.diag([1, 0.01, 0, 0])
    R = numpy.array([[1e-3]])
    U = numpy.diag([1, 0.1, 0.01, 0])

    task = ClassicControlTask(timestep, Ac, Bc, F = F, G = G, discrete_dynamics = False, noise = 'off')
    operator = IHCT_LQGController('operator', timestep, Q, R, U, C, Gamma, D, noise = 'on')
    action_bundle = SinglePlayOperatorAuto(task, operator, onreset_deterministic_first_half_step = True,
    start_at_action = True)



    class LQGPointerPolicy(WrapAsPolicy):
        def __init__(self, action_bundle, *args, **kwargs):
            action_state = State()
            action_state['action'] = StateElement(
                values = [None],
                spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,1))])
            super().__init__(action_bundle, action_state, *args, **kwargs)

        def sample(self):
            logger.info('=============== Entering Sampler ================')
            cursor = copy.copy(self.observation['task_state']['position'])
            target = copy.copy(self.observation['operator_state']['goal'])
            # allow temporarily
            cursor.mode = 'warn'
            target.mode = 'warn'

            tmp_box = StateElement( values = [None],
                spaces = gym.spaces.Box(-self.host.bundle.task.gridsize+1, self.host.bundle.task.gridsize-1 , shape = (1,)),
                possible_values = [[None]],
                clipping_mode = 'warning')

            cursor_box = StateElement( values = [None],
                spaces = gym.spaces.Box(-.5, .5, shape = (1,)),
                possible_values = [[None]],
                clipping_mode = 'warning')


            tmp_box['values'] = [numpy.array(v) for v in (target-cursor)['values']]
            init_dist = tmp_box.cast(cursor_box)['values'][0]

            _reset_x = self.xmemory
            _reset_x[0] = init_dist
            _reset_x_hat = self.xhatmemory
            _reset_x_hat[0] = init_dist
            action_bundle.reset( dic = {
            'task_state': {'x':  _reset_x },
            'operator_state': {'xhat': _reset_x_hat}
                    } )

            total_reward = 0
            N = int(pointing_task.timestep/task.timestep)

            for i in range(N):
                observation, sum_rewards, is_done, rewards = self.step()
                print(observation)
                total_reward += sum_rewards
                if is_done:
                    break

            # Store state for next usage
            self.xmemory = observation['task_state']['x']['values'][0]
            self.xhatmemory = observation['operator_state']['xhat']['values'][0]

            # Cast as delta in correct units
            cursor_box['values'] = - self.xmemory[0] + init_dist
            delta = cursor_box.cast(tmp_box)
            possible_values = [-30 + i for i in range(61)]
            value = possible_values.index(int(numpy.round(delta['values'][0])))
            action = StateElement(values = value, spaces = core.space.Discrete(61), possible_values = [possible_values])
            logger.info('{} Selected action {}'.format(self.__class__.__name__, str(action)))

            return action, total_reward

        def reset(self):
            self.xmemory = numpy.array([[0.0],[0.0],[0.0],[0.0]])
            self.xhatmemory = numpy.array([[0.0],[0.0],[0.0],[0.0]])


    pointing_task = SimplePointingTask(gridsize = 31, number_of_targets = 8, mode = 'gain')
    policy = LQGPointerPolicy(action_bundle)
    binary_operator = CarefulPointer( agent_policy = policy )
    unitcdgain = ConstantCDGain(1)
    bundle = PlayNone(pointing_task, binary_operator, unitcdgain)
    game_state = bundle.reset()
    bundle.render('plotext')

    while True:
        obs, rewards, is_done, _ = bundle.step()
        bundle.render('plotext')
        if is_done:
            break



if _str == 'LQGpointer-chenobs':
    # Error linked to LQGpointer
    # Add a state to the SimplePointingTask to memorize the old position
    class oldpositionMemorizedSimplePointingTask(SimplePointingTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memorized = None

        def reset(self, dic = {}):
            super().reset(dic = dic)
            self.state['oldposition'] = copy.deepcopy(self.state['position'])

        def operator_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['position'])
            obs, rewards, is_done, _doc = super().operator_step(*args, **kwargs)
            obs['oldposition'] = self.memorized
            return obs, rewards, is_done, _doc

        def assistant_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['position'])
            obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
            obs['oldposition'] = self.memorized
            return obs, rewards, is_done, _doc

    pointing_task = oldpositionMemorizedSimplePointingTask(gridsize = 31, number_of_targets = 8, mode = 'gain')


    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.2
    oculomotornoise = 0.2
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye( perceptualnoise, oculomotornoise, dimension = 1)
    obs_bundle = SinglePlayOperatorAuto(task, operator, start_at_action = True)



    class ChenEyeObservationEngineWrapper(WrapAsObservationEngine):
        def __init__(self, obs_bundle):
            super().__init__(obs_bundle)

        def observe(self, game_state):
            # set observation bundle to the right state and cast it to the right space
            target = game_state['task_state']['position'].cast(self.game_state['task_state']['targets'])
            fixation = game_state['task_state']['oldposition'].cast(self.game_state['task_state']['fixation'])
            reset_dic = {'task_state':
                            {   'targets': target,
                                'fixation': fixation    }
                        }
            self.reset(dic = reset_dic)

            # perform the run
            is_done = False
            rewards = 0
            while True:
                obs, reward, is_done, _doc = self.step()
                rewards += reward
                if is_done:
                    break


            # cast back to initial space and return
            obs['task_state']['fixation'].cast(game_state['task_state']['oldposition'])
            obs['task_state']['targets'].cast(game_state['task_state']['position'])

            return game_state, rewards

    # Define cascaded observation engine
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

    lqg_operator = LQGPointer(observation_engine = observation_engine)
    unitcdgain = ConstantCDGain(1)
    bundle = PlayNone(pointing_task, lqg_operator, unitcdgain)
    game_state = bundle.reset()
    bundle.render('plotext')

    while True:
        obs, rewards, is_done, _ = bundle.step()
        print(rewards, _)
        bundle.render('plotext')
        if is_done:
            break



if _str == 'screen':
    task = Screen_v0([800,500], 10,
        target_radius = 1e-1        )
    goal = StateElement(values = numpy.array([0,0]),
                        spaces = [gym.spaces.Box(low = numpy.array([-1, -1/task.aspect_ratio]), high = numpy.array([1, 1/task.aspect_ratio]), shape = (2,) )],
                        possible_values = [[None]]
                        )

    _state = State()
    _state['goal'] = goal
    operator = DummyOperator(state = _state)
    bundle = _DevelopTask(task, operator = operator)
    bundle.reset()
    bundle.render('plot')
    bundle.step([ numpy.array([-0.2,-0.2]), numpy.array([1,1]) ])
    bundle.render('plot')

if _str == 'js-ws':
    from pointing.envs import DiscretePointingTaskPipeWrapper
    task = SimplePointingTask(gridsize = 20, number_of_targets = 5)

    policy = ELLDiscretePolicy(action_space = [core.space.Discrete(3)], action_set = [[-1, 0, 1]])
    # Actions are in human values, i.e. they are not necessarily in range(0,N)
    def compute_likelihood(self, action, observation):
        # convert actions and observations
        action = action['human_values'][0]
        goal = observation['operator_state']['goal']['values'][0]
        position = observation['task_state']['position']['values'][0]

        # Write down all possible cases (5)
        # (1) Goal to the right, positive action
        if goal > position and action > 0 :
            return .99
        # (2) Goal to the right, negative action
        elif goal > position and action <= 0 :
            return .005
        # (3) Goal to the left, positive action
        if goal < position and action >= 0 :
            return .005
        # (4) Goal to the left, negative action
        elif goal < position and action < 0 :
            return .99
        elif goal == position and action == 0:
            return 1
        elif goal == position and action != 0:
            return 0
        else:
            print(goal, position, action)
            raise RunTimeError("warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition")

    # Attach likelihood function to the policy
    policy.attach_likelihood_function(compute_likelihood)

    operator = CarefulPointer(agent_policy = policy)
    assistant = ConstantCDGain(1)
    bundle = PlayNone(task, operator, assistant)
    server = Server(bundle, DiscretePointingTaskPipeWrapper, address='localhost', port = 4000)
    server.start()


# WIP
if _str == 'js-2d':
    from pointing.envs import DiscretePointingTask, DiscretePointingTaskPipeWrapper
    from pointing.operators import TwoDCarefulPointer
    task = DiscretePointingTask(dim = 2, gridsize = (31,31), number_of_targets = 5)
    operator = TwoDCarefulPointer()
    assistant = ConstantCDGain(1)
    bundle = PlayNone(task, operator, assistant)
    bundle.reset()
    server = Server(bundle, DiscretePointingTaskPipeWrapper, address='localhost', port = 4000)
    server.start()


if _str == 'ndtask':
    from pointing.envs import DiscretePointingTask
    from pointing.operators import TwoDCarefulPointer
    task = DiscretePointingTask(dim = 2, gridsize = (31,31), number_of_targets = 5)
    operator = TwoDCarefulPointer()
    assistant = ConstantCDGain(1)
    bundle = PlayNone(task, operator, assistant)
    bundle.reset()
    bundle.render('text')
    while True:
        state, rewards, is_done, reward_list = bundle.step()
        print(state['task_state']['position']['values'])
        if is_done:
            break










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