from pointing.envs import SimplePointingTask
from pointing.assistants import ConstantCDGain
from pointing.operators import CarefulPointer

from core.policy import Policy
from core.bundle import PlayOperator, Train

from gym.wrappers import FlattenObservation


from collections import OrderedDict
import gym
import numpy



class ThisActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.N = env.action_space[0].n
        self.action_space = gym.spaces.Box(low  = -1, high = 1, shape = (1,))

    def action(self, action):
        (action *self.N/2) + (self.N-1)/2
        return int(numpy.round((action *self.N/2) + (self.N-1)/2)[0])


class Pointing(gym.Env):
    def __init__(self, env_config):
        task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
        unitcdgain = ConstantCDGain(1)

        policy = Policy(    action_space = [gym.spaces.Discrete(10)],
                            action_set = [-5 + i for i in range (5)] + [i+1 for i in range(5)],
                            action_values = None
        )

        operator = CarefulPointer(agent_policy = policy)
        bundle = PlayOperator(task, operator, unitcdgain)

        if env_config['observation_mode'] == 'flat':
            observation_mode = 'flat'
        elif env_config['observation_mode'] is None:
            observation_mode = None
        else:
            observation_mode = OrderedDict({'task_state': OrderedDict({'Position': 0}), 'operator_state': OrderedDict({'Goal': 0})})

        env = Train(bundle, observation_mode = observation_mode)
        env = ThisActionWrapper(env)

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

env_config = {'observation_mode': 'flat'}
flat_env = Pointing(env_config)

env_config = {'observation_mode': None}
natural_env = Pointing(env_config)

env_config = {'observation_mode': 'dict'}
dict_env = Pointing(env_config)

exit()




import ray
import ray.rllib.agents.ppo as ppo

ray.shutdown()
ray.init(ignore_reinit_error=True)

import shutil
CHECKPOINT_ROOT = "tmp/ppo/pointing"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = "/home/jgori/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

config = ppo.DEFAULT_CONFIG.copy()
config['env_config'] = {}
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=Pointing)

N_ITER = 1
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)
