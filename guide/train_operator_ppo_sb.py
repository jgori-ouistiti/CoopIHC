from pointing.envs import SimplePointingTask
from pointing.assistants import ConstantCDGain
from pointing.users import CarefulPointer

from coopihc.policy import Policy
from coopihc.bundle import PlayUser, Train

from gym.wrappers import FlattenObservation
from coopihc.helpers import hard_flatten


from collections import OrderedDict
import gym
import numpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


class ThisActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.N = env.action_space[0].n
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def action(self, action):
        return int(numpy.round((action * (self.N - 1e-3) / 2) + (self.N - 1) / 2)[0])


def make_env(rank, seed=0):
    def _init():
        task = SimplePointingTask(gridsize=31, number_of_targets=8)
        unitcdgain = ConstantCDGain(1)

        policy = Policy(
            action_space=[gym.spaces.Discrete(10)],
            action_set=[-5 + i for i in range(5)] + [i + 1 for i in range(5)],
            action_values=None,
        )

        user = CarefulPointer(agent_policy=policy)
        bundle = PlayUser(task, user, unitcdgain)

        observation_dict = OrderedDict(
            {
                "task_state": OrderedDict({"Position": 0}),
                "user_state": OrderedDict({"Goal": 0}),
            }
        )
        env = ThisActionWrapper(
            Train(
                bundle,
                observation_mode="multidiscrete",
                observation_dict=observation_dict,
            )
        )

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    task = SimplePointingTask(gridsize=31, number_of_targets=8)
    unitcdgain = ConstantCDGain(1)

    policy = Policy(
        action_space=[gym.spaces.Discrete(10)],
        action_set=[-5 + i for i in range(5)] + [i + 1 for i in range(5)],
        action_values=None,
    )

    user = CarefulPointer(agent_policy=policy)
    bundle = PlayUser(task, user, unitcdgain)

    observation_dict = OrderedDict(
        {
            "task_state": OrderedDict({"Position": 0}),
            "user_state": OrderedDict({"Goal": 0}),
        }
    )
    md_env = ThisActionWrapper(
        Train(
            bundle, observation_mode="multidiscrete", observation_dict=observation_dict
        )
    )

    from stable_baselines3.common.env_checker import check_env

    check_env(md_env)

    num_cpu = 3
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("guide/models/basic_pointer_ppo")
    obs = md_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = md_env.step(action)
        md_env.render("plotext")
        if done:
            break
