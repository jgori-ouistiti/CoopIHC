from pointing.envs import SimplePointingTask
from pointing.assistants import ConstantCDGain
from pointing.users import CarefulPointer

from core.policy import Policy
from core.bundle import PlayUser, Train

from gym.wrappers import FlattenObservation
from core.helpers import FlattenAction

###### ==================== Outdated

import gym
from stable_baselines3 import DQN


if __name__ == '__main__':
    task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
    unitcdgain = ConstantCDGain(1)

    policy = Policy(    action_space = [gym.spaces.Discrete(10)],
                        action_set = [-5 + i for i in range (5)] + [i+1 for i in range(5)],
                        action_values = None
    )

    user = CarefulPointer(agent_policy = policy)


    bundle = PlayUser(task, user, unitcdgain)
    env = Train(bundle)
    # print(env.bundle.reset())
    env.squeeze_output([0,9])
    env = FlattenAction(FlattenObservation(env))

    model = DQN('MlpPolicy', env, verbose = 1, exploration_fraction = 0.5)
    model.learn(total_timesteps = 1e6, log_interval = 4)
    model.save('user_model')

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render('plotext')
        if done:
            break
