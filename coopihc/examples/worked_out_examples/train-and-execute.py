# imports from Coopihc
from pointing.envs import SimplePointingTask
from pointing.users import CarefulPointer
from pointing.assistants import ConstantCDGain
from coopihc.policy import BasePolicy, RLPolicy
import coopihc
from coopihc.bundle import PlayUser, PlayNone, Train

# other imports
from collections import OrderedDict
import gym
import numpy
import matplotlib.pyplot as plt

# stables baselines 2 seems much faster for some reason.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import loguru

loguru.logger.remove()


# Pointing task
task = SimplePointingTask(gridsize=31, number_of_targets=8)
# Unit cd gain assistant
unitcdgain = ConstantCDGain(1)
# The policy defines the action set that we are going to use for the
# user. The BasePolicy randomly samples actions from the action set.
# But that is okay because we don't sample from the policy during
# learning
policy = BasePolicy(
    action_space=[coopihc.space.Discrete(10)],
    action_set=[[-5 + i for i in range(5)] + [i + 1 for i in range(5)]],
    action_values=None,
)
# Re-use the previously defined user model called CarefulPointer, but
# override its policy
user = CarefulPointer(agent_policy=policy)
# Bundle the pointing task, the user model and the assistant in a POMDP, where assistant actions are queried from its policy. The bundle expects to be fed user actions
bundle = PlayUser(task, user, unitcdgain)
# Initialize the bundle to some random state
observation = bundle.reset()
# Only observe the position (task state) and goal (user state) states.
# The rest is uninformative and will slow down training.
observation_dict = OrderedDict(
    {"task_state": OrderedDict({"position": 0}), "user_state": OrderedDict({"goal": 0})}
)

# We are going to use PPO to solve the POMDP, which works with
# continuous action spaces, but our policy has discrete actions. So we introduce a wrapper to pass between continuous and discrete spaces. This is classical in RL, and does not depend on Coopihc
class ThisActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.N = env.action_space[0].n
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def action(self, action):
        return int(numpy.round((action * (self.N - 1e-3) / 2) + (self.N - 1) / 2)[0])


# First wrap the bundle in a train environment to make it compatible
# with gym, then wrap it up in a gym wrapper:
# ActionWrapper < Train < bundle > >
md_env = ThisActionWrapper(
    Train(bundle, observation_mode="multidiscrete", observation_dict=observation_dict)
)


# Verify that the environment is compatible with stable_baselines
from stable_baselines3.common.env_checker import check_env

check_env(md_env)


# Wrap everything above in a make_env function, used to parallelize
# the environments (to sample faster from the environments).
# Everything above is just to explain, only the code inside this
# function is needed to define envs.
def make_env(rank, seed=0):
    def _init():
        task = SimplePointingTask(gridsize=31, number_of_targets=8)
        unitcdgain = ConstantCDGain(1)

        policy = BasePolicy(
            action_space=[coopihc.space.Discrete(10)],
            action_set=[[-5 + i for i in range(5)] + [i + 1 for i in range(5)]],
            action_values=None,
        )

        user = CarefulPointer(agent_policy=policy)
        bundle = PlayUser(task, user, unitcdgain)

        observation_dict = OrderedDict(
            {
                "task_state": OrderedDict({"position": 0}),
                "user_state": OrderedDict({"goal": 0}),
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
    # parallelize the environments
    env = SubprocVecEnv([make_env(i) for i in range(4)])
    # define the learning algorithm
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
    # train the algorithm
    print("start training")
    model.learn(total_timesteps=6000)
    model.save("saved_model")

    # Trained policy is now saved

    # Reusing the trained and saved policy:

    # ================ Recreate the training environment ()
    class ThisActionWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.N = env.action_space[0].n
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        def action(self, action):
            return int(
                numpy.round((action * (self.N - 1e-3) / 2) + (self.N - 1) / 2)[0]
            )

    task = SimplePointingTask(gridsize=31, number_of_targets=8)
    unitcdgain = ConstantCDGain(1)

    policy = BasePolicy(
        action_space=[coopihc.space.Discrete(10)],
        action_set=[[-5 + i for i in range(5)] + [i + 1 for i in range(5)]],
        action_values=None,
    )

    user = CarefulPointer(agent_policy=policy)
    bundle = PlayUser(task, user, unitcdgain)

    observation_dict = OrderedDict(
        {
            "task_state": OrderedDict({"position": 0}),
            "user_state": OrderedDict({"goal": 0}),
        }
    )
    training_env = ThisActionWrapper(
        Train(
            bundle, observation_mode="multidiscrete", observation_dict=observation_dict
        )
    )

    # ================ Training environment finished creating

    task = SimplePointingTask(gridsize=31, number_of_targets=8)
    unitcdgain = ConstantCDGain(1)

    # # specifying user policy
    # observation_dict = OrderedDict({'task_state': OrderedDict({'position': 0}), 'user_state': OrderedDict({'goal': 0})})

    action_wrappers = OrderedDict()
    action_wrappers["ThisActionWrapper"] = (ThisActionWrapper, ())
    # Define an RL policy, by giving the model path, the library and algorithm used to train it, as well as potential wrappers used during training.
    trained_policy = RLPolicy(
        "user",
        model_path="saved_model",
        learning_algorithm="PPO",
        library="stable_baselines3",
        training_env=training_env,
        wrappers={"actionwrappers": action_wrappers, "observation_wrappers": {}},
    )
    # Override the old policy with the new policy
    user = CarefulPointer(agent_policy=trained_policy)
    # Evaluate the trained policy
    bundle = PlayNone(task, user, unitcdgain)
    game_state = bundle.reset()
    bundle.render("plotext")
    plt.tight_layout()
    while True:
        observation, sum_reward, is_done, rewards = bundle.step()
        bundle.render("plotext")
        if is_done:
            break
