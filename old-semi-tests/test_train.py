from coopihc.base import StateElement, State
import gym
import numpy
import sys

_str = sys.argv[1]

# -------- Correct assigment
if _str == "convert" or _str == "all":
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
    dict_env = Train(bundle, observation_mode="dict", observation_dict=observation_dict)
    tuple_env = Train(
        bundle, observation_mode="tuple", observation_dict=observation_dict
    )
    md_env = Train(
        bundle, observation_mode="multidiscrete", observation_dict=observation_dict
    )
    nat_env = Train(bundle)
