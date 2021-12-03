from pointing.envs import DiscretePointingTaskPipeWrapper, SimplePointingTask
from pointing.users import CarefulPointer
from pointing.assistants import ConstantCDGain

import coopihc
from coopihc.policy import ELLDiscretePolicy
from coopihc.bundle import PlayNone
from coopihc.wsbundle import Server


# Start task
task = SimplePointingTask(gridsize=20, number_of_targets=5)
# Define some policy for user action
policy = ELLDiscretePolicy(
    action_space=[coopihc.space.Discrete(3)], action_set=[[-1, 0, 1]]
)
# Actions are in human values, i.e. they are not necessarily in range(0,N)
def compute_likelihood(self, action, observation):
    # convert actions and observations
    action = action["human_values"][0]
    goal = observation["user_state"]["goal"]["values"][0]
    position = observation["task_state"]["position"]["values"][0]

    # Write down all possible cases (5)
    # (1) Goal to the right, positive action
    if goal > position and action > 0:
        return 0.99
    # (2) Goal to the right, negative action
    elif goal > position and action <= 0:
        return 0.005
    # (3) Goal to the left, positive action
    if goal < position and action >= 0:
        return 0.005
    # (4) Goal to the left, negative action
    elif goal < position and action < 0:
        return 0.99
    elif goal == position and action == 0:
        return 1
    elif goal == position and action != 0:
        return 0
    else:
        print(goal, position, action)
        raise RunTimeError(
            "warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition"
        )


# Attach likelihood function to the policy
policy.attach_likelihood_function(compute_likelihood)

# Define a user model defined elsewhere, but plug policy described just above inside to be used instead
user = CarefulPointer(agent_policy=policy)
assistant = ConstantCDGain(1)
bundle = PlayNone(task, user, assistant)
server = Server(bundle, DiscretePointingTaskPipeWrapper, address="localhost", port=4000)
server.start()
# Now run launch coopihc/pointing/task_bundles.html in a browser and hit F5 to reset the task.
