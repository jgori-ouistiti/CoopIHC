# List of kwargs for bundles init()
#    - reset_skip_first_half_step (if True, skips the first_half_step of the bundle on reset. The idea being that the internal state of the agent provided during initialization should not be updated during reset). To generate a consistent observation, what we do is run the observation engine, but without potential noisefactors.
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))

import numpy
from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.space.State import State
from coopihc.space.Space import Space
from coopihc.space.utils import discrete_space
from coopihc.space.StateElement import StateElement
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.agents.ExampleUser import ExampleUser


# [start-check-task]
# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = StateElement(0, discrete_space([-1, 0, 1]))

assistant_action_state = State()
assistant_action_state["action"] = StateElement(0, discrete_space([-1, 0, 1]))

# Bundle a task together with two BaseAgents
bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", override_agent_policy=BasePolicy(user_action_state)),
    assistant=BaseAgent(
        "assistant",
        override_agent_policy=BasePolicy(assistant_action_state),
    ),
)

# Reset the task, plot the state.
bundle.reset(turn=1)
print(bundle.game_state)
bundle.step(numpy.array([1]), numpy.array([1]))
print(bundle.game_state)

# Test simple input
bundle.step(numpy.array([1]), numpy.array([1]))

# Test with input sampled from the agent policies
bundle.reset()
while True:
    task_state, rewards, is_done = bundle.step(
        bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0]
    )
    print(task_state)
    if is_done:
        break
# [end-check-task]

# [start-check-taskuser]
class ExampleTaskWithoutAssistant(ExampleTask):
    def assistant_step(self, *args, **kwargs):
        return self.state, 0, False, {}


example_task = ExampleTaskWithoutAssistant()
example_user = ExampleUser()
bundle = Bundle(task=example_task, user=example_user)
bundle.reset(turn=1)
while 1:
    state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0])
    print(state, rewards, is_done)
    if is_done:
        break
# [end-check-taskuser]

# [start-highlevel-code]
# Define a task
example_task = ExampleTask()
# Define a user
example_user = ExampleUser()
# Define an assistant
example_assistant = BaseAgent("assistant")
# Bundle them together
bundle = Bundle(task=example_task, user=example_user)
# Reset the bundle (i.e. initialize it to a random or presecribed states)
bundle.reset(turn=1)
# Step through the bundle (i.e. play a full round)
while 1:
    state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0])
    print(state, rewards, is_done)
    if is_done:
        break
# [end-highlevel-code]
