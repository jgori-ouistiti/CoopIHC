import numpy
from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.agents.ExampleUser import ExampleUser
from coopihc.agents.ExampleAssistant import ExampleAssistant


# [start-check-task]
# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = discrete_array_element(low=-1, high=1)

assistant_action_state = State()
assistant_action_state["action"] = discrete_array_element(low=-1, high=1)


# Bundle a task together with two BaseAgents
bundle = Bundle(
    task=ExampleTask(),
    # Here policy = None, will call BasePolicy of BaseAgent with kwargs policy_kwargs
    user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
    # Here, we use the override mechanism directly. Both are equivalent
    assistant=BaseAgent(
        "assistant",
        override_policy=(BasePolicy, {"action_state": assistant_action_state}),
    ),
)


# Reset the task, step through the task
bundle.reset(go_to=1)
state, rewards, is_done = bundle.step(user_action=1, assistant_action=0)


# Test with input sampled from the agent policies
bundle.reset()
while True:
    game_state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
    if is_done:
        break
# [end-check-task]


# [start-check-taskuser]


class ExampleTaskWithoutAssistant(ExampleTask):
    def on_assistant_action(self, *args, **kwargs):
        return self.state, 0, False


example_task = ExampleTaskWithoutAssistant()
example_user = ExampleUser()
bundle = Bundle(task=example_task, user=example_user)
bundle.reset()

while True:
    state, rewards, is_done = bundle.step()
    if is_done:
        break
# [end-check-taskuser]

# [start-highlevel-code]
# Define a task
example_task = ExampleTask()
# Define a user
example_user = ExampleUser()
# Define an assistant
example_assistant = ExampleAssistant()
# Bundle them together
bundle = Bundle(task=example_task, user=example_user, assistant=example_assistant)
# Reset the bundle (i.e. initialize it to a random or prescribed states)
bundle.reset(
    go_to=1
)  # Reset in a state where the user has already produced an observation and made an inference.

# Step through the bundle (i.e. play full rounds)
while 1:
    state, rewards, is_done = bundle.step(user_action=1, assistant_action=None)
    # Do something with the state or the rewards
    if is_done:
        break
# [end-highlevel-code]
