import numpy

from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy

# [start-check-task]
# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = discrete_array_element(low=-1, high=1)

assistant_action_state = State()
assistant_action_state["action"] = discrete_array_element(low=-1, high=1)


# Bundle a task together with two BaseAgents
bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", agent_policy=BasePolicy(user_action_state)),
    assistant=BaseAgent(
        "assistant",
        agent_policy=BasePolicy(assistant_action_state),
    ),
)

# Reset the task, plot the state.
bundle.reset(go_to=1)
bundle.step(1, 1)

# Test simple input
bundle.step(1, 1)

# Test with input sampled from the agent policies
bundle.reset()
while True:
    game_state, rewards, is_done = bundle.step(
        bundle.user.take_action()[0], bundle.assistant.take_action()[0]
    )
    if is_done:
        break
# [end-check-task]
