import numpy

from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.space.State import State
from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy

# [start-check-task]
# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = StateElement(
    0, Space(numpy.array([-1, 0, 1], dtype=numpy.int16), "discrete")
)

assistant_action_state = State()
assistant_action_state["action"] = StateElement(
    0, Space(numpy.array([-1, 0, 1], dtype=numpy.int16), "discrete")
)

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
bundle.reset(turn=1)
bundle.step(numpy.array([1]), numpy.array([1]))

# Test simple input
bundle.step(numpy.array([1]), numpy.array([1]))

# Test with input sampled from the agent policies
bundle.reset()
while True:
    game_state, rewards, is_done = bundle.step(
        bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0]
    )
    # print(game_state["task_state"]["x"].squeeze().tolist())
    if is_done:
        break
# [end-check-task]
