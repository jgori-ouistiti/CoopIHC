import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))

from coopihc.interactiontask.ExampleTask import ExampleTask


# [start-check-task]
# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = StateElement(
    values=None,
    spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
)

assistant_action_state = State()
assistant_action_state["action"] = StateElement(
    values=None,
    spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
)

# Bundle a task together with two BaseAgents
bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", agent_policy=BasePolicyuser_action_state)),
    assistant=BaseAgent(
        "assistant",
        agent_policy=BasePolicy(assistant_action_state),
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
        bundle.user.policy.sample()[0], bundle.assistant.policy.sample()0]
    )
    print(task_state)
    if is_done:
        break
# [end-check-task]