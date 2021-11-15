![CoopIHC Logo](https://raw.githubusercontent.com/jgori-ouistiti/interaction-agents/cj_opensource/docs/guide/images/coopihc-logo.png)

---

_CoopIHC_, pronounced 'kopik', is a Python module that provides a common basis for describing computational Human Computer Interaction (HCI) contexts, mostly targeted at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing and extending other researcher's work
2. It can help design intelligent assistants by translating an interactive context into a problem that can be solved (via other methods).

## Installing

Copy the files into your current project and run (-e for editable, optional):

```Shell

pip install -e .

```

You can them import the necessary packages from `core`.

## Quickstart

At a high level, CoopIHC code will usually consist of defining tasks, agents (users and assistants), bundling them together, and playing several rounds of interaction until the game ends.

### Task

Tasks represent whatever the user is interacting with. They are essentially characterized by:

- An internal state, the **task state** which holds all the task's information.
- A **user step (transition) function**, which describes how the task state changes based on the user action.
- An **assistant step (transition) function**, which describes how the task state changes based on the assistant action.

```Python
from core.interactiontask import InteractionTask
from core.space import StateElement, Space

class ExampleTask(InteractionTask):
    """
    ExampleTask with two agents. The task is finished when 'x' reaches 4.
    The user and assistants can both do -1, +0, +1 to 'x'.
    """

    def __init__(self):
        super().__init__()

        self.state["x"] = StateElement(
            values=0,
            spaces=Space([numpy.arange(-4, 5, dtype=numpy.int16)])
        )

    def reset(self):
        self.state["x"]["values"] = 0

    def user_step(self):
        is_done = False
        self.state["x"] += self.user_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def assistant_step(self):
        is_done = False
        self.state["x"] += self.assistant_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def render(self):
            print(f"Turn number {self.turn_number}")
            print(self.state)
```

### Agents

Defining a new agent is done by subclassing the `BaseAgent` class:

```Python
from core.agents import BaseAgent
from core.space import StateStateElement, Space
from core.policy import BasePolicy

class ExampleUser(BaseAgent):
    """An agent that handles the BasePolicy."""

    def __init__(self):
        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = StateElement(
            values=4,
            spaces=[Space([numpy.arange(-4, 5, dtype=numpy.int16)])],
        )

        # Define a policy (random policy)
        action_state = State()
        action_state["action"] = StateElement(
            values=None,
            spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
        )
        agent_policy = BasePolicy(action_state)

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_state=state,
        )

    # Override default behaviour of BaseAgent which would randomly
    # sample new goal values on each reset.
    # Here for purpose of demonstration we impose a goal = 4.
    def reset(self, dic=None):
        self.state["goal"]["values"] = 4
```

### Bundle

Defining a bundle consists of (at least) a task and a user.
It can optionally include an assistant.

```Python
from core.bundle import Bundle

# Define a task
example_task = ExampleTask()
# Define a user
example_user = ExampleUser()
# Bundle them together
bundle = Bundle(task=example_task, user=example_user)
# Reset the bundle (i.e. initialize it to a random or presecribed states)
bundle.reset(turn=1)
# Step through the bundle (i.e. play a full round)
while 1:
    user_action = bundle.user.policy.sample()[0]
    state, rewards, is_done = bundle.step(user_action)
    print(state, rewards, is_done)
    if is_done:
        break
```

It consists of defining tasks, users, assistants, bundling them together, and playing several rounds of interaction until the game ends.

## Contribution Guidelines

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

To learn more about making a contribution to CoopIHC, please see our [Contribution page](CONTRIBUTING.md).
