from coopihc.interactiontask import InteractionTask
from coopihc.base import StateElement, Space, State

import numpy

import sys

_str = sys.argv[1]


from examples.tasks import ExampleTask

# ================= Test Task =======================
from coopihc.bundle import Bundle
from coopihc.policy import BasePolicy
from coopihc.agents import BaseAgent

# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state["action"] = StateElement(
    values=None, spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])]
)

assistant_action_state = State()
assistant_action_state["action"] = StateElement(
    values=None, spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])]
)

# Run a task together with two BaseAgents
bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", override_agent_policy=BasePolicy(user_action_state)),
    assistant=BaseAgent(
        "assistant", override_agent_policy=BasePolicy(assistant_action_state)
    ),
)


if _str == "reset":
    bundle.reset(turn=1)
    print(bundle.game_state)
    bundle.reset(turn=2)
    print(bundle.game_state)
    bundle.reset(turn=3)
    print(bundle.game_state)
    bundle.reset(turn=0)
    print(bundle.game_state)


if _str == "bundletype":
    bundle.reset(turn=0)
    bundle.step(numpy.array([1]), numpy.array([1]))
    print(bundle.game_state)
    bundle.reset(turn=0)
    bundle.step(None, numpy.array([1]))
    print(bundle.game_state)
    bundle.reset(turn=0)
    bundle.step(numpy.array([1]), None)
    print(bundle.game_state)
    bundle.reset(turn=0)
    bundle.step(None, None)
    print(bundle.game_state)


if _str == "reset-step":
    for j in range(4):
        for i in range(4):
            bundle.reset(turn=j)
            bundle.step(numpy.array([1]), numpy.array([1]), go_to_turn=i)
            print(j, i)
            print(bundle.game_state)
