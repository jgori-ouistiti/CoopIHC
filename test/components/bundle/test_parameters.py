import numpy
from coopihc.interactiontask.ExampleTask import ExampleTask
from coopihc.base.State import State
from coopihc.base.elements import (
    discrete_array_element,
    array_element,
    cat_element,
    integer_space,
    integer_set,
)
from coopihc.bundle.Bundle import Bundle
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.agents.ExampleUser import ExampleUser
from coopihc.agents.ExampleAssistant import ExampleAssistant
import copy

user_action_state = State()
user_action_state["action"] = discrete_array_element(low=-1, high=1)

assistant_action_state = State()
assistant_action_state["action"] = cat_element(N=2)



class ExampleTaskwithParams(ExampleTask):
    def __init__(self, *args, **kwargs):
        

bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
    assistant=BaseAgent(
        "assistant",
        override_policy=(BasePolicy, {"action_state": assistant_action_state}),
    ),
    seed=222,
)
