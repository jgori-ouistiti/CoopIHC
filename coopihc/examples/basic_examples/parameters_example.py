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
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
import copy

# [start-parameters-example]
class ExampleTaskwithParams(ExampleTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = {"param1": 1, "param2": 2}

    def finit(self):
        self.update_parameters({"finit": 1})


bundle = Bundle(
    task=ExampleTaskwithParams(),
    user=BaseAgent("user"),
    assistant=BaseAgent("assistant"),
)

task_dic = {"param1": 1, "param2": 2, "finit": 1}
# Parameters defined in the task available everywhere
assert bundle.parameters == task_dic
assert bundle.task.parameters == task_dic
assert bundle.user.parameters == task_dic
assert bundle.assistant.parameters == task_dic
assert bundle.user.observation_engine.parameters == task_dic
assert bundle.user.inference_engine.parameters == task_dic
assert bundle.user.policy.parameters == task_dic
# Parameters accessible as attributes everywhere
assert bundle.user.param1 == 1
assert bundle.user.policy.param1 == 1
# [end-parameters-example]
