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

user_action_state = State()
user_action_state["action"] = discrete_array_element(low=-1, high=1)

assistant_action_state = State()
assistant_action_state["action"] = cat_element(N=2)


bundle = Bundle(
    task=ExampleTask(),
    user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
    assistant=BaseAgent(
        "assistant",
        override_policy=(BasePolicy, {"action_state": assistant_action_state}),
    ),
    seed=222,
)


def test_task_param():
    class ExampleTaskwithParams(ExampleTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parameters = {"param1": 1, "param2": 2}

        def finit(self):
            self.update_parameters({"finit": 1})

    bundle = Bundle(
        task=ExampleTaskwithParams(),
        user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
        assistant=BaseAgent(
            "assistant",
            override_policy=(BasePolicy, {"action_state": assistant_action_state}),
        ),
    )
    task_dic = {"param1": 1, "param2": 2, "finit": 1}
    assert bundle.parameters == task_dic
    assert bundle.task.parameters == task_dic
    assert bundle.user.parameters == task_dic
    assert bundle.assistant.parameters == task_dic
    assert bundle.user.observation_engine.parameters == task_dic
    assert bundle.user.inference_engine.parameters == task_dic
    assert bundle.user.policy.parameters == task_dic


def test_user_param():
    class BaseAgentWithParams(BaseAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parameters = {"param1": 1, "param2": 2}

        def finit(self):
            self.update_parameters({"finit": 1})

    bundle = Bundle(
        task=ExampleTask(),
        user=BaseAgentWithParams(
            "user", policy_kwargs={"action_state": user_action_state}
        ),
        assistant=BaseAgent(
            "assistant",
            override_policy=(BasePolicy, {"action_state": assistant_action_state}),
        ),
    )
    task_dic = {"param1": 1, "param2": 2, "finit": 1}
    assert bundle.parameters == task_dic
    assert bundle.task.parameters == task_dic
    assert bundle.user.parameters == task_dic
    assert bundle.assistant.parameters == task_dic
    assert bundle.user.observation_engine.parameters == task_dic
    assert bundle.user.inference_engine.parameters == task_dic
    assert bundle.user.policy.parameters == task_dic


def test_assistant_param():
    class BaseAgentWithParams(BaseAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parameters = {"param1": 1, "param2": 2}

        def finit(self):
            self.update_parameters({"finit": 1})

    bundle = Bundle(
        task=ExampleTask(),
        user=BaseAgent("user", policy_kwargs={"action_state": user_action_state}),
        assistant=BaseAgentWithParams(
            "assistant",
            override_policy=(BasePolicy, {"action_state": assistant_action_state}),
        ),
    )
    task_dic = {"param1": 1, "param2": 2, "finit": 1}
    assert bundle.parameters == task_dic
    assert bundle.task.parameters == task_dic
    assert bundle.user.parameters == task_dic
    assert bundle.assistant.parameters == task_dic
    assert bundle.user.observation_engine.parameters == task_dic
    assert bundle.user.inference_engine.parameters == task_dic
    assert bundle.user.policy.parameters == task_dic


class CustomAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = {"attribute_shared": 1}


class SharedCustomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def reset(self, *args, **kwargs):
        assert hasattr(self, "attribute_shared")


class NotSharedCustomPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        assert not hasattr(self, "attribute_not_shared")


def test_parameter_in_policy():

    agent_one = CustomAgent("user", agent_policy=SharedCustomPolicy())
    agent_one.reset_all()


def test_parameter_not_in_policy():
    agent_one = CustomAgent("user", agent_policy=NotSharedCustomPolicy())
    agent_one.reset_all()


def test_policy_in_agent():
    test_parameter_in_policy()
    test_parameter_not_in_policy()


def test_parameter_in_policy_in_bundle():
    agent_one = CustomAgent("user", agent_policy=SharedCustomPolicy())
    agent_two = BaseAgent("assistant")
    task = ExampleTask()
    bundle = Bundle(task=task, user=agent_one, assistant=agent_two)
    bundle.reset()


def test_parameter_not_in_policy_in_bundle():
    agent_one = CustomAgent("user", agent_policy=NotSharedCustomPolicy())
    agent_two = BaseAgent("assistant")
    task = ExampleTask()
    bundle = Bundle(task=task, user=agent_one, assistant=agent_two)
    bundle.reset()


def test_policy_in_bundle():
    test_parameter_in_policy_in_bundle()
    test_parameter_not_in_policy_in_bundle()


if __name__ == "__main__":
    test_task_param()
    test_user_param()
    test_assistant_param()
    test_policy_in_agent()
    test_policy_in_bundle()
