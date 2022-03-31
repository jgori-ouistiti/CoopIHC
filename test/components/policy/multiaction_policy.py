import numpy
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.policy.BasePolicy import BasePolicy

from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, cat_element
from coopihc.base.StateElement import StateElement


def test_init():
    action_state = State(
        **{
            "action1": cat_element(2),
            "action2": discrete_array_element(init=3, low=3, high=5),
        }
    )
    policy = BasePolicy(action_state=action_state)
    assert policy.action == (0, 3)


action_state = State(
    **{
        "action1": cat_element(2),
        "action2": discrete_array_element(init=3, low=3, high=5),
    }
)
policy = BasePolicy(action_state=action_state)


def test_reset():
    global policy
    policy.reset(random=True)


def test_set():
    global policy
    policy.action = (1, 5)
    assert policy.action == (1, 5)
    assert isinstance(policy.action_state["action1"], StateElement)


if __name__ == "__main__":
    test_init()
    test_reset()
    test_set()
