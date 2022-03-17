import numpy
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.policy.BasePolicy import BasePolicy

from coopihc.base.State import State
from coopihc.base.StateElement import StateElement
from coopihc.base.utils import space
from coopihc.base.utils import autospace


def test_init():
    action_state = State(
        **{
            "action1": StateElement(
                0, Space(numpy.array([0, 1], dtype=numpy.int16), "discrete")
            ),
            "action2": StateElement(
                3, Space(numpy.array([3, 4, 5], dtype=numpy.int16), "discrete")
            ),
        }
    )
    policy = BasePolicy(action_state=action_state)
    assert policy.action == (0, 3)


action_state = State(
    **{
        "action1": StateElement(
            0, Space(numpy.array([0, 1], dtype=numpy.int16), "discrete")
        ),
        "action2": StateElement(
            3, Space(numpy.array([3, 4, 5], dtype=numpy.int16), "discrete")
        ),
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
