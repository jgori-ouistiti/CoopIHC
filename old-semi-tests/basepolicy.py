from coopihc.space import Space, State, StateElement
from coopihc.policy import BasePolicy
import numpy

# Discrete

action_state = State(
    **{
        "action": StateElement(
            values=None, spaces=Space([numpy.array([1, 2, 3, 4], dtype=numpy.int16)])
        )
    }
)

policy = BasePolicy(action_state)
print(policy.sample())

# MultiDiscrete

action_state = State(
    **{
        "action": StateElement(
            values=None,
            spaces=Space(
                [
                    numpy.array([1, 2, 3, 4], dtype=numpy.int16),
                    numpy.array([-5, -4], dtype=numpy.int16),
                ]
            ),
        )
    }
)

policy = BasePolicy(action_state)
print(policy.sample())

# Continuous

action_state = State(
    **{
        "action": StateElement(
            values=None,
            spaces=Space(
                [
                    -numpy.ones((2, 2), dtype=numpy.float32),
                    numpy.ones((2, 2), dtype=numpy.float32),
                ]
            ),
        )
    }
)

policy = BasePolicy(action_state)
print(policy.sample())
