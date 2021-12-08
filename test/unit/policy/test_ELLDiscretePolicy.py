import numpy
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space

def test_init():
    se = StateElement(
        values = [None],
        spaces = Space(
            [numpy.array([1,2,3,4,5,6,7], dtype = numpy.int16)]
        )
    )
    _seed = 123
    action_state = State(**{'action': se})
    policy = ELLDiscretePolicy(action_state, seed = _seed)
    assert policy.action_state is action_state
    assert policy.rng.uniform() == numpy.random.default_rng(_seed).uniform()

    
