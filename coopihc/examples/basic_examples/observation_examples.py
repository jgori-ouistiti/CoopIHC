import numpy
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.space.utils import autospace
from coopihc.observation.ExampleObservationEngine import ExampleObservationEngine

import numpy


x = StateElement(
    numpy.array([0]).reshape(1, 1),
    autospace(
        numpy.array([-1.0]).reshape(1, 1),
        numpy.array([1.0]).reshape(1, 1),
    ),
)
y = StateElement(2, autospace([1, 2, 3]))
s1 = State(substate_x=x, substate_y=y)
a = StateElement(0, autospace([0, 1, 2]))
s2 = State(substate_a=a)
S = State()
S["substate1"] = s1
S["substate_2"] = s2

# [start-obseng-example]
obs_engine = ExampleObservationEngine("substate1")
# Game state before observation
# >>> print(S)
# ----------  ----------  -  ----------
# substate1   substate_x  0  Cont(1, 1)
#             substate_y  2  Discr(3)
# substate_2  substate_a  0  Discr(3)
# ----------  ----------  -  ----------

# Produced Observation
# >>> print(obs_engine.observe(S)[0])
# ---------  ----------  -  ----------
# substate1  substate_x  0  Cont(1, 1)
#            substate_y  2  Discr(3)
# ---------  ----------  -  ----------
# [end-obseng-example]
