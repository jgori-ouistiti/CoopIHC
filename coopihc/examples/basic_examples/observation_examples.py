import numpy
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.observation.ExampleObservationEngine import ExampleObservationEngine

import numpy


x = array_element(low=-1, high=1)
y = discrete_array_element(low=1, high=3, init=2)
s1 = State(substate_x=x, substate_y=y)
a = cat_element(3)
s2 = State(substate_a=a)
S = State()
S["substate1"] = s1
S["substate_2"] = s2

# [start-obseng-example]
obs_engine = ExampleObservationEngine("substate1")
# Game state before observation
# >>> print(S)
# ----------  ----------  -  ---------
# substate1   substate_x  0  Numeric()
#             substate_y  2  Numeric()
# substate_2  substate_a  0  CatSet(3)
# ----------  ----------  -  ---------

print(obs_engine.observe(game_state=S)[0])
# Produced Observation
# >>> print(obs_engine.observe(S)[0])
# ---------  ----------  -  ----------
# substate1  substate_x  0  Cont(1, 1)
#            substate_y  2  Discr(3)
# ---------  ----------  -  ----------
# [end-obseng-example]
