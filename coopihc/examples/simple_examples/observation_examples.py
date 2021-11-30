import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))

import numpy
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.observation.ExampleObservationEngine import ExampleObservationEngine

x = StateElement(
    values=None,
    spaces=Space(
        [
            numpy.array([-1.0]).reshape(1, 1),
            numpy.array([1.0]).reshape(1, 1),
        ]
    ),
)

# Discrete substate. Provide Space([range]). Value is optional
y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], type=numpy.int)]))

# Define a State, composed of two substates previously defined
s1 = State(substate_x=x, substate_y=y)

# Define a super-State that is composed of the State previously defined
S = State()

a = StateElement(values=None, spaces=Space([numpy.array([0, 1, 2], dtype=numpy.int8)]))
s2 = State(substate_a=a)
S["substate1"] = s1
S["substate_2"] = s2
obs_engine = ExampleObservationEngine("substate1")
print("Game state before observation")
print(S)
print("Observation")
print(obs_engine.observe(S)[0])
