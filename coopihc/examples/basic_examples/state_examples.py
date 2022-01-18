import numpy
from coopihc.space.Space import Space
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    autospace,
    discrete_space,
    continuous_space,
    multidiscrete_space,
)

numpy.set_printoptions(precision=3, suppress=True)

# [start-state-example]

state = State()  # Empty super state
substate = State()  # Empty state substate
substate["x1"] = StateElement(
    1, discrete_space([1, 2, 3])
)  # discrete subsubstate substate/x1
substate["x2"] = StateElement(  # multidiscrete subsubstate substate/x2
    [1, 2, 3],
    multidiscrete_space(
        [
            [0, 1, 2],
            [1, 2, 3],
            [
                0,
                1,
                2,
                3,
            ],
        ]
    ),
)
substate["x3"] = StateElement(  # continuous subsubstate substate/x3
    1.5 * numpy.ones((3, 3)),
    continuous_space(numpy.ones((3, 3)), 2 * numpy.ones((3, 3))),
)

substate2 = State()  # Empty substate substate2
substate2["y1"] = StateElement(
    1, discrete_space([1, 2, 3])
)  # discrete subsubstate substate2/y1
substate2["y2"] = StateElement(  # multidiscrete subsubstate substate2/y2
    [1, 2, 3],
    multidiscrete_space(
        [
            [0, 1, 2],
            [1, 2, 3],
            [
                0,
                1,
                2,
                3,
            ],
        ]
    ),
)

state["sub1"] = substate  # assign substates to super state
state["sub2"] = substate2
# >>> print(state)
# ----  --  ---------------  -------------------
# sub1  x1  1                Discr(3)
#       x2  [1 2 3]          MultiDiscr[3, 3, 4]
#       x3  [[1.5 1.5 1.5]   Cont(3, 3)
#            [1.5 1.5 1.5]
#            [1.5 1.5 1.5]]
# sub2  y1  1                Discr(3)
#       y2  [1 2 3]          MultiDiscr[3, 3, 4]
# ----  --  ---------------  -------------------

# [end-state-example]

# [start-state-reset]
reset_dic = {
    "sub1": {"x1": 3, "x2": numpy.array([0, 1, 0]).reshape(3, 1)},
    "sub2": {"y1": 3},
}
state.reset(reset_dic)
# [end-state-reset]

# [start-state-filter]
filterdict = dict(
    {
        "sub1": dict({"x1": 0, "x2": slice(0, 2)}),
        "sub2": dict({"y2": 2}),
    }
)

f_state = state.filter(mode="array", filterdict=filterdict)
f_state = state.filter(mode="stateelement", filterdict=filterdict)
f_state = state.filter(mode="spaces")
f_state = state.filter(mode="array")

# [end-state-filter]

# [start-state-serialize]
state.serialize()
# [end-state-serialize]
