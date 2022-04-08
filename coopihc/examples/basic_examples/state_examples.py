import numpy
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element


numpy.set_printoptions(precision=3, suppress=True)

# [start-state-example]

state = State()

substate = State()
substate["x1"] = discrete_array_element(init=1, low=1, high=3)
substate["x3"] = array_element(
    init=1.5 * numpy.ones((2, 2)), low=numpy.ones((2, 2)), high=2 * numpy.ones((2, 2))
)

substate2 = State()
substate2["y1"] = discrete_array_element(init=1, low=1, high=3)

state["sub1"] = substate
state["sub2"] = substate2

# [end-state-example]

# [start-state-reset]
reset_dic = {
    "sub1": {"x1": 3},
    "sub2": {"y1": 3},
}
state.reset(dic=reset_dic)
assert state["sub1"]["x1"] == 3
assert state["sub2"]["y1"] == 3
# [end-state-reset]

# [start-state-filter]
filterdict = dict(
    {
        "sub1": dict({"x1": ..., "x3": slice(0, 1)}),
        "sub2": dict({"y1": ...}),
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
