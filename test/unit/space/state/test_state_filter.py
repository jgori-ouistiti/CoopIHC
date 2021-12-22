from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.space.utils import discrete_space, multidiscrete_space, continuous_space

import numpy
from collections import OrderedDict

state = State()
substate = State()
substate["x1"] = StateElement(values=1, spaces=discrete_space([1, 2, 3]))
substate["x2"] = StateElement(
    values=[1, 2, 3],
    spaces=multidiscrete_space(
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
substate["x3"] = StateElement(
    values=1.5 * numpy.ones((3, 3)),
    spaces=continuous_space(numpy.ones((3, 3)), 2 * numpy.ones((3, 3))),
)
substate["x4"] = StateElement(
    values=[1, 2, 3],
    spaces=[
        discrete_space([0, 1, 2]),
        discrete_space([1, 2, 3]),
        discrete_space(
            [
                0,
                1,
                2,
                3,
            ]
        ),
    ],
)

substate2 = State()
substate2["y1"] = StateElement(values=1, spaces=discrete_space([1, 2, 3]))
substate2["y2"] = StateElement(
    values=[1, 2, 3],
    spaces=multidiscrete_space(
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
state["sub1"] = substate
state["sub2"] = substate2

filterdict = OrderedDict(
    {
        "sub1": OrderedDict({"x1": 0, "x2": 0, "x4": slice(0, 2)}),
        "sub2": OrderedDict({"y2": 0}),
    }
)


def test_filter():
    # too long to write out the assertions
    f_state = state.filter("spaces", filterdict=filterdict)
    print(f_state)
    f_state = state.filter("values", filterdict=filterdict)
    print(f_state)
    f_state = state.filter("spaces")
    print(f_state)
    f_state = state.filter("values")
    print(f_state)


if __name__ == "__main__":
    test_filter()
