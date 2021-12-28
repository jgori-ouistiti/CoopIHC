import coopihc
from coopihc.space.StateElement import StateElement
from coopihc.space.State import State
from coopihc.space.utils import (
    StateNotContainedError,
    autospace,
    discrete_space,
    multidiscrete_space,
    continuous_space,
)
import numpy

s = 0


def test__init__():
    global s
    x = StateElement(1, autospace([1, 2, 3]))
    s = State()
    s["x"] = x
    assert State(x=x) == s
    s["y"] = StateElement(
        0 * numpy.ones((2, 2)), autospace(-numpy.ones((2, 2)), numpy.ones((2, 2)))
    )
    assert "x" in s.keys()
    assert "y" in s.keys()


def test_reset():
    global s
    s.reset()
    assert s["x"] in autospace([1, 2, 3])
    assert s["y"] in autospace(-numpy.ones((2, 2)), numpy.ones((2, 2)))
    reset_dic = {"x": 3, "y": numpy.zeros((2, 2))}
    s.reset(reset_dic)
    assert s["x"] == 3
    assert (s["y"] == numpy.zeros((2, 2))).all()


state = State()
substate = State()
substate["x1"] = StateElement(1, discrete_space([1, 2, 3]))
substate["x2"] = StateElement(
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
substate["x3"] = StateElement(
    1.5 * numpy.ones((3, 3)),
    continuous_space(numpy.ones((3, 3)), 2 * numpy.ones((3, 3))),
)

substate2 = State()
substate2["y1"] = StateElement(1, discrete_space([1, 2, 3]))
substate2["y2"] = StateElement(
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
state["sub1"] = substate
state["sub2"] = substate2

filterdict = dict(
    {
        "sub1": dict({"x1": 0, "x2": slice(0, 2)}),
        "sub2": dict({"y2": 0}),
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
    test__init__()
    test_reset()
    test_filter()
