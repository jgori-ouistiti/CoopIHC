import coopihc
from coopihc.base.elements import (
    discrete_array_element,
    array_element,
    cat_element,
    integer_space,
    box_space,
)
from coopihc.base.State import State
from coopihc.base.Space import Space
from coopihc.base.utils import StateNotContainedError
from coopihc.base.elements import example_game_state


import numpy
from tabulate import tabulate

s = 0


def test__init__():
    global s
    x = discrete_array_element(init=1, low=1, high=3)
    s = State()
    s["x"] = x
    assert State(x=x) == s
    s["y"] = array_element(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    assert "x" in s.keys()
    assert "y" in s.keys()


def test_reset_small():
    global s
    s.reset()
    assert s["x"] in integer_space(start=1, stop=3)
    assert s["y"] in box_space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    reset_dic = {"x": 3, "y": numpy.ones((2, 2))}
    s.reset(dic=reset_dic)
    assert s["x"] == 3
    assert (s["y"] == numpy.ones((2, 2))).all()


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

filterdict = dict(
    {
        "sub1": dict({"x1": 0, "x3": slice(0, 1)}),
        "sub2": dict({"y1": 0}),
    }
)


def test_filter():
    global filterdict, state
    f_state = state.filter(mode="space", filterdict=filterdict)
    assert f_state == {
        "sub1": {
            "x1": Space(low=1, high=3),
            "x3": Space(low=numpy.ones((2, 2)), high=2 * numpy.ones((2, 2))),
        },
        "sub2": {"y1": Space(low=1, high=3)},
    }

    f_state = state.filter(mode="array", filterdict=filterdict)
    # print(f_state)
    f_state = state.filter(mode="stateelement", filterdict=filterdict)
    # print(f_state)
    f_state = state.filter(mode="space")
    # print(f_state)
    f_state = state.filter(mode="array")
    # print(f_state)
    f_state = state.filter(mode="array-Gym")
    # print(f_state)


def test_serialize():
    global state
    state.serialize()


def test_reset_full():
    reset_dic = {
        "sub1": {"x1": 3},
        "sub2": {"y1": 3},
    }
    state.reset(dic=reset_dic)
    assert state["sub1"]["x1"] == 3
    assert state["sub2"]["y1"] == 3


def test_tabulate_small():
    x = discrete_array_element(init=1, low=1, high=3)
    s = State()
    s["x"] = x
    s["y"] = array_element(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    print(s._tabulate())
    print(tabulate(s._tabulate()[0]))


def test_tabulate_full():
    global state
    state["sub3"] = cat_element(N=3)
    print(state._tabulate())
    print(tabulate(state._tabulate()[0]))


def test_tabulate():
    test_tabulate_small()
    test_tabulate_full()


def test_equals_soft():
    _example_state = example_game_state()
    obs = {
        "game_info": {"turn_index": numpy.array(0), "round_index": numpy.array(0)},
        "task_state": {"position": numpy.array(2), "targets": numpy.array([0, 1])},
        "user_action": {"action": numpy.array(0)},
        "assistant_action": {"action": numpy.array(2)},
    }
    del _example_state["user_state"]
    del _example_state["assistant_state"]
    assert _example_state == obs
    assert _example_state.equals(obs, mode="soft")


def test_equals_hard():
    _example_state = example_game_state()
    obs = {
        "game_info": {"turn_index": numpy.array(0), "round_index": numpy.array(0)},
        "task_state": {"position": numpy.array(2), "targets": numpy.array([0, 1])},
        "user_action": {"action": numpy.array(0)},
        "assistant_action": {"action": numpy.array(2)},
    }
    del _example_state["user_state"]
    del _example_state["assistant_state"]
    assert not _example_state.equals(obs, mode="hard")


def test_equals():
    test_equals_soft()
    test_equals_hard()


if __name__ == "__main__":
    test__init__()
    test_filter()
    test_serialize()
    test_reset_small()
    test_reset_full()
    test_tabulate()
    test_equals()
