from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
    NumpyFunctionNotHandledWarning,
    discrete_space,
    continuous_space,
    multidiscrete_space,
    autospace,
)

import numpy
import pytest
import json
import copy
from tabulate import tabulate

cont_space = None
discr_space = None
multidiscr_space = None


def test_array_init_discrete():
    global discr_space
    discr_space = discrete_space([1, 2, 3])

    x = StateElement([2], discr_space, out_of_bounds_mode="error")
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (1,)
    assert x == numpy.array([2], dtype=numpy.int16)


def test_array_init_continuous():
    global cont_space
    cont_space = continuous_space(-numpy.ones((2, 2)), numpy.ones((2, 2)))
    x = StateElement(
        numpy.array([[0, 0], [0, 0]]), cont_space, out_of_bounds_mode="error"
    )
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (2, 2)
    assert (x == numpy.zeros((2, 2), dtype=numpy.float32)).all()


def test_array_init_cont_extra():
    s = StateElement(
        numpy.array([1 / 8 for i in range(8)]),
        autospace(
            numpy.zeros((1, 8)),
            numpy.ones((1, 8)),
        ),
        out_of_bounds_mode="error",
    )


def test_array_init_multidiscrete():
    global multidiscr_space
    multidiscr_space = multidiscrete_space(
        [
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1, 3, 5, 8]),
        ]
    )
    x = StateElement([1, 5, 3], multidiscr_space, out_of_bounds_mode="error")
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (3, 1)
    assert (x == numpy.array([[1], [5], [3]], dtype=numpy.int16)).all()


def test_array_short():
    global multidiscr_space
    multidiscr_space = multidiscrete_space(
        [
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1, 3, 5, 8]),
        ]
    )
    with pytest.raises(StateNotContainedError):
        x = StateElement([1, 5], multidiscr_space, out_of_bounds_mode="error")


def test_array_init():
    test_array_init_discrete()
    test_array_init_continuous()
    test_array_init_multidiscrete()
    test_array_short()
    test_array_init_cont_extra()


def test_array_init_error_discrete():
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(4, discr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(-3, discr_space, out_of_bounds_mode="error")


def test_array_init_error_continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(2 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(
            -2 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error"
        )


def test_array_init_error_multidiscrete():
    global multidiscr_space

    x = StateElement([1, 2, 8], multidiscr_space, out_of_bounds_mode="error")

    with pytest.raises(StateNotContainedError):
        x = StateElement([0, 2, 8], multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement([4, 2, 8], multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement([1, -1, 8], multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement([1, 6, 8], multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement([1, 1, 2], multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement([1, 1, 7], multidiscr_space, out_of_bounds_mode="error")


def test_array_init_error():
    test_array_init_error_discrete()
    test_array_init_error_continuous()
    test_array_init_error_multidiscrete()


def test_array_init_warning_discrete():
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(4, discr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(-3, discr_space, out_of_bounds_mode="warning")


def test_array_init_warning_continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            2 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            -2 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="warning"
        )


def test_array_init_warning_multidiscrete():
    global multidiscr_space

    x = StateElement([1, 2, 8], multidiscr_space, out_of_bounds_mode="warning")

    with pytest.warns(StateNotContainedWarning):
        x = StateElement([0, 2, 8], multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement([4, 2, 8], multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement([1, -1, 8], multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement([1, 6, 8], multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement([1, 1, 2], multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement([1, 1, 7], multidiscr_space, out_of_bounds_mode="warning")


def test_array_init_warning():
    test_array_init_warning_discrete()
    test_array_init_warning_continuous()
    test_array_init_warning_multidiscrete()


def test_array_init_clip_discrete():
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array([2])
    x = StateElement(4, discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array([3])
    x = StateElement(-3, discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array([1])


def test_array_init_clip_continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="clip")
    assert (x == numpy.zeros((2, 2))).all()
    x = StateElement(2 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="clip")
    assert (x == numpy.ones((2, 2))).all()
    x = StateElement(-2.0 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="clip")
    assert (x == -1.0 * numpy.ones((2, 2))).all()


def test_array_init_clip_multidiscrete():
    global multidiscr_space
    x = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [1], [8]])).all()

    x = StateElement([0, 1, 8], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [1], [8]])).all()
    x = StateElement([4, 1, 8], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[3], [1], [8]])).all()

    x = StateElement([1, -1, 8], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [1], [8]])).all()
    x = StateElement([1, 9, 8], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [5], [8]])).all()

    x = StateElement([1, 1, -2], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [1], [1]])).all()
    x = StateElement([1, 1, 10], multidiscr_space, out_of_bounds_mode="clip")
    assert (x == numpy.array([[1], [1], [8]])).all()


def test_array_init_clip():
    test_array_init_clip_discrete()
    test_array_init_clip_continuous()
    test_array_init_clip_multidiscrete()


def test_array_init_dtype_discrete():
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="warning")
    assert x.dtype == numpy.int16
    new_discr_space = autospace([1, 2, 3], dtype=numpy.int64)
    x = StateElement(2, new_discr_space, out_of_bounds_mode="warning")
    assert x.dtype == numpy.int64


def test_array_init_dtype_continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="warning")
    assert x.dtype == numpy.float32
    new_cont_space = autospace([[1]], [[2]], dtype=numpy.int8)
    x = StateElement(
        1.0 * numpy.ones((1, 1)), new_cont_space, out_of_bounds_mode="warning"
    )
    assert x.dtype == numpy.int8


def test_array_init_dtype_multidiscrete():
    global multidiscr_space
    x = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="warning")
    assert x.dtype == numpy.int16
    new_multidiscr_space = autospace(
        [[1, 2, 3], [1, 2, 3, 4, 5], [1, 8]], dtype=numpy.int64
    )
    x = StateElement([1, 1, 8], new_multidiscr_space, out_of_bounds_mode="warning")
    assert x.dtype == numpy.int64


def test_array_init_dtype():
    test_array_init_dtype_discrete()
    test_array_init_dtype_continuous()
    test_array_init_dtype_multidiscrete()


def test__array_ufunc__discrete():
    # Simple arithmetic
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="error")
    assert x + numpy.array(1) == 3
    assert x + 1 == 3
    assert x - 1 == 1
    assert 3 - x == 1
    assert x - numpy.array(1) == 1
    assert numpy.array(3) - x == 1
    assert 1 + x == 3
    x += 1
    y = x - 1
    assert y.out_of_bounds_mode == "error"

    with pytest.raises(StateNotContainedError):
        1 - x
    with pytest.raises(StateNotContainedError):
        x + 2
    with pytest.raises(StateNotContainedError):
        x += 5


def test__array_ufunc__continuous():
    # some matrix operations
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
    assert (x + numpy.ones((2, 2)) == numpy.ones((2, 2))).all()
    assert (x + 1 == numpy.ones((2, 2))).all()
    assert (1 + x == numpy.ones((2, 2))).all()
    assert (x - 1 == -numpy.ones((2, 2))).all()
    assert (1 - x == numpy.ones((2, 2))).all()
    assert ((1 + x) * 0.5 == 0.5 * numpy.ones((2, 2))).all()
    assert (0.5 * (1 + x) @ numpy.ones((2, 2)) == numpy.ones((2, 2))).all()


def test__array_ufunc__multidiscrete():
    global multidiscr_space
    x = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="error")
    assert (x + numpy.array([[1], [1], [-3]]) == numpy.array([[2], [2], [5]])).all()
    with pytest.raises(StateNotContainedError):
        x + numpy.array([[1], [1], [1]])


def test__array_ufunc__comparisons():
    global discr_space
    x = StateElement(2, discr_space, out_of_bounds_mode="error")
    assert x > 1 == True
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
    assert (x < 0).all() == False
    global multidiscr_space
    x = StateElement(
        numpy.array([[1], [1], [1]]), multidiscr_space, out_of_bounds_mode="error"
    )
    assert (x >= numpy.array([[1], [0], [1]])).all() == True
    assert (x >= numpy.array([[1], [5], [1]])).all() == False
    comp = x >= numpy.array([[1], [5], [1]])
    assert (comp == numpy.array([[True], [False], [True]])).all()


def test__array_ufunc__trigonometry():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
    assert (numpy.cos(x) == numpy.ones((2, 2))).all()


def test__array_ufunc__floating():
    global cont_space
    x = StateElement(
        numpy.array([[0.2, 0.3], [1, 0.95]]), cont_space, out_of_bounds_mode="error"
    )
    assert numpy.isfinite(x).all() == True


def test__array_ufunc__out_of_bounds_mode():
    x = StateElement(
        numpy.array([[0.2, 0.3], [1, 0.95]]), cont_space, out_of_bounds_mode="error"
    )
    y = StateElement(
        numpy.array([[-0.2, -0.3], [-1, -0.95]]),
        cont_space,
        out_of_bounds_mode="warning",
    )
    z = StateElement(
        numpy.array([[0.0, 0.0], [0.0, 0.0]]),
        cont_space,
        out_of_bounds_mode="silent",
    )
    u = x + y
    assert u.out_of_bounds_mode == "error"
    u = y + x
    assert u.out_of_bounds_mode == "error"
    u = z + x
    assert u.out_of_bounds_mode == "error"
    u = y + z
    assert u.out_of_bounds_mode == "warning"
    u = z + 0
    assert u.out_of_bounds_mode == "silent"


def test__array_ufunc__():
    test__array_ufunc__discrete()
    test__array_ufunc__continuous()
    test__array_ufunc__multidiscrete()
    test__array_ufunc__comparisons()
    test__array_ufunc__trigonometry()
    test__array_ufunc__floating()
    test__array_ufunc__out_of_bounds_mode()


def test_amax_nothandled():
    StateElement.HANDLED_FUNCTIONS = {}
    cont_space = autospace(
        [[-1, -1], [-1, -1]], [[1, 1], [1, 1]], dtype=numpy.float64
    )  # Here the
    x = StateElement(
        numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
        cont_space,
        out_of_bounds_mode="warning",
    )

    # Without handled function
    with pytest.warns(NumpyFunctionNotHandledWarning):
        y = numpy.max(x)
    assert isinstance(y, numpy.ndarray)
    assert not isinstance(y, StateElement)
    assert y == 0.8
    assert not hasattr(y, "spaces")
    assert not hasattr(y, "out_of_bounds_mode")


def test_amax_implements_decorator():
    cont_space = autospace([[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64)
    x = StateElement(
        numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
        cont_space,
        out_of_bounds_mode="warning",
    )

    @StateElement.implements(numpy.amax)
    def amax(arr, **keywordargs):
        spaces, out_of_bounds_mode, kwargs = (
            arr.spaces,
            arr.out_of_bounds_mode,
            arr.kwargs,
        )
        obj = arr.view(numpy.ndarray)
        argmax = numpy.argmax(obj, **keywordargs)
        index = numpy.unravel_index(argmax, arr.spaces.shape)
        obj = numpy.amax(obj, **keywordargs)
        obj = numpy.asarray(obj).view(StateElement)
        if arr.spaces.space_type == "continuous":
            obj.spaces = autospace(
                numpy.atleast_2d(arr.spaces.low[index[0], index[1]]),
                numpy.atleast_2d(arr.spaces.high[index[0], index[1]]),
            )
        else:
            raise NotImplementedError
        obj.out_of_bounds_mode = arr.out_of_bounds_mode
        obj.kwargs = arr.kwargs
        return obj

    y = numpy.amax(x)
    assert isinstance(y, StateElement)
    assert StateElement.HANDLED_FUNCTIONS.get(numpy.amax) is not None
    assert x.HANDLED_FUNCTIONS.get(numpy.amax) is not None
    assert y.shape == ()
    assert y == 0.8
    assert y.spaces.space_type == "continuous"
    assert y.spaces.shape == (1, 1)
    assert y.spaces.low == numpy.array([[-2]])
    assert y.spaces.high == numpy.array([[3]])


def test_array_function_simple():
    test_amax_nothandled()
    test_amax_implements_decorator()


def test__array_function__():
    test_array_function_simple()


def test_equals_discrete():
    global discr_space
    discr_space = discrete_space([1, 2, 3])
    new_discr_space = discrete_space([1, 2, 3, 4])
    x = StateElement(numpy.array(1), discr_space)
    y = StateElement(numpy.array(1), discr_space)
    assert x.equals(y)
    assert x.equals(y, mode="hard")
    z = StateElement(numpy.array(2), discr_space)
    assert not x.equals(z)
    w = StateElement(numpy.array(1), new_discr_space)
    assert w.equals(x)
    assert not w.equals(x, "hard")


def test_equals_continuous():
    cont_space = autospace([[-1, -1], [-1, -1]], [[1, 1], [1, 1]], dtype=numpy.float64)
    new_cont_space = autospace(
        [[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64
    )

    x = StateElement(numpy.zeros((2, 2)), cont_space)
    y = StateElement(numpy.zeros((2, 2)), cont_space)
    assert (x.equals(y)).all()
    assert (x.equals(y, mode="hard")).all()
    z = StateElement(numpy.eye(2), cont_space)
    assert not (x.equals(z)).all()
    w = StateElement(numpy.zeros((2, 2)), new_cont_space)
    assert (w.equals(x)).all()
    assert not (w.equals(x, "hard")).all()


def test_equals_multidiscrete():
    global multidiscr_space

    new_multidiscr_space = multidiscrete_space(
        [
            numpy.array(
                [
                    1,
                    2,
                ]
            ),
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1, 3, 5, 8]),
        ]
    )
    x = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="warning")
    y = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="warning")
    z = StateElement([1, 2, 8], multidiscr_space, out_of_bounds_mode="warning")
    w = StateElement([1, 1, 8], new_multidiscr_space, out_of_bounds_mode="warning")
    assert (x.equals(y)).all()
    assert (x.equals(y, mode="hard")).all()
    assert not (x.equals(z)).all()
    assert (w.equals(x)).all()
    assert not (w.equals(x, "hard")).all()


def test_equals():
    test_equals_discrete()
    test_equals_continuous()
    test_equals_multidiscrete()


def test__iter__discrete():
    global discr_space
    discr_space = discrete_space([1, 2, 3])

    x = StateElement([2], discr_space, out_of_bounds_mode="error")
    for _x in x:
        assert isinstance(_x, StateElement)
        assert _x == 2
        assert _x == x
        assert _x.spaces == x.spaces
        assert _x.out_of_bounds_mode == x.out_of_bounds_mode


def test__iter__continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space)
    for _x in x:
        for __x in _x:
            assert isinstance(__x, StateElement)
            assert __x.equals(
                StateElement(numpy.array([[0]]), autospace([[-1]], [[1]])), mode="hard"
            )
            assert not __x.equals(
                StateElement(numpy.array([[0]]), autospace([[-3]], [[3]])), mode="hard"
            )
            assert __x.equals(
                StateElement(numpy.array([[0]]), autospace([[-3]], [[3]])), mode="soft"
            )


def test__iter__multidiscrete():
    global multidiscr_space
    multidiscr_space = multidiscrete_space(
        [
            numpy.array([1, 2]),
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1, 3, 5, 8]),
        ]
    )
    x = StateElement(numpy.array([[1], [1], [3]]), multidiscr_space)
    for n, _x in enumerate(x):
        assert isinstance(_x, StateElement)
        if n == 0:
            assert _x.equals(StateElement(numpy.array([1]), autospace([1, 2])))
        elif n == 1:
            assert _x.equals(StateElement(numpy.array([1]), autospace([1, 2, 3, 4, 5])))
        elif n == 2:
            assert _x.equals(StateElement(numpy.array([3]), autospace([1, 3, 5, 8])))


def test__iter__():
    test__iter__discrete()
    test__iter__continuous()
    test__iter__multidiscrete()


def test__repr__discrete():
    global discr_space
    x = StateElement(numpy.array([2]), discr_space)
    # print(x)


def test__repr__continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space)
    # print(x)


def test__repr__multidiscrete():
    global multidiscr_space
    x = StateElement(numpy.array([[1], [1], [3]]), multidiscr_space)
    # print(x)


def test__repr__():
    test__repr__discrete()
    test__repr__continuous()
    test__repr__multidiscrete()


def test_serialize_discrete():
    global discr_space
    x = StateElement(numpy.array([2]), discr_space)
    assert x.serialize() == {
        "values": [2],
        "spaces": {
            "array_list": [1, 2, 3],
            "space_type": "discrete",
            "seed": None,
            "contains": "soft",
        },
    }
    assert isinstance(json.dumps(x.serialize()), str)


def test_serialize_continuous():
    global cont_space
    x = StateElement(numpy.zeros((2, 2)), cont_space)
    assert x.serialize() == {
        "values": [[0.0, 0.0], [0.0, 0.0]],
        "spaces": {
            "array_list": [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            "space_type": "continuous",
            "seed": None,
            "contains": "soft",
        },
    }
    assert isinstance(json.dumps(x.serialize()), str)


def test_serialize_multidiscrete():
    global multidiscr_space
    x = StateElement(numpy.array([[1], [1], [8]]), multidiscr_space)
    assert x.serialize() == {
        "values": [[1], [1], [8]],
        "spaces": {
            "array_list": [[1, 2], [1, 2, 3, 4, 5], [1, 3, 5, 8]],
            "space_type": "multidiscrete",
            "seed": None,
            "contains": "soft",
        },
    }
    assert isinstance(json.dumps(x.serialize()), str)


def test_serialize():
    test_serialize_discrete()
    test_serialize_continuous()
    test_serialize_multidiscrete()


def test_reset_discrete():
    global discr_space
    x = StateElement(numpy.array([2]), discr_space, out_of_bounds_mode="error")
    xset = {}
    for i in range(1000):
        x.reset()
        _x = x.squeeze().tolist()
        xset.update({str(_x): _x})
    assert sorted(xset.values()) == [1, 2, 3]
    # forced reset:
    x.reset(value=2)
    assert x == StateElement(numpy.array([2]), discr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x.reset(value=5)
    x.out_of_bounds_mode = "clip"
    x.reset(value=5)
    assert x == StateElement(numpy.array([3]), discr_space, out_of_bounds_mode="clip")


def test_reset_continuous():
    global cont_space
    x = StateElement(numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error")
    for i in range(1000):
        x.reset()
    x.reset(0.59 * numpy.ones((2, 2)))
    assert (
        x
        == StateElement(
            0.59 * numpy.ones((2, 2)), cont_space, out_of_bounds_mode="error"
        )
    ).all()


def test_reset_multidiscrete():
    global multidiscr_space
    x = StateElement(
        numpy.array([[1], [1], [8]]), multidiscr_space, out_of_bounds_mode="error"
    )
    for i in range(1000):
        x.reset()

    x.reset([1, 1, 8])
    assert (
        x
        == StateElement(
            numpy.array([[1], [1], [8]]), multidiscr_space, out_of_bounds_mode="error"
        )
    ).all()


def test_reset():
    test_reset_discrete()
    test_reset_continuous()
    test_reset_multidiscrete()


def test__setitem__():
    global discr_space
    x = StateElement(1, discr_space)
    with pytest.warns(StateNotContainedWarning):
        x[:] = 4


def test__getitem__discrete():
    global discr_space
    x = StateElement(1, discr_space)
    assert x[0, {"spaces": True}] == x
    assert x[0, {"spaces": True}] is not x
    assert x[0] == x


def test__getitem__continuous():
    global cont_space
    x = StateElement(numpy.array([[0.0, 0.1], [0.2, 0.3]]), cont_space)
    assert x[0, 0] == 0.0
    assert x[0, 0, {"spaces": True}] == StateElement(
        numpy.array([[0.0]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
    )
    assert x[0, 1, {"spaces": True}] == StateElement(
        numpy.array([[0.1]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
    )
    assert x[1, 0, {"spaces": True}] == StateElement(
        numpy.array([[0.2]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
    )
    assert x[1, 1, {"spaces": True}] == StateElement(
        numpy.array([[0.3]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
    )
    assert (x[:, 1] == numpy.array([0.1, 0.3], dtype=numpy.float32)).all()
    assert (
        x[:, 1, {"spaces": True}]
        == StateElement(
            numpy.array([0.1, 0.3], dtype=numpy.float32),
            autospace(numpy.array([[-1], [-1]]), numpy.array([[1], [1]])),
        )
    ).all()

    # case below solved by numpy_input_array = numpy.atleast_2d(numpy_input_array)
    y = StateElement(
        numpy.array([[0.0, 0.1, 0.2, 0.3]]), autospace([[0, 0, 0, 0]], [[1, 1, 1, 1]])
    )
    y[:] = numpy.array([0.8, 0.8, 0.8, 0.8])


def test__getitem__multidiscrete():
    global multidiscr_space
    x = StateElement(
        numpy.array([[1], [1], [8]]), multidiscr_space, out_of_bounds_mode="error"
    )
    assert x[0] == StateElement(
        numpy.array([1]), autospace([1, 2]), out_of_bounds_mode="error"
    )
    assert x[1] == StateElement(
        numpy.array([1]), autospace([1, 2, 3, 4, 5]), out_of_bounds_mode="error"
    )
    assert x[2] == StateElement(
        numpy.array([8]), autospace([1, 3, 5, 8]), out_of_bounds_mode="error"
    )


def test__getitem__():
    test__getitem__discrete()
    test__getitem__continuous()
    test__getitem__multidiscrete()


def test_tabulate_discrete():
    global discr_space
    x = StateElement(1, discr_space)
    print(x._tabulate())
    print(tabulate(x._tabulate()[0]))


def test_tabulate_continuous():
    cont_space = autospace(-numpy.ones((3, 3)), numpy.ones((3, 3)))
    x = StateElement(numpy.zeros((3, 3)), cont_space)
    print(x._tabulate())
    print(tabulate(x._tabulate()[0]))


def test_tabulate_multidiscrete():
    global multidiscr_space
    x = StateElement(
        numpy.array([[1], [1], [8]]), multidiscr_space, out_of_bounds_mode="error"
    )
    print(x._tabulate())
    print(tabulate(x._tabulate()[0]))


def test_tabulate():
    test_tabulate_discrete()
    test_tabulate_continuous()
    test_tabulate_multidiscrete()


if __name__ == "__main__":

    test_array_init()
    test_array_init_error()
    test_array_init_warning()
    test_array_init_clip()
    test_array_init_dtype()
    test__array_ufunc__()
    test__array_function__()
    test_equals()
    test__iter__()
    test__repr__()
    test_serialize()
    test_reset()
    test__setitem__()
    test__getitem__()
    test_tabulate()
