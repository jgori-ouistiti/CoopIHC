from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement, StateElementNumPy
from coopihc.space.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
    discrete_space,
    multidiscrete_space,
    continuous_space,
)

import numpy
import pytest

_cont_space = None
_discr_space = None
_multidiscr_space = None


def test_array_init():
    global _cont_space, _discr_space, _multidiscr_space
    _cont_space = continuous_space(-numpy.array([1]), numpy.array([1]))
    x = StateElementNumPy([2], spaces=_cont_space, out_of_bounds_mode="silent")
    print(x, type(x))
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (1,)
    assert x == numpy.array([2], dtype=numpy.float32)

    _discr_space = discrete_space([1, 2, 3])
    x = StateElementNumPy(3, spaces=_discr_space, out_of_bounds_mode="silent")
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == ()
    assert x == numpy.array(3, dtype=numpy.int16)

    _multidiscr_space = multidiscrete_space([[1, 2, 3], [0, 1, 2], [0, 1, 5, 8]])
    x = StateElementNumPy(
        [2, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="silent"
    )
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (3,)
    assert (x == numpy.array([2, 0, 8], dtype=numpy.int16)).all()


def test_array_init_error():
    StateElementNumPy([0], spaces=_cont_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy([2], spaces=_cont_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy([-2], spaces=_cont_space, out_of_bounds_mode="error")
    StateElementNumPy(2, spaces=_discr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(4, spaces=_discr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(-3, spaces=_discr_space, out_of_bounds_mode="error")

    StateElementNumPy([1, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [0, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [4, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [1, -1, 8], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [1, 3, 8], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [1, 1, 2], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )
    with pytest.raises(StateNotContainedError):
        x = StateElementNumPy(
            [1, 1, 7], spaces=_multidiscr_space, out_of_bounds_mode="error"
        )


def test_array_init_warning():
    StateElementNumPy([0], spaces=_cont_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy([2], spaces=_cont_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy([-2], spaces=_cont_space, out_of_bounds_mode="warning")
    StateElementNumPy(2, spaces=_discr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(4, spaces=_discr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(-3, spaces=_discr_space, out_of_bounds_mode="warning")

    StateElementNumPy([1, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [0, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [4, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [1, -1, 8], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [1, 3, 8], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [1, 1, 2], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElementNumPy(
            [1, 1, 7], spaces=_multidiscr_space, out_of_bounds_mode="warning"
        )


def test_array_init_clip():
    # Continuous
    x = StateElementNumPy([0], spaces=_cont_space, out_of_bounds_mode="clip")
    assert x == numpy.array([0])
    x = StateElementNumPy([2], spaces=_cont_space, out_of_bounds_mode="clip")
    assert x == numpy.array([1])
    x = StateElementNumPy([-2], spaces=_cont_space, out_of_bounds_mode="clip")
    assert x == numpy.array([-1])
    # Discrete
    x = StateElementNumPy(2, spaces=_discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array(2)
    x = StateElementNumPy(4, spaces=_discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array(3)
    x = StateElementNumPy(-3, spaces=_discr_space, out_of_bounds_mode="clip")
    assert x == numpy.array([1])
    # Multidiscrete

    x = StateElementNumPy(
        [1, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 0, 8])).all()
    x = StateElementNumPy(
        [0, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 0, 8])).all()
    x = StateElementNumPy(
        [4, 0, 8], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([3, 0, 8])).all()

    x = StateElementNumPy(
        [1, -1, 8], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 0, 8])).all()
    x = StateElementNumPy(
        [1, 3, 8], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 2, 8])).all()

    x = StateElementNumPy(
        [1, 1, -2], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 1, 0])).all()

    x = StateElementNumPy(
        [1, 1, 10], spaces=_multidiscr_space, out_of_bounds_mode="clip"
    )
    assert (x == numpy.array([1, 1, 8])).all()


def test_typing_priority():
    x = StateElementNumPy(2.0, spaces=_discr_space, out_of_bounds_mode="warning")
    assert x == numpy.array(2, dtype=numpy.int16)
    x = StateElementNumPy([1], spaces=_cont_space, out_of_bounds_mode="warning")
    assert x == numpy.array([1], dtype=numpy.float32)


def test__array_ufunc__():
    # See https://numpy.org/doc/stable/reference/ufuncs.html#math-operations for many more

    print("inside test_arrayèufunc")
    # Discrete
    _discr_space = discrete_space([1, 2, 3])
    x = StateElementNumPy(2, spaces=_discr_space, out_of_bounds_mode="warning")
    print("la")
    y = x + numpy.array(1)
    print(y, x)
    # add, radd, sub, rsub
    assert 1 + x == 3
    assert x + 1 == 3
    assert x - 1 == 1
    assert 1 - x == -1
    # mul, rmul
    assert x * 3 == 6
    assert 3 * x == 6
    # pow
    assert x ** 2 == 4

    # Continuous
    cont_space = continuous_space(
        numpy.array([[1, 2], [3, 4]]), numpy.array([[6, 6], [6, 6]])
    )
    x = StateElementNumPy(
        numpy.array([[2, 3], [5, 5]]),
        spaces=cont_space,
        out_of_bounds_mode="warning",
    )
    # add, radd, sub, rsub
    assert (1 + x == numpy.array([[3, 4], [6, 6]])).all()
    assert (x + 1 == numpy.array([[3, 4], [6, 6]])).all()
    assert (x - 1 == numpy.array([[1, 2], [4, 4]])).all()
    # with pytest.warns(StateNotContainedWarning):
    assert (1 - x == numpy.array([[-1, -2], [-4, -4]])).all()


def test_comparisons():
    _discr_space = discrete_space([1, 2, 3])
    x = StateElementNumPy(2, spaces=_discr_space, out_of_bounds_mode="warning")
    assert x < 3
    assert x <= 2
    assert x > 1
    assert x >= 2


# if __name___ == "__main__":
test_array_init()
test_array_init_error()
test_array_init_warning()
test_array_init_clip()
test_typing_priority()
test__array_ufunc__()
test_comparisons()
