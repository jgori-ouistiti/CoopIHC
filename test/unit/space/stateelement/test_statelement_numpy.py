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


def test_array_init_multidiscrete():
    global multidiscr_space
    multidiscr_space = multidiscrete_space(
        [
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1, 3, 5, 8]),
        ]
    )
    x = StateElement([1, 5], multidiscr_space, out_of_bounds_mode="error")
    assert hasattr(x, "spaces")
    assert isinstance(x.spaces, Space)
    assert x.shape == (2, 1)
    assert (x == numpy.array([[1], [5]], dtype=numpy.int16)).all()


def test_array_init():
    test_array_init_discrete()
    test_array_init_continuous()
    test_array_init_multidiscrete()


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


def test__array_ufunc__():
    # math operations
    test__array_ufunc__discrete()
    test__array_ufunc__continuous()
    test__array_ufunc__multidiscrete()
    # comparison
    test__array_ufunc__comparisons()
    # trigonometric
    test__array_ufunc__trigonometry()
    # floating
    test__array_ufunc__floating()


# def test_squeeze():
#     cont_space = continuous_space(numpy.array([[4]]), numpy.array([[6]]))
#     x =  StateElement(
#         numpy.array([[5]]),
#         spaces=cont_space,
#         out_of_bounds_mode="warning",
#     )

#     ## Without handled function
#     y = numpy.squeeze(x)
#     with pytest.warns(NumpyFunctionNotHandledWarning):
#         y = numpy.squeeze(x)
#     assert isinstance(y, numpy.ndarray)
#     assert not isinstance(y,  StateElement)
#     assert y.shape == ()
#     assert not hasattr(y, "spaces")
#     assert not hasattr(y, "out_of_bounds_mode")


if __name__ == "__main__":

    test_array_init()
    test_array_init_error()
    test_array_init_warning()
    test_array_init_clip()
    test_array_init_dtype()
    test__array_ufunc__()
    # test_squeeze()
