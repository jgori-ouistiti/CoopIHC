from coopihc.base.StateElement import StateElement
from coopihc.base.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
)
from coopihc import discrete_array_element, array_element

import numpy
import pytest
import json
import copy
from tabulate import tabulate


def test__array_ufunc__():
    # Simple arithmetic
    x = discrete_array_element(low=1, high=3, init=2, out_of_bounds_mode="error")
    _space = x.space
    assert not isinstance((x + 1), StateElement)

    x += 1
    assert isinstance(x, StateElement)
    assert x.out_of_bounds_mode == "error"
    assert x.space == _space
    with pytest.raises(StateNotContainedError):
        x += 5
    x = discrete_array_element(low=1, high=3, init=2, out_of_bounds_mode="error")
    numpy.add.at(numpy.atleast_1d(x), 0, 1)
    with pytest.raises(StateNotContainedError):
        numpy.add.at(numpy.atleast_1d(x), 0, 5)

    x = discrete_array_element(low=1, high=3, init=2, out_of_bounds_mode="clip")

    assert x + 10 == 12
    x += 10
    assert x == 3
    numpy.add.at(numpy.atleast_1d(x), 0, 4)
    assert x == 3

    x = array_element(init=numpy.ones((2, 2)))
    y = array_element(init=numpy.ones((2, 2)))
    assert (x - y).shape == x.shape


# def test__array_ufunc__continuous():
#     # some matrix operations
#     global cont_space
#     x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
#     assert (x + numpy.ones((2, 2)) == numpy.ones((2, 2))).all()
#     assert (x + 1 == numpy.ones((2, 2))).all()
#     assert (1 + x == numpy.ones((2, 2))).all()
#     assert (x - 1 == -numpy.ones((2, 2))).all()
#     assert (1 - x == numpy.ones((2, 2))).all()
#     assert ((1 + x) * 0.5 == 0.5 * numpy.ones((2, 2))).all()
#     assert (0.5 * (1 + x) @ numpy.ones((2, 2)) == numpy.ones((2, 2))).all()


# def test__array_ufunc__multidiscrete():
#     global multidiscr_space
#     x = StateElement([1, 1, 8], multidiscr_space, out_of_bounds_mode="error")
#     assert (x + numpy.array([[1], [1], [-3]]) == numpy.array([[2], [2], [5]])).all()
#     with pytest.raises(StateNotContainedError):
#         x + numpy.array([[1], [1], [1]])


# def test__array_ufunc__comparisons():
#     global discr_space
#     x = StateElement(2, discr_space, out_of_bounds_mode="error")
#     assert x > 1 == True
#     global cont_space
#     x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
#     assert (x < 0).all() == False
#     global multidiscr_space
#     x = StateElement(
#         numpy.array([[1], [1], [1]]), multidiscr_space, out_of_bounds_mode="error"
#     )
#     assert (x >= numpy.array([[1], [0], [1]])).all() == True
#     assert (x >= numpy.array([[1], [5], [1]])).all() == False
#     comp = x >= numpy.array([[1], [5], [1]])
#     assert (comp == numpy.array([[True], [False], [True]])).all()


# def test__array_ufunc__trigonometry():
#     global cont_space
#     x = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")
#     assert (numpy.cos(x) == numpy.ones((2, 2))).all()


# def test__array_ufunc__floating():
#     global cont_space
#     x = StateElement(
#         numpy.array([[0.2, 0.3], [1, 0.95]]), cont_space, out_of_bounds_mode="error"
#     )
#     assert numpy.isfinite(x).all() == True


# def test__array_ufunc__out_of_bounds_mode():
#     x = StateElement(
#         numpy.array([[0.2, 0.3], [1, 0.95]]), cont_space, out_of_bounds_mode="error"
#     )
#     y = StateElement(
#         numpy.array([[-0.2, -0.3], [-1, -0.95]]),
#         cont_space,
#         out_of_bounds_mode="warning",
#     )
#     z = StateElement(
#         numpy.array([[0.0, 0.0], [0.0, 0.0]]),
#         cont_space,
#         out_of_bounds_mode="silent",
#     )
#     u = x + y
#     assert u.out_of_bounds_mode == "error"
#     u = y + x
#     assert u.out_of_bounds_mode == "error"
#     u = z + x
#     assert u.out_of_bounds_mode == "error"
#     u = y + z
#     assert u.out_of_bounds_mode == "warning"
#     u = z + 0
#     assert u.out_of_bounds_mode == "silent"


def test__array_ufunc():
    test__array_ufunc__()
    # test__array_ufunc__continuous()
    # test__array_ufunc__multidiscrete()
    # test__array_ufunc__comparisons()
    # test__array_ufunc__trigonometry()
    # test__array_ufunc__floating()
    # test__array_ufunc__out_of_bounds_mode()


# def test_amax_nothandled():
#     StateElement.HANDLED_FUNCTIONS = {}
#     cont_space = autospace(
#         [[-1, -1], [-1, -1]], [[1, 1], [1, 1]], dtype=numpy.float64
#     )  # Here the
#     x = StateElement(
#         numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
#         cont_space,
#         out_of_bounds_mode="warning",
#     )

#     # Without handled function
#     with pytest.warns(NumpyFunctionNotHandledWarning):
#         y = numpy.max(x)
#     assert isinstance(y, numpy.ndarray)
#     assert not isinstance(y, StateElement)
#     assert y == 0.8
#     assert not hasattr(y, "space")
#     assert not hasattr(y, "out_of_bounds_mode")


# def test_amax_implements_decorator():
#     cont_space = autospace([[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64)
#     x = StateElement(
#         numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
#         cont_space,
#         out_of_bounds_mode="warning",
#     )

#     @StateElement.implements(numpy.amax)
#     def amax(arr, **keywordargs):
#         space, out_of_bounds_mode, kwargs = (
#             arr.space,
#             arr.out_of_bounds_mode,
#             arr.kwargs,
#         )
#         obj = arr.view(numpy.ndarray)
#         argmax = numpy.argmax(obj, **keywordargs)
#         index = numpy.unravel_index(argmax, arr.space.shape)
#         obj = numpy.amax(obj, **keywordargs)
#         obj = numpy.asarray(obj).view(StateElement)
#         if arr.space.space_type == "continuous":
#             obj.space = autospace(
#                 numpy.atleast_2d(arr.space.low[index[0], index[1]]),
#                 numpy.atleast_2d(arr.space.high[index[0], index[1]]),
#             )
#         else:
#             raise NotImplementedError
#         obj.out_of_bounds_mode = arr.out_of_bounds_mode
#         obj.kwargs = arr.kwargs
#         return obj

#     y = numpy.amax(x)
#     assert isinstance(y, StateElement)
#     assert StateElement.HANDLED_FUNCTIONS.get(numpy.amax) is not None
#     assert x.HANDLED_FUNCTIONS.get(numpy.amax) is not None
#     assert y.shape == ()
#     assert y == 0.8
#     assert y.space.space_type == "continuous"
#     assert y.space.shape == (1, 1)
#     assert y.space.low == numpy.array([[-2]])
#     assert y.space.high == numpy.array([[3]])


# def test_array_function_simple():
#     test_amax_nothandled()
#     test_amax_implements_decorator()


# def test__array_function__():
#     test_array_function_simple()

if __name__ == "__main__":

    test__array_ufunc()  # kept here just in case
    # test__array_function__() # kept here just in case
