from coopihc.base.StateElement import StateElement
from coopihc.base.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
    NumpyFunctionNotHandledWarning,
    integer_set,
    lin_space,
    box_space,
    space,
)

import numpy
import pytest
import json
import copy
from tabulate import tabulate


def test_array_init_integer():
    x = StateElement(2, integer_set(3))
    assert hasattr(x, "space")
    assert x.shape == ()
    assert x == 2


def test_array_init_numeric():
    x = StateElement(
        numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))), out_of_bounds_mode="error"
    )
    assert hasattr(x, "space")
    assert x.shape == (2, 2)
    assert (x == numpy.zeros((2, 2))).all()


def test_array_init():
    test_array_init_integer()
    test_array_init_numeric()


def test_array_init_error_integer():
    x = StateElement(2, integer_set(3), out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(4, integer_set(3), out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x = StateElement(-3, integer_set(3), out_of_bounds_mode="error")


def test_array_init_error_numeric():
    x = StateElement(
        numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))), out_of_bounds_mode="error"
    )
    with pytest.raises(StateNotContainedError):
        x = StateElement(
            2 * numpy.ones((2, 2)),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="error",
        )
    with pytest.raises(StateNotContainedError):
        x = StateElement(
            -2 * numpy.ones((2, 2)),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="error",
        )
    with pytest.raises(StateNotContainedError):
        x = StateElement(
            numpy.array([[0, 0], [-2, 0]]),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="error",
        )


def test_array_init_error():
    test_array_init_error_integer()
    test_array_init_error_numeric()


def test_array_init_warning_integer():
    x = StateElement(2, integer_set(3), out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(4, integer_set(3), out_of_bounds_mode="warning")
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(-3, integer_set(3), out_of_bounds_mode="warning")


def test_array_init_warning_numeric():
    x = StateElement(
        numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))), out_of_bounds_mode="warning"
    )
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            2 * numpy.ones((2, 2)),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="warning",
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            -2 * numpy.ones((2, 2)),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="warning",
        )
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            numpy.array([[0, 0], [-2, 0]]),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="warning",
        )


def test_array_init_warning():
    test_array_init_warning_integer()
    test_array_init_warning_numeric()


def test_array_init_clip_integer():
    x = StateElement(2, integer_set(3), out_of_bounds_mode="clip")
    assert x == numpy.array([2])
    x = StateElement(4, integer_set(3), out_of_bounds_mode="clip")
    assert x == numpy.array([2])
    x = StateElement(-3, integer_set(3), out_of_bounds_mode="clip")
    assert x == numpy.array([0])


def test_array_init_clip_numeric():
    x = StateElement(
        numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))), out_of_bounds_mode="clip"
    )
    assert (x == numpy.zeros((2, 2))).all()
    x = StateElement(
        2 * numpy.ones((2, 2)),
        box_space(numpy.ones((2, 2))),
        out_of_bounds_mode="clip",
    )
    assert (x == numpy.ones((2, 2))).all()
    x = StateElement(
        -2 * numpy.ones((2, 2)),
        box_space(numpy.ones((2, 2))),
        out_of_bounds_mode="clip",
    )
    assert (x == -1.0 * numpy.ones((2, 2))).all()


def test_array_init_clip():
    test_array_init_clip_integer()
    test_array_init_clip_numeric()


def test_array_init_dtype_integer():
    x = StateElement(2, integer_set(3), out_of_bounds_mode="warning")
    assert x.dtype == numpy.int64
    x = StateElement(2, integer_set(3, dtype=numpy.int16), out_of_bounds_mode="warning")
    assert x.dtype == numpy.int16


def test_array_init_dtype_numeric():
    x = StateElement(
        numpy.zeros((2, 2)),
        box_space(numpy.ones((2, 2))),
        out_of_bounds_mode="warning",
    )
    assert x.dtype == numpy.float64
    x = StateElement(
        numpy.zeros((2, 2)),
        box_space(numpy.ones((2, 2), dtype=numpy.float32)),
        out_of_bounds_mode="warning",
    )
    assert x.dtype == numpy.float32
    x = StateElement(
        numpy.zeros((2, 2)),
        box_space(numpy.ones((2, 2), dtype=numpy.int8)),
        out_of_bounds_mode="warning",
    )
    assert x.dtype == numpy.int8


def test_array_init_dtype():
    test_array_init_dtype_integer()
    test_array_init_dtype_numeric()


# def test__array_ufunc__discrete():
#     # Simple arithmetic
#     global discr_space
#     x = StateElement(2, discr_space, out_of_bounds_mode="error")
#     assert x + numpy.array(1) == 3
#     assert x + 1 == 3
#     assert x - 1 == 1
#     assert 3 - x == 1
#     assert x - numpy.array(1) == 1
#     assert numpy.array(3) - x == 1
#     assert 1 + x == 3
#     x += 1
#     y = x - 1
#     assert y.out_of_bounds_mode == "error"

#     with pytest.raises(StateNotContainedError):
#         1 - x
#     with pytest.raises(StateNotContainedError):
#         x + 2
#     with pytest.raises(StateNotContainedError):
#         x += 5


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


# def test__array_ufunc__():
#     test__array_ufunc__discrete()
#     test__array_ufunc__continuous()
#     test__array_ufunc__multidiscrete()
#     test__array_ufunc__comparisons()
#     test__array_ufunc__trigonometry()
#     test__array_ufunc__floating()
#     test__array_ufunc__out_of_bounds_mode()


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


def test_equals_integer():
    int_space = integer_set(3)
    other_int_space = integer_set(4)
    x = StateElement(numpy.array(1), int_space)
    y = StateElement(numpy.array(1), other_int_space)
    assert x.equals(y)
    assert not x.equals(y, mode="hard")
    z = StateElement(numpy.array(2), int_space)
    assert not x.equals(z)


def test_equals_numeric():
    numeric_space = box_space(numpy.ones((2, 2)))
    other_numeric_space = box_space(
        low=numpy.array([[-1, -1], [-1, -2]]), high=numpy.array([[1, 2], [1, 1]])
    )

    x = StateElement(numpy.zeros((2, 2)), numeric_space)
    y = StateElement(numpy.zeros((2, 2)), other_numeric_space)

    assert (x.equals(y)).all()
    assert not (x.equals(y, mode="hard")).all()
    z = StateElement(numpy.eye(2), numeric_space)
    assert not (x.equals(z)).all()


def test_equals():
    test_equals_integer()
    test_equals_numeric()


def test__iter__integer():
    x = StateElement([2], integer_set(3))
    with pytest.raises(TypeError):
        next(iter(x))


def test__iter__numeric():
    x = StateElement(
        numpy.array([[0.2, 0.3], [0.4, 0.5]]), box_space(numpy.ones((2, 2)))
    )
    for i, _x in enumerate(x):
        if i == 0:
            assert (
                _x == StateElement(numpy.array([0.2, 0.3]), box_space(numpy.ones((2,))))
            ).all()
        if i == 1:
            assert (
                _x == StateElement(numpy.array([0.4, 0.5]), box_space(numpy.ones((2,))))
            ).all()
        for j, _xx in enumerate(_x):
            if i == 0 and j == 0:
                assert _xx == StateElement(
                    numpy.array(0.2), box_space(numpy.float64(1))
                )
            elif i == 0 and j == 1:
                assert _xx == StateElement(
                    numpy.array(0.3), box_space(numpy.float64(1))
                )
            elif i == 1 and j == 0:
                assert _xx == StateElement(
                    numpy.array(0.4), box_space(numpy.float64(1))
                )
            elif i == 1 and j == 1:
                assert _xx == StateElement(
                    numpy.array(0.5), box_space(numpy.float64(1))
                )


def test__iter__():
    test__iter__integer()
    test__iter__numeric()


def test__repr__integer():
    x = StateElement(2, integer_set(3))
    assert x.__repr__() == "StateElement(array(2), CatSet([0 1 2]), 'warning')"


def test__repr__numeric():
    x = StateElement(numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))))
    x.__repr__()


def test__repr__():
    test__repr__integer()
    test__repr__numeric()


def test_serialize_integer():
    x = StateElement(numpy.array([2]), integer_set(3))
    assert x.serialize() == {
        "values": 2,
        "space": {
            "space": "CatSet",
            "seed": None,
            "array": [0, 1, 2],
            "dtype": "dtype[int64]",
        },
    }


def test_serialize_numeric():
    x = StateElement(numpy.zeros((2, 2)), box_space(numpy.ones((2, 2))))
    assert x.serialize() == {
        "values": [[0.0, 0.0], [0.0, 0.0]],
        "space": {
            "space": "Numeric",
            "seed": None,
            "low,high": [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            "shape": (2, 2),
            "dtype": "dtype[float64]",
        },
    }


def test_serialize():
    test_serialize_integer()
    test_serialize_numeric()


def test__getitem__integer():
    x = StateElement(1, integer_set(3))
    assert x[..., {"space": True}] == x
    assert x[..., {"space": True}] is x
    assert x[...] == x


def test__getitem__numeric():
    x = StateElement(
        numpy.array([[0.0, 0.1], [0.2, 0.3]]), box_space(numpy.ones((2, 2)))
    )
    assert x[0, 0] == 0.0
    assert x[0, 0, {"space": True}] == StateElement(0.0, box_space(numpy.float64(1)))

    assert x[0, 1, {"space": True}] == StateElement(0.1, box_space(numpy.float64(1)))

    assert x[1, 0, {"space": True}] == StateElement(0.2, box_space(numpy.float64(1)))

    assert x[1, 1, {"space": True}] == StateElement(0.3, box_space(numpy.float64(1)))

    assert (x[:, 1] == numpy.array([0.1, 0.3])).all()
    assert (
        x[:, 1, {"space": True}]
        == StateElement(numpy.array([0.1, 0.3]), box_space(numpy.ones((2,))))
    ).all()


def test__getitem__():
    test__getitem__integer()
    test__getitem__numeric()


def test__setitem__integer():
    x = StateElement(1, integer_set(3))
    x[...] = 2
    assert x == StateElement(2, integer_set(3))
    with pytest.warns(StateNotContainedWarning):
        x[...] = 4


def test__setitem__numeric():
    x = StateElement(
        numpy.array([[0.0, 0.1], [0.2, 0.3]]), box_space(numpy.ones((2, 2)))
    )
    x[0, 0] = 0.5
    x[0, 1] = 0.6
    x[1, 0] = 0.7
    x[1, 1] = 0.8

    assert (
        x
        == StateElement(
            numpy.array([[0.5, 0.6], [0.7, 0.8]]), box_space(numpy.ones((2, 2)))
        )
    ).all()

    with pytest.warns(StateNotContainedWarning):
        x[0, 0] = 1.3

    x = StateElement(
        numpy.array([[0.0, 0.1], [0.2, 0.3]]),
        box_space(numpy.ones((2, 2))),
        out_of_bounds_mode="clip",
    )
    x[:, 0] = numpy.array([0.9, 0.9])
    x[0, :] = numpy.array([1.2, 0.2])
    x[1, 1] = 0.5

    assert (
        x
        == StateElement(
            numpy.array([[1, 0.2], [0.9, 0.5]]),
            box_space(numpy.ones((2, 2))),
            out_of_bounds_mode="clip",
        )
    ).all()


def test__setitem__():
    test__setitem__integer()
    test__setitem__numeric()


def test_reset_integer():
    x = StateElement(numpy.array([2]), integer_set(3), out_of_bounds_mode="error")
    xset = {}
    for i in range(1000):
        x.reset()
        _x = x.squeeze().tolist()
        xset.update({str(_x): _x})
    assert sorted(xset.values()) == [0, 1, 2]
    # forced reset:
    x.reset(value=0)
    assert x == StateElement(0, integer_set(3), out_of_bounds_mode="error")
    with pytest.raises(StateNotContainedError):
        x.reset(value=5)
    x.out_of_bounds_mode = "clip"
    x.reset(value=5)
    assert x == StateElement(
        numpy.array([2]), integer_set(3), out_of_bounds_mode="clip"
    )


def test_reset_numeric():
    x = StateElement(numpy.ones((2, 2)), box_space(numpy.ones((2, 2))))
    for i in range(1000):
        x.reset()
    x.reset(0.59 * numpy.ones((2, 2)))
    assert (
        x == StateElement(0.59 * numpy.ones((2, 2)), box_space(numpy.ones((2, 2))))
    ).all()


def test_reset():
    test_reset_integer()
    test_reset_numeric()


def test_tabulate_integer():
    x = StateElement(1, integer_set(3))
    x._tabulate()
    tabulate(x._tabulate()[0])


def test_tabulate_numeric():
    x = StateElement(numpy.zeros((3, 3)), box_space(numpy.ones((3, 3))))
    x._tabulate()
    tabulate(x._tabulate()[0])


def test_tabulate():
    test_tabulate_integer()
    test_tabulate_numeric()


def test_cast_discrete_to_cont():
    discr_box_space = box_space(low=numpy.int8(1), high=numpy.int8(3))
    cont_box_space = box_space(low=numpy.float64(-1.5), high=numpy.float64(1.5))

    x = StateElement(1, discr_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="edges")
    assert ret_stateElem == StateElement(-1.5, cont_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="center")
    assert ret_stateElem == StateElement(-1, cont_box_space)

    x = StateElement(2, discr_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="edges")
    assert ret_stateElem == StateElement(0, cont_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="center")
    assert ret_stateElem == StateElement(0, cont_box_space)

    x = StateElement(3, discr_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="edges")
    assert ret_stateElem == StateElement(1.5, cont_box_space)
    ret_stateElem = x.cast(cont_box_space, mode="center")
    assert ret_stateElem == StateElement(1, cont_box_space)


def test_cast_cont_to_discrete():
    cont_box_space = box_space(low=numpy.float64(-1.5), high=numpy.float64(1.5))
    discr_box_space = box_space(low=numpy.int8(1), high=numpy.int8(3))
    x = StateElement(0, cont_box_space)
    ret_stateElem = x.cast(discr_box_space, mode="center")
    assert ret_stateElem == StateElement(2, discr_box_space)
    ret_stateElem = x.cast(discr_box_space, mode="edges")
    assert ret_stateElem == StateElement(2, discr_box_space)

    center = []
    edges = []
    for i in numpy.linspace(-1.5, 1.5, 100):
        x = StateElement(i, cont_box_space)
        ret_stateElem = x.cast(discr_box_space, mode="center")
        if i < -0.75:
            assert ret_stateElem == StateElement(1, discr_box_space)
        if i > -0.75 and i < 0.75:
            assert ret_stateElem == StateElement(2, discr_box_space)
        if i > 0.75:
            assert ret_stateElem == StateElement(3, discr_box_space)
        center.append(ret_stateElem.tolist())

        ret_stateElem = x.cast(discr_box_space, mode="edges")
        if i < -0.5:
            assert ret_stateElem == StateElement(1, discr_box_space)
        if i > -0.5 and i < 0.5:
            assert ret_stateElem == StateElement(2, discr_box_space)
        if i > 0.5:
            assert ret_stateElem == StateElement(3, discr_box_space)

        edges.append(ret_stateElem.tolist())

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(
    #     numpy.linspace(-1.5, 1.5, 100), numpy.array(center) - 0.05, "+", label="center"
    # )
    # ax.plot(
    #     numpy.linspace(-1.5, 1.5, 100), numpy.array(edges) + 0.05, "o", label="edges"
    # )
    # ax.legend()
    # plt.show()


def test_cast_cont_to_cont():
    cont_space = box_space(numpy.full((2, 2), 1), dtype=numpy.float32)
    other_cont_space = box_space(
        low=numpy.full((2, 2), 0), high=numpy.full((2, 2), 4), dtype=numpy.float32
    )

    for i in numpy.linspace(-1, 1, 100):
        x = StateElement(numpy.full((2, 2), i), cont_space)
        ret_stateElement = x.cast(other_cont_space)
        assert (ret_stateElement == (x + 1) * 2).all()


def test_cast_discr_to_discr():
    discr_box_space = box_space(low=numpy.int8(1), high=numpy.int8(4))
    other_discr_box_space = box_space(low=numpy.int8(11), high=numpy.int8(14))

    for i in [1, 2, 3, 4]:
        x = StateElement(i, discr_box_space)
        ret_stateElement = x.cast(other_discr_box_space)
        assert ret_stateElement == x + 10


def test_cast():
    test_cast_discrete_to_cont()
    test_cast_cont_to_discrete()
    test_cast_cont_to_cont()
    test_cast_discr_to_discr()


if __name__ == "__main__":
    test_array_init()
    test_array_init_error()
    test_array_init_warning()
    test_array_init_clip()
    test_array_init_dtype()
    # test__array_ufunc__() # kept here just in case
    # test__array_function__() # kept here just in case
    test_equals()
    test__iter__()
    test__repr__()
    test_serialize()
    test__setitem__()
    test__getitem__()
    test_reset()
    test_tabulate()
    test_cast()
