from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    StateNotContainedError,
    StateNotContainedWarning,
)

import numpy
import pytest


def test_lists_applied_if_inputs_not_in_list_form():
    """Tests if lists are applied if inputs are not in list form."""
    x = StateElement(
        values=None,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert x["values"] == numpy.array([None])
    assert x["spaces"] == [
        Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        )
    ]


def test_clipping_mode_error():
    """Tests that appropriate error is raised by clipping mode."""
    with pytest.raises(StateNotContainedError):
        x = StateElement(
            values=3.0,
            spaces=Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            clipping_mode="error",
        )


def test_clipping_mode_warning():
    """Tests that appropriate error is raised by clipping mode."""
    with pytest.warns(StateNotContainedWarning):
        x = StateElement(
            values=3.0,
            spaces=Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            clipping_mode="warning",
        )
    with pytest.warns(StateNotContainedWarning):
        y = StateElement(
            values=-3.0,
            spaces=Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            clipping_mode="warning",
        )
    assert x["values"] == numpy.array([[3]], dtype=numpy.float32)
    assert y["values"] == numpy.array([[-3]], dtype=numpy.float32)


def test_clipping():
    """Tests that clipping works."""
    x = StateElement(
        values=3.0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
        clipping_mode="clip",
    )
    y = StateElement(
        values=-3.0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
        clipping_mode="clip",
    )
    assert x["values"] == numpy.array([[1]], dtype=numpy.float32)
    assert y["values"] == numpy.array([[-1]], dtype=numpy.float32)


def test_clipping_mode():
    """Tests clipping mode (assumes typing priority set to
    default (= space))."""
    test_clipping_mode_error()
    test_clipping_mode_warning()
    test_clipping()


def test_typing_priority():
    """Tests clipping mode (assumes typing priority set to
    default (= space))."""
    x = StateElement(
        values=3,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
        typing_priority="space",
    )
    assert x["values"][0].dtype == numpy.float32
    x = StateElement(
        values=numpy.array([3], dtype=numpy.int16),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
        typing_priority="value",
    )
    assert x["values"][0].dtype == numpy.int16


def test_init_simple():
    test_lists_applied_if_inputs_not_in_list_form()
    test_clipping_mode()
    test_typing_priority()


def test_init_more_complex():
    x = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )
    # ---------------------- testing values = None mechanism
    assert (x["values"] == numpy.array([None, None, None])).all()

    # ========================= clipping mode (assumes typing priority set to default (= space))
    # ---------------- error
    x.clipping_mode = "error"
    with pytest.raises(StateNotContainedError):
        x["values"] = [0, 2, 0]
    with pytest.raises(StateNotContainedError):
        x["values"] = [0, 0, -3]
    with pytest.raises(StateNotContainedError):
        x["values"] = [2, 2, -3]
    with pytest.raises(StateNotContainedError):
        x["values"] = [-2, -2, 2]
    x["values"] = [0, 2, -3]

    x.clipping_mode = "warning"
    with pytest.warns(StateNotContainedWarning):
        x["values"] = [0, 2, 0]
    with pytest.warns(StateNotContainedWarning):
        x["values"] = [0, 0, -3]
    with pytest.warns(StateNotContainedWarning):
        x["values"] = [2, 2, -3]
    with pytest.warns(StateNotContainedWarning):
        x["values"] = [-2, -2, 2]
    x["values"] = [0, 2, -3]

    x.clipping_mode = "clip"
    x["values"] = [0, 2, 0]
    print(x["values"])
    assert x["values"] == [
        numpy.array([[0.0]], dtype=numpy.float32),
        numpy.array([[2]], dtype=numpy.int16),
        numpy.array([[-1]], dtype=numpy.int16),
    ]
    x["values"] = [0, 0, -3]
    assert x["values"] == [
        numpy.array([[0.0]], dtype=numpy.float32),
        numpy.array([[1]], dtype=numpy.int16),
        numpy.array([[-3]], dtype=numpy.int16),
    ]
    x["values"] = [2, 2, -3]
    assert x["values"] == [
        numpy.array([[1.0]], dtype=numpy.float32),
        numpy.array([[2]], dtype=numpy.int16),
        numpy.array([[-3]], dtype=numpy.int16),
    ]
    x["values"] = [-2, -2, 2]
    assert x["values"] == [
        numpy.array([[-1.0]], dtype=numpy.float32),
        numpy.array([[1]], dtype=numpy.int16),
        numpy.array([[-1]], dtype=numpy.int16),
    ]
    x["values"] = [0, 2, -3]

    # ====================== Typing priority
    # This test is currently not passed, solve this.
    x.clipping_mode = "error"
    x.typing_priority = "space"
    x["values"] = [0, 2.0, -3.0]
    assert x["values"][0].dtype == numpy.float32
    assert x["values"][1].dtype == numpy.int16
    assert x["values"][2].dtype == numpy.int16
    x.typing_priority = "value"
    x["values"] = [0, 2.0, -3.0]
    assert x["values"][0].dtype == numpy.int64
    assert x["values"][1].dtype == numpy.float64
    assert x["values"][2].dtype == numpy.float64


def test_init():
    """Tests the initializer functions."""
    test_init_simple()
    test_init_more_complex()


####### Comparisons

######    __eq__
######    __lt__
######    __gt__
######    __le__
######    __ge__


def test_compare_eq():
    x = StateElement(
        values=1.0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    y = StateElement(
        values=[0, 2, -4],
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )

    assert x == StateElement(
        values=1.0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert x != StateElement(
        values=0.5,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert x != StateElement(
        values=1.0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float64),
                numpy.array([1], dtype=numpy.float64),
            ]
        ),
    )
    assert x == StateElement(
        values=numpy.array([1.0], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float64),
                numpy.array([1], dtype=numpy.float64),
            ]
        ),
        typing_priority="value",
    )

    assert y == StateElement(
        values=[0, 2, -4],
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )
    assert y != StateElement(
        values=[0, 3, -4],
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )


def test_compare_lt():
    pass


def test_compare_gt():
    pass


def test_compare_le():
    pass


def test_compare_ge():
    pass


def test_comparison():
    """Tests the comparison methods."""
    test_compare_eq()
    test_compare_lt()
    test_compare_gt()
    test_compare_le()
    test_compare_ge()


####### Arithmetic

######    __neg__
######    __add__
######    __radd__
######    __sub__
######    __rsub__
######    __mul__
######    __rmul__
######    __pow__
######    __matmul__
######    __rmatmul__


x = StateElement(
    values=0.2,
    spaces=Space(
        [
            numpy.array([-1], dtype=numpy.float32),
            numpy.array([1], dtype=numpy.float32),
        ]
    ),
)
y = StateElement(
    values=0.2,
    spaces=Space(
        [
            numpy.array([-1], dtype=numpy.float32),
            numpy.array([1], dtype=numpy.float32),
        ]
    ),
)


def test_neg():
    y["values"] = -0.2
    assert -x == y
    assert x == -y
    assert -(-x) == x


def test_add_radd():
    y["values"] = 0.2
    assert x + y == StateElement(
        values=0.4,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert y + x == StateElement(
        values=0.4,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    a = 0.5
    assert x + a == StateElement(
        values=0.7,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert a + x == StateElement(
        values=0.7,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )


def test_sub_rsub():
    assert x - y == StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert y - x == StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    b = 0.2
    assert b - x == StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    assert x - b == StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )


def test_mul():
    pass


def test_rmul():
    pass


def test_pow():
    pass


def test_matmul():
    pass


def test_rmatmul():
    pass


def test_arithmetic():
    test_neg()
    test_add_radd()
    test_sub_rsub()

    test_mul()
    test_rmul()
    test_pow()
    test_matmul()
    test_rmatmul()


def test__getattr__switch():
    x = StateElement(
        values=0.2,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    # test_init()
    # test_comparison()
    # test_arithmetic()
    x = StateElement(
        values=0.2,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    x.extract()
    numpy.squeeze(x)
