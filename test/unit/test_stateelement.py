from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    SpaceLengthError,
    StateNotContainedError,
    StateNotContainedWarning,
)

from coopihc.helpers import flatten
import numpy
import copy
import pytest


def test_init_simple():
    # ============================= check if lists are applied if inputs not in list form
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

    # ========================= clipping mode (assumes typing priority set to default (= space))
    # ---------------- error
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
    # --------------- warning
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
    # --------------- clip
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

    # ====================== Typing priority
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
    assert x["values"][0].dtype == numpy.int16
    assert x["values"][0].dtype == numpy.int16
    x.typing_priority = "value"
    x["values"] = [0, 2.0, -3.0]
    assert x["values"][0].dtype == numpy.int16
    assert x["values"][0].dtype == numpy.float32
    assert x["values"][0].dtype == numpy.float32


# ===========================================================
#     gridsize = (11, 11)
#     number_of_targets = 3
#     y = StateElement(
#         values=None,
#         spaces=[
#             Space(
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#             )
#             for j in range(number_of_targets)
#         ],
#         clipping_mode="error",
#     )


# x = StateElement(
#     values=None,
#     spaces=[
#         coopihc.space.Space(
#             [
#                 numpy.array([-1], dtype=numpy.float32),
#                 numpy.array([1], dtype=numpy.float32),
#             ]
#         ),
#         coopihc.space.Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
#         coopihc.space.Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
#     ],
# )


# gridsize = (11, 11)
# number_of_targets = 3
# y = StateElement(
#     values=None,
#     spaces=[
#         Space(
#             [
#                 numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                 numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#             ]
#         )
#         for j in range(number_of_targets)
#     ],
#     clipping_mode="error",
# )
