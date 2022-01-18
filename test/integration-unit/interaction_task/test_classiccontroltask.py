"""This module provides tests for the ClassicControlTask class of the
coopihc package."""


import numpy
from coopihc.interactiontask.ClassicControlTask import ClassicControlTask

import pytest


def test_init_no_kwargs():
    timestep = 0.01
    A = numpy.eye(2)
    B = numpy.array([[1], [1]])

    task = ClassicControlTask(timestep, A, B)

    assert (task.A_d == A).all()
    assert (task.B_d == B).all()


def test_no_kwargs():
    test_init_no_kwargs()


# task = ClassicControlTask(
#     timestep,
#     A,
#     B,
#     *args,
#     F=None,
#     G=None,
#     H=None,
#     discrete_dynamics=True,
#     noise="on",
#     timespace="discrete",
#     **kwargs
# )


if __name__ == "__main__":
    test_no_kwargs()
