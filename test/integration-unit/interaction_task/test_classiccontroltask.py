"""This module provides tests for the ClassicControlTask class of the
coopihc package."""


from coopihc.interactiontask.ClassicControlTask import ClassicControlTask
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import autospace

import pytest
import numpy
import copy

task = None


def test_init_no_kwargs():
    global task
    timestep = 0.01
    A = numpy.eye(2)
    B = numpy.array([[1], [1]])

    task = ClassicControlTask(timestep, A, B)

    assert (task.A_d == A).all()
    assert (task.B_d == B).all()


def test_finit_no_kwargs():
    task.finit()
    assert (task.A == task.A_d).all()
    assert (task.B == task.B_d).all()


def test_reset_no_kwargs():
    task.reset()
    x = task.state["x"]
    assert (x[1:] == numpy.zeros(x[1:].shape)).all()


def test_user_step():
    u = StateElement(1, autospace([[-1]], [[1]]))
    old_x = copy.copy(task.state["x"])
    new_state, reward, is_done = task.user_step(user_action=u)
    assert (new_state["x"] == old_x + numpy.array([[1], [1]])).all()


def test_no_kwargs():
    test_init_no_kwargs()
    test_finit_no_kwargs()
    test_reset_no_kwargs()
    test_user_step()


def test_init_no_kwargs_A():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])

    task = ClassicControlTask(timestep, A, B)
    task.finit()
    task.state["x"][:] = numpy.array([[1], [1]])


def test_user_step_A():
    u = StateElement(1, autospace([[-1]], [[1]]))
    old_x = copy.copy(task.state["x"])
    new_state, reward, is_done = task.user_step(user_action=u)
    assert (
        new_state["x"] == numpy.full((2, 2), 1) @ old_x + numpy.array([[1], [1]])
    ).all()


def test_no_kwargs_A():
    test_init_no_kwargs_A()
    test_user_step_A()


def test_noise_F():
    # Incomplete test, could test the std of the new_state['x'] and see if it corresponds to Gaussian noise with std = F*sqrt(timestep)
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(timestep, A, B, F=F)
    task.finit()

    u = StateElement(1, autospace([[-1]], [[1]]))
    noise_sample = []
    for i in range(10000):
        task.state["x"][:] = numpy.array([[1], [1]])
        new_state, reward, is_done = task.user_step(user_action=u)
        noise_sample.append(new_state["x"][0].squeeze().tolist() - 3)

    mean = numpy.mean(numpy.array(noise_sample))
    assert abs(mean) <= 0.01
    assert 0 < abs(mean)


def test_noise_G():
    # incomplete test
    pass


def test_noise_H():
    # Incomplete test
    pass


def test_noise_selector():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(timestep, A, B, F=F, noise="off")
    task.finit()

    u = StateElement(1, autospace([[-1]], [[1]]))
    noise_sample = []
    for i in range(1000):
        task.state["x"][:] = numpy.array([[1], [1]])
        new_state, reward, is_done = task.user_step(user_action=u)
        noise_sample.append(new_state["x"][0].squeeze().tolist() - 3)

    mean = numpy.mean(numpy.array(noise_sample))
    assert 0 == abs(mean)


def test_noise():
    test_noise_F()
    test_noise_G()
    test_noise_H()
    test_noise_selector()


def test_discrete_dynamics_discrete_timespace():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(
        timestep, A, B, discrete_dynamics=True, timespace="discrete"
    )
    task.finit()

    assert (task.A_d == A).all()
    assert (task.B_d == B).all()
    assert (task.A_c == (A - numpy.eye(2)) / timestep).all()
    assert (task.B_c == 1 / timestep * B).all()
    assert (task.A == task.A_d).all()
    assert (task.B_d == task.B).all()


def test_discrete_dynamics_continuous_timespace():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(
        timestep, A, B, discrete_dynamics=True, timespace="continuous"
    )
    task.finit()

    assert (task.A_d == A).all()
    assert (task.B_d == B).all()
    assert (task.A_c == (A - numpy.eye(2)) / timestep).all()
    assert (task.B_c == 1 / timestep * B).all()
    assert (task.A == task.A_c).all()
    assert (task.B_c == task.B).all()


def test_continuous_dynamics_discrete_timespace():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(
        timestep, A, B, discrete_dynamics=False, timespace="discrete"
    )
    task.finit()

    assert (task.A_c == A).all()
    assert (task.B_c == B).all()
    assert (task.A_d == numpy.eye(2) + timestep * A).all()
    assert (task.B_d == timestep * B).all()
    assert (task.A == task.A_d).all()
    assert (task.B_d == task.B).all()


def test_continuous_dynamics_continuous_timespace():
    global task
    timestep = 0.01
    A = numpy.array([[1, 1], [1, 1]])
    B = numpy.array([[1], [1]])
    F = numpy.eye(2)

    task = ClassicControlTask(
        timestep, A, B, discrete_dynamics=False, timespace="continuous"
    )
    task.finit()

    assert (task.A_c == A).all()
    assert (task.B_c == B).all()
    assert (task.A_d == numpy.eye(2) + timestep * A).all()
    assert (task.B_d == timestep * B).all()
    assert (task.A == task.A_c).all()
    assert (task.B_c == task.B).all()


def test_dynamics_timespace():
    test_discrete_dynamics_discrete_timespace()
    test_discrete_dynamics_continuous_timespace()
    test_continuous_dynamics_discrete_timespace()
    test_continuous_dynamics_continuous_timespace()


if __name__ == "__main__":
    test_no_kwargs()
    test_no_kwargs_A()
    test_noise()
    test_dynamics_timespace()
