from coopihc.interactiontask.ClassicControlTask import ClassicControlTask
from coopihc.agents.lqrcontrollers.IHDT_LQRController import IHDT_LQRController
from coopihc.bundle.Bundle import Bundle


import numpy


I = 0.25
b = 0.2
ta = 0.03
te = 0.04

a1 = b / (ta * te * I)
a2 = 1 / (ta * te) + (1 / ta + 1 / te) * b / I
a3 = b / I + 1 / ta + 1 / te
bu = 1 / (ta * te * I)

timestep = 0.01
# Task dynamics
Ac = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -a1, -a2, -a3]])
Bc = numpy.array([[0, 0, 0, bu]]).reshape((-1, 1))

# Task noise
F = numpy.diag([0, 0, 0, 0.001])
G = 0.03 * numpy.diag([1, 1, 0, 0])

# Determinstic Observation Filter
C = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

# Motor and observation noise
Gamma = numpy.array(0.08)
D = numpy.array([[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.05, 0], [0, 0, 0, 0]])

# Cost matrices
Q = numpy.diag([1, 0.01, 0, 0])
R = numpy.array([[1e-3]])
U = numpy.diag([1, 0.1, 0.01, 0])

task = ClassicControlTask(
    timestep,
    Ac,
    Bc,
    F=F,
    G=G,
    discrete_dynamics=False,
    noise="on",
    timespace="continuous",
)
user = IHCT_LQGController("user", timestep, Q, R, U, C, Gamma, D, noise="on")
bundle = Bundle(task=task, user=user, onreset_deterministic_first_half_step=True)
obs = bundle.reset(
    turn=1,
    dic={
        "task_state": {"x": numpy.array([[0.5], [0], [0], [0]])},
        "user_state": {"xhat": numpy.array([[0.5], [0], [0], [0]])},
    },
)
bundle.playspeed = 0.001
bundle.render("plot")
for i in range(250):
    obs, rewards, is_done, reward_list = bundle.step()
    print(reward_list)
    print(i)
    if not i % 5:
        bundle.render("plot")
