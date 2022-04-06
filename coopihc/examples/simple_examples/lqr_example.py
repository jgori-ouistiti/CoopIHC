from coopihc.interactiontask.ClassicControlTask import ClassicControlTask
from coopihc.agents.lqrcontrollers.IHDT_LQRController import IHDT_LQRController
from coopihc.bundle.Bundle import Bundle


import numpy

m, d, k = 1, 1.2, 3
Q = numpy.array([[1, 0], [0, 0]])
R = 1e-4 * numpy.array([[1]])

Ac = numpy.array([[0, 1], [-k / m, -d / m]])

Bc = numpy.array([0, 1]).reshape(2, 1)


task = ClassicControlTask(0.002, Ac, Bc, discrete_dynamics=False)
user = IHDT_LQRController("user", Q, R)
bundle = Bundle(task=task, user=user)
bundle.reset(go_to=0)
bundle.playspeed = 0.01
# bundle.render("plot")
for i in range(1500):
    state, reward, is_done = bundle.step()
    # print(state)
    # if not i % 5:
    # bundle.render("plot")
