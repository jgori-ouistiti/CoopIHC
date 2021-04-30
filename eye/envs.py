import numpy
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
import gym


from core.interactiontask import InteractionTask
import eye.noise


class ChenEyePointingTask(InteractionTask):
    """ A pointing task performed by the Eye, according to Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021. This tasks only requires an operator (human perception task).

    :param fitts_W: (float) size of the target aimed at
    :param fitts_D: (float) distance of the target aimed at
    :param ocular_std: (float) magnitude of the oculo-motor noise

    :meta public:
    """
    def __init__(self, fitts_W, fitts_D, ocular_std):
        super().__init__()
        self.fitts_W = fitts_W
        self.fitts_D = fitts_D
        self.ocular_std = ocular_std
        self.state = OrderedDict({'Targets': [[0,0]]})

    def get_new_target(self, D):
        """ Generates a target at a random angle, distance D away.

        :param D: distance of the target

        :return: (numpy.ndarray) the target that the eye is going to localize.

        :meta public:
        """
        angle=numpy.random.uniform(0,math.pi*2)
        x_target=math.cos(angle)*D
        y_target=math.sin(angle)*D
        return numpy.array([x_target,y_target])

    def reset(self, *args):
        """ Reset the task.

        Pick out a new target at random orientation.

        :meta public:
        """
        if args:
            raise NotImplementedError
        else:
            self.state['Targets'] = [self.get_new_target(self.fitts_D)]





    def operator_step(self, operator_action):
        """ Use the difference between the old and new fixation to generate a covariance matrix for noise. Then, sample a 2D Gaussian with the corresponding covariance matrix.

        If the new fixation is inside the target, then stop.

        :param operator_action: the new fixation

        :return: new task state, reward, is_done, {}

        :meta public:
        """
        self.turn += 1/2
        fixation = numpy.array(self.bundle.operator.state['Fixation'])
        action = numpy.array(operator_action)
        cov = eye.noise.eccentric_noise(action, fixation, self.ocular_std)
        ocular_noise = numpy.random.multivariate_normal( numpy.zeros(shape = (2,)), cov)
        new_fixation = numpy.clip(action + ocular_noise, -1, 1)
        self.bundle.operator.state['Fixation'] = new_fixation

        if numpy.sqrt(numpy.sum((new_fixation - self.state['Targets'][0])**2)) <= self.fitts_W/2:
            is_done = True
            reward = 0
        else:
            is_done = False
            reward = -1

        return self.state, reward, is_done, {}



    def assistant_step(self):
        """ Do nothing but incrementing turns and rounds.

        :meta public:
        """
        self.turn += 1/2
        self.round += 1
        is_done = False
        return self.state, 0, is_done, {}

    def render(self, ax, *args, mode="text"):
        """ Render the task.

        In plot mode plots the fixations on axtask. In text mode prints turns and goal.

        :meta public:
        """
        goal = self.state['Targets'][0].tolist()
        if 'text' in mode:
            print('\n')
            print("Turn number {:f}".format(self.turn))
            print('Target:')
            print(goal)

        if 'plot' in mode:
            if self.ax is not None:
                traj = self.ax.plot(*self.bundle.operator.state['Fixation'], 'og')
                pgoal = self.ax.plot(*goal, 'ro')
            else:
                self.ax = ax
                self.ax.set_xlim([-1.3,1.3])
                self.ax.set_ylim([-1.3,1.3])
                self.ax.set_aspect('equal')
                pgoal = self.ax.plot(*goal, 'ro')
                traj = self.ax.plot(*self.bundle.operator.state['Fixation'], 'og')
        if not ('plot' in mode or 'text' in mode):
            raise NotImplementedError
