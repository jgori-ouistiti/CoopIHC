import numpy
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
import gym
from core.interactiontask import InteractionTask

class ChenEyePointingTask(InteractionTask):
    def __init__(self, fitts_W, fitts_D, ocular_std):
        super().__init__()
        self.fitts_W = fitts_W
        self.fitts_D = fitts_D
        self.ocular_std = ocular_std
        self.state = OrderedDict({'Targets': [[0,0]]})

    def get_new_target(self, D):
        '''
        generate a target at a random angle, distance D away.
        '''
        angle=numpy.random.uniform(0,math.pi*2)
        x_target=math.cos(angle)*D
        y_target=math.sin(angle)*D
        return numpy.array([x_target,y_target])

    def reset(self, *args):
        if args:
            raise NotImplementedError
        else:
            self.state['Targets'] = [self.get_new_target(self.fitts_D)]


    def operator_step(self, operator_action):
        self.turn += 1/2
        fixation = numpy.array(self.bundle.operator.state['Fixation'])
        action = numpy.array(operator_action)
        distance = numpy.sqrt(numpy.sum((fixation - action)**2))
        ocular_noise = numpy.random.normal(0, self.ocular_std * distance, action.shape)
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
        self.turn += 1/2
        self.round += 1
        is_done = False
        return self.state, 0, is_done, {}

    def render(self, ax, *args, mode="text"):
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
