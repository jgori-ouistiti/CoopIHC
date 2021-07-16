import numpy
import math
import matplotlib.pyplot as plt
import gym


from core.interactiontask import InteractionTask
from core.space import State, StateElement
import eye.noise

from loguru import logger

class ChenEyePointingTask(InteractionTask):
    """ A pointing task performed by the Eye, adapted from Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021. This tasks only requires an operator (human perception task).



    :param fitts_W: (float) size of the target aimed at
    :param fitts_D: (float) distance of the target aimed at
    :param threshold: (float) The distance between target and fixation is compared to threshold to determine if the task is done. The default value is based on a calculation where the foval vision has a 5° angle opening and the eye is half a meter away from the screen.
    :param dimension: (int) Whether the task is one or two dimensional.

    :meta public:
    """
    # threshold => d = L alpha with L = 0.5m and alpha = 5°
    def __init__(self, fitts_W, fitts_D, threshold = 0.04, dimension = 2):
        super().__init__()

        self.threshold = threshold
        self.fitts_W = fitts_W
        self.fitts_D = fitts_D
        self.dimension = dimension
        self.parity = 1


        self.handbook['render_mode'].extend(['plot', 'text'])
        _threshold = {"name": 'threshold', "value": threshold, "meaning": 'If the distance between the fixation and the target is less or equal than threshold, the environment will return True'}
        _fitts_W = { "name": 'W', "value": fitts_W, 'meaning': 'radius of the circular target used'}
        _fitts_D = { "name": 'D','value': fitts_D, 'meaning': 'distance between targets'}
        _dimension = {'name': 'dimension', 'value': dimension, 'meaning': 'Dimension of the task used. In 1D the targets appear alternatively left or right, while in 2D the targets appear at a random angle and fixed radius.'}
        self.params = [_threshold, _fitts_W, _dimension]
        self.handbook['parameters'].extend(self.params)


        self.state['Targets'] = StateElement(
            values = [numpy.array([0 for i in range(dimension)])],
            spaces = [gym.spaces.Box(-1,1, shape = (dimension,))],
            possible_values = [None],
            mode = 'warn'
        )
        self.state["Fixation"] = StateElement(
            values = [numpy.array([0 for i in range(dimension)])],
            spaces = [gym.spaces.Box(-1,1, shape = (dimension,))],
            possible_values = [None],
            mode = 'warn')



    def get_new_target(self, D):
        """ Generates a target, D/2 away from the center.

        If the task is two dimensional, the angle is uniformly chosen in [0, 2Pi]. If the task is one dimensional, the angle is uniformly chosen in {0, Pi}.

        :param D: distance of the target

        :return: (numpy.ndarray) the target that the eye is going to localize.

        :meta public:
        """
        if self.dimension == 2:
            angle=numpy.random.uniform(0,math.pi*2)
            # d = numpy.random.uniform(-D/2,D/2)
            d = D/2
            x_target=math.cos(angle)*d
            y_target=math.sin(angle)*d
            return numpy.array([x_target,y_target])
        elif self.dimension == 1:
            self.parity = (self.parity +1)%2
            # angle = self.parity*math.pi
            angle = numpy.random.binomial(1, 1/2)*math.pi
            d = numpy.random.uniform(-D/2,D/2)
            x_target=math.cos(angle)*d
            return numpy.array([x_target])
        else:
            raise NotImplementedError



    def reset(self, dic = None):
        """ Reset the task.

        Pick out a new target at random orientation.

        :meta public:
        """

        self.state['Targets']['values'] = [self.get_new_target(self.fitts_D)]
        self.state['Fixation']['values'] = [numpy.array([0 for i in range(self.dimension)])]
        super().reset(dic)



    def _is_done_operator(self):
        if numpy.sqrt(numpy.sum((self.state['Fixation']['values'] - self.state['Targets']['values'][0])**2)) - self.fitts_W/2 < self.threshold:
            is_done = True
        else:
            is_done = False
        logger.info("Task {} done: {}".format(self.__class__.__name__, is_done))
        logger.info('Condition: distance(Fixation,Targets) < threshold + half target width ---- {} < {} + {}'.format(str(numpy.sqrt(numpy.sum((self.state['Fixation']['values'] - self.state['Targets']['values'][0])**2))), str(self.threshold), str(self.fitts_W/2)  ))
        return is_done

    def operator_step(self, operator_action):
        """ Use the difference between the old and new fixation to generate a covariance matrix for noise. Then, sample a 2D Gaussian with the corresponding covariance matrix.

        If the new fixation is inside the target, then stop.

        :param operator_action: the new fixation

        :return: new task state, reward, is_done, {}

        :meta public:
        """
        self.turn += 1/2
        action = operator_action['values'][0]
        self.state['Fixation']['values'] = action

        reward = -1

        return self.state, reward, self._is_done_operator(), {}





    def assistant_step(self, assistant_action):
        """ Do nothing but incrementing turns and rounds.

        :meta public:
        """
        self.turn += 1/2
        self.round += 1
        is_done = False
        return self.state, 0, self._is_done_assistant(), {}

    def render(self, *args, mode="text", **kwargs):
        """ Render the task.

        In plot mode plots the fixations on axtask. In text mode prints turns and goal.

        :meta public:
        """
        goal = self.state['Targets']['values'][0].tolist()
        fx = self.state['Fixation']['values'][0].tolist()

        if 'text' in mode:
            print('\n')
            print("Turn number {:f}".format(self.turn))
            print('Target:')
            print(goal)
            print('Fixation:')
            print(fx)

        if 'plot' in mode:
            try:
                axtask, axoperator, axassistant = args[:3]
            except ValueError:
                raise ValueError('You have to provide the three axes (task, operator, assistant) to render in plot mode.')
            if self.ax is not None:
                pass
            else:
                self.ax = axtask
                self.ax.set_xlim([-1.3,1.3])
                self.ax.set_ylim([-1.3,1.3])
                self.ax.set_aspect('equal')

                axoperator.set_xlim([-1.3,1.3])
                axoperator.set_ylim([-1.3,1.3])
                axoperator.set_aspect('equal')


            if self.dimension == 1:
                goal = [goal[0], 0]
                fx = [fx[0], 0]
            pgoal = self.ax.plot(*goal, 'ro')
            traj = self.ax.plot(*fx, 'og')
        if not ('plot' in mode or 'text' in mode):
            raise NotImplementedError
