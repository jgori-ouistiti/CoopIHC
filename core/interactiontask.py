from abc import ABC, abstractmethod
from collections import OrderedDict
import gym
import numpy
from core.helpers import flatten
import scipy.linalg
from core.space import State, StateElement
import core.space
from core.core import Core, Handbook


import sys
from loguru import logger
# logger.add('interaction_agent.log', format = "{time} {level} {message}")
# logger.add(sys.stderr, colorize = True, format="<green>{time}</green> <level>{message}</level>")



class InteractionTask(Core):
    """ The class that defines an Interaction Task. Subclass this to define any new task. When doing so, make sure to call ``super()`` in ``__init__()``.

    The main API methods for this class are:

        __init__

        finit

        reset

        operator_step

        assistant_step

        render

    :meta public:
    """


    def __init__(self):
        # self._state = TestState('task_state')
        self.state = State()
        # self._state._on_state_update = self._on_state_update
        self.bundle = None
        self.round = 0
        self.turn = 0
        # Render stuff
        self.ax = None

        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})
        logger.info('Initializing task {}'.format(self.__class__.__name__))

    def finit(self):
        logger.info(str(self.handbook))


    def _is_done_operator(self, *args, **kwargs):
        logger.info('{} is done on operator step: False (Inherited from InteractionTask)'.format(self.__class__.__name__))
        return False

    def _is_done_assistant(self, *args, **kwargs):
        logger.info('{} is done on assistant step: False (Inherited from InteractionTask)'.format(self.__class__.__name__))
        return False

    def operator_step(self,  *args, **kwargs):
        """ Describe how the task state evolves after an operator action. This method has to be redefined when subclassing this class.

        :param operator_action: (list) operator action

        :return: state, reward, is_done, metadata: state (OrderedDict) of the task, reward (float) associated with the step, is_done (bool) flags whether the task is finished, metadata (dictionnary) for compatibility with gym environments.

        :meta public:
        """
        self.turn += 1/2
        return self.state, -1/2, self._is_done_operator(), {}

    def assistant_step(self,  *args, **kwargs):
        """ Describe how the task state evolves after an assistant action. This method has to be redefined when subclassing this class.

        :param operator_action: (list) assistant action

        :return: state, reward, is_done, metadata: state (OrderedDict) of the task, reward (float) associated with the step, is_done (bool) flags whether the task is finished, metadata (dictionnary) for compatibility with gym environments.

        :meta public:
        """
        self.turn += 1/2
        self.round += 1
        return self.state, -1/2, self._is_done_assistant(), {}

    def render(self, *args, **kwargs):
        """ Render the task on the main plot.

        :param mode: (str) text or plot
        :param args: (list) list of axis in order axtask, axoperator, axassistant

        """
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'

        if 'text' in mode:
            print(self.state)
        else:
            pass

    def reset(self, dic = None):
        """ Describe how the task state should be reset. This method has to be redefined when subclassing this class.

        :param args: (OrderedDict) state to which the task should be reset, if provided.

        :return: state (OrderedDict) of the task.

        :meta public:
        """
        self.turn = 0
        self.round = 0
        if not dic:
            return

        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value['values']
            if value is not None:
                self.state[key]['values'] = value



class TaskWrapper(InteractionTask):
    def __init__(self, task, *args, **kwargs):
        self.task = task
        self.__dict__.update(task.__dict__)

    def operator_step(self, *args, **kwargs):
        return self.task.operator_step(*args, **kwargs)

    def assistant_step(self, *args, **kwargs):
        return self.task.assistant_step(*args, **kwargs)

    def reset(self, dic = None):
        return self.task.reset(dic = dic)

    def render(self, *args, **kwargs):
        return self.task.render(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.task.unwrapped


class ClassicControlTask(InteractionTask):
    """ use of F and G not supported yet in the discrete case --> Need to verify the actual conversions.
    """
    def __init__(self, timestep, A, B, F = None, G = None, discrete_dynamics = True, noise = 'on'):
        super().__init__()
        self.dim = A.shape[1]
        x = numpy.array([0 for i in range(self.dim)])
        self.state = State()
        self.state['x'] = [x, [core.space.Array(x)], None]
        self.timestep = timestep

        # Convert dynamics between discrete and continuous.
        if discrete_dynamics:
            self.A_d = A
            self.B_d = B
            # Euleur method
            self.A_c = 1/timestep*(A-numpy.eye(A.shape[0]))
            self.B_c = B/timestep
        else:
            self.A_c = A
            self.B_c = B
            # Euler Method
            self.A_d = numpy.eye(A.shape[0]) + timestep*A
            self.B_d = timestep*B

        if F is None:
            F = numpy.zeros(A.shape)
        if G is None:
            G = numpy.zeros(A.shape)

        self.F = F
        self.G = G
        self.noise = noise

    def finit(self):
        if self.bundle.operator.timespace == 'continuous':
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic = None):
        self.state.reset()
        # Force zero velocity
        self.state['x'][0][1] = 0
        # Call super().reset after to force states if needed
        super().reset(dic)
        self.last_x = self.state['_value_x'].copy()

    def operator_step(self, operator_action):
        # print(operator_action, type(operator_action))
        is_done = False
        # Call super for counters
        super().operator_step(operator_action)

        # For readability
        A, B, F, G = self.A, self.B, self.F, self.G
        u = operator_action[0]
        x = self.state['_value_x'].reshape(-1,1)

        # Generate noise samples
        if self.noise == 'on':
            beta, gamma = numpy.random.normal(0, numpy.sqrt(self.timestep), (2,))
            omega = numpy.random.normal(0, numpy.sqrt(self.timestep),(self.dim,1))
        else:
            beta, gamma = numpy.random.normal(0, 0, (2,))
            omega = numpy.random.normal(0, 0,(self.dim,1))


        # Store last_x for render
        self.last_x = x.copy()
        # Deterministic update + State dependent noise + independent noise
        if self.bundle.operator.timespace == 'discrete':
            x = (A@x + B*u) + F@x*beta + G@omega
        else:
            x += (A@x + B*u)*self.timestep + F@x*beta + G@omega

        self.state['_value_x'] = x
        if abs(x[0,0]) <= 0.01:
            is_done = True

        return self.state, -1/2, is_done, {}

    def assistant_step(self, assistant_action):
        return super().assistant_step(assistant_action)

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'


        if 'text' in mode:
            print('state')
            print(self.state['_value_x'])
        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.ax is not None:
                self.draw()
                if self.turn == 1:
                    self.ax.legend(handles = [self.axes[i].lines[0] for i in range(self.dim)])
            else:
                self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                self.labels = ['x[{:d}]'.format(i) for i in range(self.dim)]
                self.axes = [axtask]
                self.ax = axtask
                for i in range(self.dim-1):
                    self.axes.append(self.ax.twinx())

                for i,ax in enumerate(self.axes):
                    # ax.yaxis.label.set_color(self.color[i])
                    ax.tick_params(axis='y', colors = self.color[i])

                self.draw()


    def draw(self):
        if (self.last_x == self.state["_value_x"]).all():
            pass
        else:
            for i in range(self.dim):
                self.axes[i].plot([(self.turn-1)*self.timestep, self.turn*self.timestep], flatten([self.last_x[i,0].tolist(), self.state['_value_x'][i,0].tolist()]), '-', color = self.color[i], label = self.labels[i])

        return
