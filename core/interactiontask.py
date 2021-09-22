from abc import ABC, abstractmethod
from collections import OrderedDict
import gym
import numpy
from core.helpers import flatten
import scipy.linalg
from core.space import State, StateElement
import core.space
from core.core import Core, Handbook

import copy
import sys
from loguru import logger

import asyncio
import json
import websockets

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
        self._state = State()
        # self._state._on_state_update = self._on_state_update
        self.bundle = None
        self.round = 0
        self.turn = 0
        self.timestep = 0.1

        # Render stuff
        self.ax = None
        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})
        logger.info('Initializing task {}'.format(self.__class__.__name__))




    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def operator_action(self):
        if self.bundle:
            return self.bundle.game_state['operator_action']['action']

    @property
    def assistant_action(self):
        if self.bundle:
            return self.bundle.game_state['assistant_action']['action']


    def __content__(self):
        return {'Name': self.__class__.__name__, "State": self.state.__content__()}

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
            self.state.reset(dic = {})
            return

        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value['values']
            if value is not None:
                self.state[key]['values'] = value



class PipeTaskWrapper(InteractionTask, ABC):
    def __init__(self, task, pipe):
        self.__dict__ = task.__dict__
        self.task = task
        self.pipe = pipe
        self.pipe.send({'type': 'init', 'parameters': self.parameters})
        is_done = False
        while True:
            self.pipe.poll(None)
            received_state = self.pipe.recv()
            # This assumes that the final message sent by the client is a task_state message. Below should be changed to remove that assumption (i.e. client can send whatever order)
            if received_state['type'] == 'task_state':
                is_done = True
            self.update_state(received_state)
            if is_done:
                break

    def __getattr__(self, attr):
        if self.__dict__:
            return getattr(self.__dict__['task'], attr)
        else:
            # should never happen
            pass

    def __setattr__(self, name, value):
        if name == '__dict__' or name == 'task':
            super().__setattr__(name, value)
            return
        if self.__dict__:
            setattr(self.__dict__['task'], name, value)


    def update_state(self, state):
        print("updating state")
        if state['type'] == 'task_state':
            del state['type']
            self.update_task_state(state)
        elif state['type'] == 'operator_state':
            del state['type']
            self.update_operator_state(state)


    @abstractmethod
    def update_task_state(self, state):
        pass

    @abstractmethod
    def update_operator_state(self, state):
        pass


    def operator_step(self,  *args, **kwargs):
        super().operator_step(*args, **kwargs)
        operator_action_msg = { 'type': "operator_action", 'value': self.bundle.game_state["operator_action"]['action'].serialize()}
        self.pipe.send(operator_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic['state']
        self.update_state(received_state)
        return self.state, received_dic['reward'], received_dic['is_done'], {}

    def assistant_step(self,  *args, **kwargs):
        super().assistant_step(*args, **kwargs)
        assistant_action_msg = { 'type': "assistant_action", 'value': self.bundle.game_state["assistant_action"]['action'].serialize()}
        self.pipe.send(assistant_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic['state']
        self.update_state(received_state)
        return self.state, received_dic['reward'], received_dic['is_done'], {}


    def reset(self, dic = None):
        super().reset(dic = dic)
        reset_msg = {'type':'reset', 'reset_dic': dic}
        self.pipe.send(reset_msg)
        self.pipe.poll(None)
        received_state = self.pipe.recv()
        self.update_state(received_state)
        self.bundle.reset(task = False)
        return self.state


class ClassicControlTask(InteractionTask):
    """ verify F and G conversions.
    """
    def __init__(self, timestep, A, B, F = None, G = None, discrete_dynamics = True, noise = 'on'):
        super().__init__()


        self.handbook['render_mode'].extend(['plot', 'text'])
        _timestep = { "name": 'timestep', "value": timestep, 'meaning': 'Duration of one step in this environment'}

        _A = {"name": 'A', "value": A, "meaning": 'Uncontrolled dynamics (x(t+1) += A x(t))'}
        _B = {"name": 'B', "value": B, "meaning": 'Controlled dynamics (x(t+1) += B u(t))'}
        _F = { "name": 'F', "value": F, 'meaning': 'signal dependent noise matrix (x(t+1) += F x(t) * N(.,.) )'}
        _G = { "name": 'G', "value": G, 'meaning': 'independent noise matrix (x(t+1) += G * N(.,.))'}
        _discrete_dynamics = { "name": 'discrete_dynamics', "value": discrete_dynamics, 'meaning': 'Whether or not the values in A, B, F, G are for continuous or discrete systems. Continuous matrices are cast to discrete matrices.'}
        _noise = { "name": 'noise','value': noise, 'meaning': "whether or not noise is present in the system"}
        self.params = [_timestep, _A, _B, _F, _G, _discrete_dynamics,_noise]
        self.handbook['parameters'].extend(self.params)

        self.handbook['operator_constraints'] = []
        _op = {'attr': 'timespace', 'values': ['discrete', 'continuous'], 'meaning': 'The operator should have a timespace attribute which is either discrete or continuous'}





        self.dim = max(A.shape)
        self.state = State()
        self.state['x'] = StateElement(   values = numpy.zeros((self.dim, 1)),
                            spaces = gym.spaces.Box(low = -numpy.ones((self.dim, 1))*numpy.inf, high = numpy.ones((self.dim,1))*numpy.inf, shape = (self.dim,1 )),
                            possible_values = [[None]]
                            )
        self.state_last_x = copy.copy(self.state['x']['values'])
        self.timestep = timestep

        if F is None:
            self.F = numpy.zeros(A.shape)
        else:
            self.F = F
        if G is None:
            self.G = numpy.zeros(A.shape)
        else:
            self.G = G
        # Convert dynamics between discrete and continuous.
        if discrete_dynamics:
            self.A_d = A
            self.B_d = B
            # Euler method
            self.A_c = 1/timestep*(A-numpy.eye(A.shape[0]))
            self.B_c = B/timestep
        else:
            self.A_c = A
            self.B_c = B
            # Euler Method
            self.A_d = numpy.eye(A.shape[0]) + timestep*A
            self.B_d = timestep*B

        self.noise = noise

    def finit(self):
        if self.bundle.operator.timespace == 'continuous':
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic = None):
        super().reset()
        # Force zero velocity
        self.state['x']['values'][0][1:] = 0

        if dic is not None:
            super().reset(dic = dic)

        self.state_last_x = copy.copy(self.state['x']['values'])


    def operator_step(self, operator_action):
        # print(operator_action, type(operator_action))
        is_done = False
        # Call super for counters
        super().operator_step(operator_action)

        # For readability
        A, B, F, G = self.A, self.B, self.F, self.G
        u = operator_action['values'][0]
        x = self.state['x']['values'][0].reshape(-1,1)

        # Generate noise samples
        if self.noise == 'on':
            beta, gamma = numpy.random.normal(0, numpy.sqrt(self.timestep), (2,1))
            omega = numpy.random.normal(0, numpy.sqrt(self.timestep),(self.dim,1))
        else:
            beta, gamma = numpy.random.normal(0, 0, (2,1))
            omega = numpy.random.normal(0, 0,(self.dim,1))


        # Store last_x for render
        self.state_last_x = copy.deepcopy(self.state['x']['values'])
        # Deterministic update + State dependent noise + independent noise
        if self.bundle.operator.timespace == 'discrete':
            x = (A@x + B*u) + F@x*beta + G@omega
        else:
            x += (A@x + B*u)*self.timestep + F@x*beta + G@omega


        self.state['x']['values'] = x
        if abs(x[0,0]) <= 0.01:
            is_done = True

        return self.state, 0, is_done, {}

    def assistant_step(self, assistant_action):
        # return super().assistant_step(assistant_action)
        return self.state, 0, False, {}

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'


        if 'text' in mode:
            print('state')
            print(self.state['x'])
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
        if (self.state_last_x[0]== self.state["x"]['values'][0]).all():
            pass
        else:
            for i in range(self.dim):
                self.axes[i].plot([(self.turn-1)*self.timestep, self.turn*self.timestep], flatten([self.state_last_x[0][i,0].tolist(), self.state['x']['values'][0][i,0].tolist()]), '-', color = self.color[i], label = self.labels[i])

        return



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
