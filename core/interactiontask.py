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
        self.params = []
        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': self.params})
        logger.info('Initializing task {}'.format(self.__class__.__name__))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


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


class WebsocketInteractionTask(InteractionTask):
    def __init__(self):
        pass
    # def __init__(self, pathname = 'localhost', portnumber = 4000):
    #     self.users = set()
    #     await self.register(websocket)
    #     super().__init__()
    #
    #
    # async def register(websocket):
    #     self.users.add(websocket)
    #     await self.notify_users()
    #
    # @property
    # def state(self):
    #     return self._state
    #
    # @state.setter
    # def state(self, value):
    #     self._state = value
    #     print("sending {}".format(json.dumps.serialize()))
    #     await asyncio.wait([user.send(json.dumps(value.serialize())) for user in self.users])



class MyWebsocket:
    async def __aenter__(self, pathname = 'localhost', portnumber = '4000'):
        self._conn = websockets.connect('ws://{}:{}'.format(pathname, portnumber))
        self.websocket = await self._conn.__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self._conn.__aexit__(*args, **kwargs)

    async def send(self, message):
        await self.websocket.send(message)

    async def receive(self):
        return await self.websocket.recv()

class WebsocketWrapper(InteractionTask):
    def __init__(self, task):
        # self.__dict__['task'] = task # get around the __setattr__ override
        self.__dict__['task'] = task
        self.ws = MyWebsocket()
        self.loop = asyncio.get_event_loop()


    # @property
    # def state(self):
    #     print('accessing')
    #     return self.task._state


    def send_state_decorator(func):
        def function_wrapper(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            serialized_state = json.dumps(self.state.serialize())
            print("in {}, sending {}".format(func.__name__, serialized_state))
            self.loop.run_until_complete(self.__async__send_state(serialized_state))
            return ret
        return function_wrapper

    async def __async__send_state(self, serialized_state):
        async with self.ws as ws:
            await ws.send(serialized_state)
            # return await ws.receive()

    def __getattr__(self, attr):
        return getattr(self.__dict__['task'], attr)

    def __setattr__(self, name, value):
        # handle data descriptor invocation (i.e. the state property), see https://stackoverflow.com/questions/15750522/class-properties-and-setattr
        # if isinstance(getattr(type(self), name, None), property):
        #     print('here')
        #     # super().__setattr__(name, value)
        #     self._state_setter(name, value)
        # else:
        #     setattr(self.__dict__['task'], name, value)
        setattr(self.__dict__['task'], name, value)


    def __content__(self):
        return self.task.__content__()

    @send_state_decorator
    def finit(self):
        self.task.finit()

    def _is_done_operator(self, *args, **kwargs):
        return self.task._is_done_operator(*args, **kwargs)

    def _is_done_assistant(self, *args, **kwargs):
        return self.task._is_done_assistant(*args, **kwargs)

    @send_state_decorator
    def operator_step(self,  *args, **kwargs):
        return self.task.operator_step(*args, **kwargs)

    @send_state_decorator
    def assistant_step(self,  *args, **kwargs):
        return self.task.assistant_step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.task.render(*args, **kwargs)

    @send_state_decorator
    def reset(self, dic = None):
        return self.task.reset(dic = dic)


class WebSocketTask(InteractionTask):

    def __init__(self):
        super().__init__()
        self.users = set()

    def state_event(self):
        # return json.dumps({"type": "state", 'state': self.state.serialize()})
        return json.dumps({"type": "state"})

    def users_event(self):
        return json.dumps({"type": "users", "count": len(self.users)})

    def update_state(self, JSON_stringified_state):
        state = dict(JSON_stringified_state)
        for element in self.state:
            try:
                value = state.pop(element)
                self.state[element]['value'] = value
            except KeyError('Key {} defined in task state was not found in the received data'):
                # trigger error regardless
                value = state.pop(element)
        if state:
            print("warning: the received data has not been consumed. {} does not match any current task state".format(str(state)))




    async def _init(self):
        if self.users:
            await asyncio.wait([user.send(json.dumps({'type': 'init', 'params': self.params})) for user in self.users])
            # select first user (i.e. only dealing with a single user case for now)
            user = next(iter(self.users))
            JSON_stringified_init_state = await user.recv()
            self.update_state(JSON_stringified_init_state)

    async def notify_state(self):
        if self.users:  # asyncio.wait doesn't accept an empty list
            message = self.state_event()
            await asyncio.wait([user.send(message) for user in self.users])


    async def notify_users(self):
        if self.users:  # asyncio.wait doesn't accept an empty list
            message = self.users_event()
            await asyncio.wait([user.send(message) for user in self.users])


    async def register(self, websocket):
        self.users.add(websocket)
        print("added user")
        await self.notify_users()


    async def unregister(self, websocket):
        self.users.remove(websocket)
        print('removed user')
        await self.notify_users()


    async def interact(self, websocket, path):
        # register(websocket) sends user_event() to websocket
        await self.register(websocket)
        await self._init()
        try:
            await websocket.send(self.state_event())
            # await self.send_state({'Position': 2})
            async for message in websocket:
                data = json.loads(message)
                print(data)
                if data["action"] == "minus":
                    self.state["value"] -= 1
                    await self.notify_state()
                elif data["action"] == "plus":
                    self.state["value"] += 1
                    await self.notify_state()
                else:
                    logging.error("unsupported event: %s", data)
        finally:
            await self.unregister(websocket)

    async def send_state(self, state):
        if self.users:
            print('sending state')
            await asyncio.wait([user.send(json.dumps({'type': 'state', **state})) for user in self.users])

# class WebSocketTask(InteractionTask):
#     def __init__(self):
#         super().__init__()
#         self.ws = MyWebsocket()
#
#     async def _init(self):
#         await self.__async__send_msg(json.dumps({'init': None}))
#
#     async def __async__send_msg(self, msg):
#         async with self.ws as ws:
#             await ws.send(msg)
#             print('sent ' + msg)
#
async def create_websocket_task(*args, **kwargs):
    task = WebSocketTask(*args, **kwargs)
    await task._init()
    print("finished waiting")
    return task

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

        return self.state, -1/2, is_done, {}

    def assistant_step(self, assistant_action):
        return super().assistant_step(assistant_action)

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
