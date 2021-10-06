from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy

from core.helpers import flatten
import scipy.linalg
from core.space import State, StateElement
import core.space
from core.core import Core, Handbook

import copy
import sys

import asyncio
import json
import websockets


# updated 04-10-21


class InteractionTask(Core, ABC):
    """The class that defines an Interaction Task. Subclass this to define any new task. When doing so, make sure to call ``super()`` in ``__init__()``.

    The main API methods for this class are:

        __init__

        finit

        reset

        user_step

        assistant_step

        render

    :meta public:
    """

    def __init__(self):
        self._state = State()
        self.bundle = None
        self.round = 0
        self.timestep = 0.1

        # Render
        self.ax = None

    def finit(self):
        return

    @property
    def turn_number(self):
        return self.bundle.turn_number

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def user_action(self):
        if self.bundle:
            return self.bundle.game_state["user_action"]["action"]

    @property
    def assistant_action(self):
        if self.bundle:
            return self.bundle.game_state["assistant_action"]["action"]

    def __content__(self):
        return {
            "Name": self.__class__.__name__,
            "State": self.state.__content__(),
        }

    @abstractmethod
    def user_step(self, *args, **kwargs):
        return None

    @abstractmethod
    def assistant_step(self, *args, **kwargs):
        return None

    @abstractmethod
    def reset(self, dic=None):
        return None

    def base_user_step(self, *args, **kwargs):
        """Describe how the task state evolves after an user action. This method has to be redefined when subclassing this class.

        :param user_action: (list) user action

        :return: state, reward, is_done, metadata: state (OrderedDict) of the task, reward (float) associated with the step, is_done (bool) flags whether the task is finished, metadata (dictionnary) for compatibility with gym environments.

        :meta public:
        """
        ret = self.user_step(*args, **kwargs)
        if ret is None:
            return self.state, -1 / 2, False, {}
        else:
            return ret

    def base_assistant_step(self, *args, **kwargs):
        """Describe how the task state evolves after an assistant action. This method has to be redefined when subclassing this class.

        :param user_action: (list) assistant action

        :return: state, reward, is_done, metadata: state (OrderedDict) of the task, reward (float) associated with the step, is_done (bool) flags whether the task is finished, metadata (dictionnary) for compatibility with gym environments.

        :meta public:
        """
        self.round += 1

        ret = self.assistant_step(*args, **kwargs)
        if ret is None:
            return self.state, -1 / 2, False, {}
        else:
            return ret

    def render(self, *args, **kwargs):
        """Render the task on the main plot.

        :param mode: (str) text or plot
        :param args: (list) list of axis in order axtask, axuser, axassistant

        """
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "text" in mode:
            print(self.state)
        else:
            pass

    def base_reset(self, dic=None):
        """Describe how the task state should be reset. This method has to be redefined when subclassing this class.

        :param args: (OrderedDict) state to which the task should be reset, if provided.

        :return: state (OrderedDict) of the task.

        :meta public:
        """
        self.round = 0

        if not dic:
            self.state.reset(dic={})
            self.reset(dic=dic)
            return

        self.reset(dic=dic)
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value["values"]
            if value is not None:
                self.state[key]["values"] = value


# Outdated


class PipeTaskWrapper(InteractionTask, ABC):
    def __init__(self, task, pipe):
        self.__dict__ = task.__dict__
        self.task = task
        self.pipe = pipe
        self.pipe.send({"type": "init", "parameters": self.parameters})
        is_done = False
        while True:
            self.pipe.poll(None)
            received_state = self.pipe.recv()
            # This assumes that the final message sent by the client is a task_state message. Below should be changed to remove that assumption (i.e. client can send whatever order)
            if received_state["type"] == "task_state":
                is_done = True
            self.update_state(received_state)
            if is_done:
                break

    def __getattr__(self, attr):
        if self.__dict__:
            return getattr(self.__dict__["task"], attr)
        else:
            # should never happen
            pass

    def __setattr__(self, name, value):
        if name == "__dict__" or name == "task":
            super().__setattr__(name, value)
            return
        if self.__dict__:
            setattr(self.__dict__["task"], name, value)

    def update_state(self, state):
        print("updating state")
        if state["type"] == "task_state":
            del state["type"]
            self.update_task_state(state)
        elif state["type"] == "user_state":
            del state["type"]
            self.update_user_state(state)

    @abstractmethod
    def update_task_state(self, state):
        pass

    @abstractmethod
    def update_user_state(self, state):
        pass

    def user_step(self, *args, **kwargs):
        super().user_step(*args, **kwargs)
        user_action_msg = {
            "type": "user_action",
            "value": self.bundle.game_state["user_action"][
                "action"
            ].serialize(),
        }
        self.pipe.send(user_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic["state"]
        self.update_state(received_state)
        return self.state, received_dic["reward"], received_dic["is_done"], {}

    def assistant_step(self, *args, **kwargs):
        super().assistant_step(*args, **kwargs)
        assistant_action_msg = {
            "type": "assistant_action",
            "value": self.bundle.game_state["assistant_action"][
                "action"
            ].serialize(),
        }
        self.pipe.send(assistant_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic["state"]
        self.update_state(received_state)
        return self.state, received_dic["reward"], received_dic["is_done"], {}

    def reset(self, dic=None):
        super().reset(dic=dic)
        reset_msg = {"type": "reset", "reset_dic": dic}
        self.pipe.send(reset_msg)
        self.pipe.poll(None)
        received_state = self.pipe.recv()
        self.update_state(received_state)
        self.bundle.reset(task=False)
        return self.state


class ClassicControlTask(InteractionTask):
    """verify F and G conversions."""

    def __init__(
        self,
        timestep,
        A,
        B,
        F=None,
        G=None,
        discrete_dynamics=True,
        noise="on",
    ):
        super().__init__()

        self.dim = max(A.shape)
        self.state = State()
        self.state["x"] = StateElement(
            values=numpy.zeros((self.dim, 1)),
            spaces=Space(
                [
                    -numpy.ones((self.dim, 1)) * numpy.inf,
                    numpy.ones((self.dim, 1)) * numpy.inf,
                ]
            ),
        )
        self.state_last_x = copy.copy(self.state["x"]["values"])
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
            self.A_c = 1 / timestep * (A - numpy.eye(A.shape[0]))
            self.B_c = B / timestep
        else:
            self.A_c = A
            self.B_c = B
            # Euler Method
            self.A_d = numpy.eye(A.shape[0]) + timestep * A
            self.B_d = timestep * B

        self.noise = noise

    def finit(self):
        if self.bundle.user.timespace == "continuous":
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic=None):
        super().reset()
        # Force zero velocity
        self.state["x"]["values"][0][1:] = 0

        if dic is not None:
            super().reset(dic=dic)

        self.state_last_x = copy.copy(self.state["x"]["values"])

    def user_step(self, user_action):
        # print(user_action, type(user_action))
        is_done = False
        # Call super for counters
        super().user_step(user_action)

        # For readability
        A, B, F, G = self.A, self.B, self.F, self.G
        u = user_action["values"][0]
        x = self.state["x"]["values"][0].reshape(-1, 1)

        # Generate noise samples
        if self.noise == "on":
            beta, gamma = numpy.random.normal(
                0, numpy.sqrt(self.timestep), (2, 1)
            )
            omega = numpy.random.normal(
                0, numpy.sqrt(self.timestep), (self.dim, 1)
            )
        else:
            beta, gamma = numpy.random.normal(0, 0, (2, 1))
            omega = numpy.random.normal(0, 0, (self.dim, 1))

        # Store last_x for render
        self.state_last_x = copy.deepcopy(self.state["x"]["values"])
        # Deterministic update + State dependent noise + independent noise
        if self.bundle.user.timespace == "discrete":
            x = (A @ x + B * u) + F @ x * beta + G @ omega
        else:
            x += (A @ x + B * u) * self.timestep + F @ x * beta + G @ omega

        self.state["x"]["values"] = x
        if abs(x[0, 0]) <= 0.01:
            is_done = True

        return self.state, 0, is_done, {}

    def assistant_step(self, assistant_action):
        # return super().assistant_step(assistant_action)
        return self.state, 0, False, {}

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "text" in mode:
            print("state")
            print(self.state["x"])
        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                self.draw()
                if self.turn_number == 3:
                    self.ax.legend(
                        handles=[
                            self.axes[i].lines[0] for i in range(self.dim)
                        ]
                    )
            else:
                self.color = ["b", "g", "r", "c", "m", "y", "k"]
                self.labels = ["x[{:d}]".format(i) for i in range(self.dim)]
                self.axes = [axtask]
                self.ax = axtask
                for i in range(self.dim - 1):
                    self.axes.append(self.ax.twinx())

                for i, ax in enumerate(self.axes):
                    # ax.yaxis.label.set_color(self.color[i])
                    ax.tick_params(axis="y", colors=self.color[i])

                self.draw()

    def draw(self):
        if (self.state_last_x[0] == self.state["x"]["values"][0]).all():
            pass
        else:
            for i in range(self.dim):
                self.axes[i].plot(
                    [
                        ((self.turn_number - 1) / 2 - 1) * self.timestep,
                        (self.turn_number - 1) / 2 * self.timestep,
                    ],
                    flatten(
                        [
                            self.state_last_x[0][i, 0].tolist(),
                            self.state["x"]["values"][0][i, 0].tolist(),
                        ]
                    ),
                    "-",
                    color=self.color[i],
                    label=self.labels[i],
                )

        return


class TaskWrapper(InteractionTask):
    def __init__(self, task, *args, **kwargs):
        self.task = task
        self.__dict__.update(task.__dict__)

    def user_step(self, *args, **kwargs):
        return self.task.user_step(*args, **kwargs)

    def assistant_step(self, *args, **kwargs):
        return self.task.assistant_step(*args, **kwargs)

    def reset(self, dic=None):
        return self.task.reset(dic=dic)

    def render(self, *args, **kwargs):
        return self.task.render(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.task.unwrapped


# ======================== Examples ======================


class ExampleTask(InteractionTask):
    """ExampleTask with two agents. The task is finished when 'x' reaches 4. The user and assistants can both do -1, +0, +1 to 'x'."""

    def __init__(self, *args, **kwargs):
        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task
        super().__init__(*args, **kwargs)

        self.state["x"] = StateElement(
            values=0,
            spaces=Space(
                [
                    numpy.array(
                        [-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16
                    )
                ]
            ),
            clipping_mode="clip",
        )

    def reset(self, dic=None):
        self.state["x"]["values"] = numpy.array([0])
        return

    def user_step(self, *args, **kwargs):
        is_done = False
        self.state["x"] += self.user_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def assistant_step(self, *args, **kwargs):
        is_done = False
        self.state["x"] += self.assistant_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def render(self, *args, mode="text"):
        if "text" in mode:
            print("\n")
            print(
                "Turn number {:f}".format(self.turn_number.squeeze().tolist())
            )
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError
