"""This module provides access to the InteractionTask class."""


from abc import ABC, abstractmethod
from coopihc.base.State import State
from coopihc.base.StateElement import StateElement
import numpy


"""
    The main API methods for this class are:

        __init__

        finit

        reset

        on_user_action

        on_assistant_action

        render

    :meta public:
    """


class InteractionTask(ABC):
    """InteractionTask

    The class that defines an Interaction Task. Subclass this task when
    creating a new task to ensure compatibility with CoopIHC. When doing so,
    make sure to call ``super()`` in ``__init__()``.

    """

    def __init__(self, *args, **kwargs):

        self._state = State()
        self.bundle = None
        self.timestep = 0.1
        self._parameters = {}

        # Render
        self.ax = None

    def finit(self):
        return

    @property
    def turn_number(self):
        """Turn number.

        The turn number of the game

        :return: turn number
        :rtype: numpy.ndarray
        """
        if self.bundle is None:
            no_bundle_specified = "turn_number accesses the bundle's turn number. self.bundle was None. Is this task part of a bundle?"
            raise Exception(no_bundle_specified)
        return self.bundle.turn_number

    @property
    def round_number(self):
        if self.bundle is None:
            no_bundle_specified = "turn_number accesses the bundle's turn number. self.bundle was None. Is this task part of a bundle?"
            raise Exception(no_bundle_specified)
        return self.bundle.round_number

    @property
    def state(self):
        """state

        The current state of the task.

        :return: task state
        :rtype: :py:class:`State<coopihc.base.State.State>`
        """
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def user_action(self):
        """user action

        The last action input by the user.

        :return: user action
        :rtype: :py:class:`State<coopihc.base.State.State>`
        """
        try:
            return self.bundle.user.action
        except AttributeError:
            raise AttributeError("This task has not been connected to a user yet")

    @property
    def assistant_action(self):
        """assistant action

        The last action input by the assistant.

        :return: assistant action
        :rtype: :py:class:`State<coopihc.base.State.State>`
        """
        try:
            return self.bundle.assistant.action
        except AttributeError:
            raise AttributeError("This task has not been connected to an assistant yet")

    # def __getattr__(self, value):
    #     try:
    #         return self.parameters.__getitem__(value)
    #     except:
    #         raise AttributeError(
    #             f"'{self.__class__.__name__}' object has no attribute '{value}'"
    #         )

    @property
    def parameters(self):
        if self.bundle:
            return self.bundle.parameters
        return self._parameters

    def __content__(self):
        """Custom class representation.

        A custom class representation.

        :return: custom repr
        :rtype: dictionnary
        """
        return {
            "Name": self.__class__.__name__,
            "State": self.state.__content__(),
        }
        """Describe how the task state should be reset. This method has to be
        redefined when subclassing this class.

        :param args: (OrderedDict) state to which the task should be reset,
        if provided.

        :return: state (OrderedDict) of the task.

        :meta public:
        """

    def _base_reset(self, dic=None, random=True):
        """base reset

        Method that wraps the user defined reset() method. Takes care of the
        dictionary reset mechanism and updates rounds.

        :param dic: reset dictionary (passed by bundle),
        :type dic: dictionary, optional
        :param random: whether to randomize task states, defaults to True
        :type random: boolean, optional
        """

        if random:
            # Reset everything randomly before  starting
            self.state.reset(dic={})
        # Apply end-user defined reset
        self.reset(dic=dic)

        if not dic:
            return

        # forced reset with dic
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                self.state[key] = value
                continue
            elif isinstance(value, numpy.ndarray):
                self.state[key][...] = value

            elif value is None:
                continue
            else:
                raise NotImplementedError(
                    "Values in the reset dictionnary should be of type StateElement or numpy.ndarray, but you provided values of type {} ({})".format(
                        value.__class__.__name__, str(value)
                    )
                )

    def base_on_user_action(self, *args, **kwargs):
        """base user step

        Wraps the user defined on_user_action() method. For now does little but
        provide default values, may be useful later.

        :return: (task state, task reward, is_done flag, metadata):
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float, boolean, dictionnary)
        """
        ret = self.on_user_action(*args, **kwargs)
        if ret is None:
            return self.state, -1 / 2, False, {}
        else:
            return ret

    def base_on_assistant_action(self, *args, **kwargs):
        """base assistant step

        Wraps the assistant defined on_assistant_action() method. For now does
        little but provide default values, may be useful later.

        :return: (task state, task reward, is_done flag, metadata):
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float, boolean, dictionnary)
        """
        ret = self.on_assistant_action(*args, **kwargs)
        if ret is None:
            return self.state, -1 / 2, False, {}
        else:
            return ret

    @abstractmethod
    def on_user_action(self, *args, user_action=None, **kwargs):
        """on_user_action

        Redefine this to specify the task state transitions and rewards issued.

        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float, boolean, dictionnary)
        """
        return None

    @abstractmethod
    def on_assistant_action(self, *args, assistant_action=None, **kwargs):
        """on_assistant_action

        Redefine this to specify the task state transitions and rewards issued.

        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float, boolean, dictionnary)
        """
        return None

    @abstractmethod
    def reset(self):
        """reset

        Redefine this to specify how to reinitialize the task before each new
        game.

        """
        return None

    def render(self, mode="text", ax_user=None, ax_assistant=None, ax_task=None):
        """Render the task on the main plot.

        :param mode: (str) text or plot
        :param args: (list) list of axis in order axtask, axuser, axassistant

        """
        if mode is None:
            mode = "text"

        if "text" in mode:
            print(self.state)
        else:
            pass
