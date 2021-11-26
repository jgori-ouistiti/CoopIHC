from abc import ABC, abstractmethod
from coopihc.space import State, StateElement


class InteractionTask(ABC):
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

    def _base_reset(self, dic=None):
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
