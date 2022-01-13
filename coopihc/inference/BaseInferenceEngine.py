from collections import OrderedDict


# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)
class BaseInferenceEngine:
    """BaseInferenceEngine

    The base Inference Engine from which other engines can be defined. This engine does nothing but return the same state. Any new inference method can subclass ``InferenceEngine`` to have a buffer and ``add_observation`` method (required by the bundle)

    :param buffer_depth: number of observations that are stored, defaults to 1
    :type buffer_depth: int, optional
    """

    """"""

    def __init__(self, *args, buffer_depth=1, **kwargs):
        self.buffer = None
        self.buffer_depth = buffer_depth
        self.render_flag = None
        self.ax = None

    def __content__(self):
        """__content__

        Custom class representation

        :return: representation
        :rtype: string
        """
        return self.__class__.__name__

    @property
    def observation(self):
        """observation

        The last observation.

        :return: last observation
        :rtype: :py:class:`State<coopihc.space.State.State>`
        """
        return self.buffer[-1]

    @property
    def state(self):
        """state

        The current agent state

        :return: agent state
        :rtype: :py:class:`State<coopihc.space.State.State>`
        """
        return self.buffer[-1]["{}_state".format(self.host.role)]

    @property
    def action(self):
        """action

        The agent's last action

        :return: agent action
        :rtype: :py:class:`State<coopihc.space.State.State>`
        """
        return self.host.policy.action_state["action"]

    @property
    def unwrapped(self):
        return self

    def add_observation(self, observation):
        """add observation

        Add an observation to a buffer. If the buffer does not exist, create a naive buffer. The buffer has a size given by buffer length

        :param observation: observation produced by an engine
        :type observation: :py:class:`State<coopihc.space.State.State>`
        """

        if self.buffer is None:
            self.buffer = []
        if len(self.buffer) < self.buffer_depth:
            self.buffer.append(observation)
        else:
            self.buffer = self.buffer[1:] + [observation]

    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    def bind(self, func, as_name=None):
        """bind

        Bind function to the engine with a given name. If as_name is None, then the func name is used.

        :param func: function to bind
        :type func: function
        :param as_name: name of resulting method, defaults to None
        :type as_name: string, optional
        :return: bound method
        :rtype: method
        """
        # print("\n")
        # print(func, as_name)
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(self, self.__class__)
        setattr(self, as_name, bound_method)
        return bound_method

    def infer(self, user_state=None):
        """infer

        The main method of this class. Return the new value of the internal state of the agent, as well as the reward associated with inferring the state. By default, this inference engine does nothing, and just returns the state with a null reward.


        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        if user_state is not None:
            return user_state, 0
        # do something with information inside buffer
        if self.host.role == "user":
            try:
                return self.observation["user_state"], 0
            except KeyError:
                return {}, 0
        else:
            try:
                return self.observation["assistant_state"], 0
            except KeyError:
                return {}, 0

    def reset(self):
        """reset

        Reset the engine (clear buffer)
        """
        self.buffer = None

    def render(self, *args, **kwargs):
        """render

        Render the engine.
        """
        mode = kwargs.get("mode")

        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        if render_flag:

            if "plot" in mode:
                ax = args[0]
                if self.ax is not None:
                    pass
                else:
                    self.ax = ax
                    self.ax.set_title(type(self).__name__)

            if "text" in mode:
                print(type(self).__name__)
