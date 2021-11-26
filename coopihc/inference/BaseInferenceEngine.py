from collections import OrderedDict


# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)
class BaseInferenceEngine:
    """Does nothing but return the same state. Any new inference method can subclass
    InferenceEngine to have a buffer and add_observation method (required by the bundle)"""

    def __init__(self, buffer_depth=1):
        self.buffer = None
        self.buffer_depth = buffer_depth
        self.render_flag = None
        self.ax = None

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        return self.buffer[-1]

    @property
    def state(self):
        return self.buffer[-1]["{}_state".format(self.host.role)]

    @property
    def action(self):
        return self.host.policy.action_state["action"]

    @property
    def unwrapped(self):
        return self

    def add_observation(self, observation):
        """add an observation  to a naive buffer.

        :param observation: verify type.
        """

        if self.buffer is None:
            self.buffer = []
        if len(self.buffer) < self.buffer_depth:
            self.buffer.append(observation)
        else:
            self.buffer = self.buffer[1:] + [observation]

    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    def bind(self, func, as_name=None):
        # print("\n")
        # print(func, as_name)
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(self, self.__class__)
        setattr(self, as_name, bound_method)
        return bound_method

    def infer(self):
        """The main method of this class.

        Return the new value of the internal state of the agent, as well as the reward associated with inferring the . By default, this inference engine does nothing, and just returns the state.

        :return: new_internal_state (OrderedDict), reward (float)
        """
        # do something with information inside buffer

        if self.host.role == "user":
            try:
                return self.buffer[-1]["user_state"], 0
            except KeyError:
                return OrderedDict({}), 0
        elif self.host.role == "assistant":
            try:
                return self.buffer[-1]["assistant_state"], 0
            except KeyError:
                return OrderedDict({}), 0

    def reset(self):
        self.buffer = None

    def render(self, *args, **kwargs):
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
