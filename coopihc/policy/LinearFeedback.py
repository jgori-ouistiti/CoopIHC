import numpy
from coopihc.policy.BasePolicy import BasePolicy

# ============== General Policies ===============


class LinearFeedback(BasePolicy):
    def __init__(
        self,
        state_indicator,
        index,
        action_state,
        *args,
        feedback_gain="identity",
        **kwargs
    ):
        super().__init__(action_state, *args, **kwargs)
        self.state_indicator = state_indicator
        self.index = index
        self.noise_function = kwargs.get("noise_function")
        # bind the noise function
        if self.noise_function is not None:
            self._bind(self.noise_function, as_name="noise_function")

        self.noise_args = kwargs.get("noise_function_args")
        self.feedback_gain = feedback_gain

    def set_feedback_gain(self, gain):
        self.feedback_gain = gain

    def sample(self):
        if isinstance(self.index, list):
            raise NotImplementedError
        substate = self.observation
        for key in self.state_indicator:
            substate = substate[key]
        substate = substate[self.index]

        if isinstance(self.feedback_gain, str):
            if self.feedback_gain == "identity":
                self.feedback_gain = -numpy.eye(max(substate["values"][0].shape))

        noiseless_feedback = -self.feedback_gain @ substate.reshape((-1, 1))
        noise = self.noise_function(
            noiseless_feedback, self.observation, *self.noise_args
        )
        action = self.action
        action["values"] = noiseless_feedback + noise.reshape((-1, 1))
        # if not hasattr(noise, '__iter__'):
        #     noise = [noise]
        # header = ['action', 'noiseless', 'noise']
        # rows = [action, noiseless_feedback , noise]
        # logger.info('Policy {} selected action\n{})'.format(self.__class__.__name__, tabulate(rows, header) ))
        return action, 0
