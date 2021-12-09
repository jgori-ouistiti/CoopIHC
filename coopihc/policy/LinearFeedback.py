import numpy
from coopihc.policy.BasePolicy import BasePolicy

# ============== General Policies ===============


class LinearFeedback(BasePolicy):
    """LinearFeedback

    Linear Feedback policy, which applies a feedback gain to a given state of the observation.

    :param state_indicator: which state to be used as feedback
    :type state_indicator: string
    :param slice: slice to select substates of state_indicator
    :type slice: slice
    :param action_state: see `BasePolicy<coopihc.policy.BasePolicy.BasePolicy`
    :type action_state: `State<coopihc.space.State.State`
    :param feedback_gain: feedback gain matrix, defaults to "identity"
    :type feedback_gain: numpy.ndarray, optional
    """

    def __init__(
        self,
        state_indicator,
        action_state,
        *args,
        feedback_gain="identity",
        slice=None,
        **kwargs
    ):
        super().__init__(*args, action_state=action_state, **kwargs)
        self.state_indicator = state_indicator
        self.slice = slice
        self.noise_function = kwargs.get("noise_function")
        # bind the noise function
        if self.noise_function is not None:
            self._bind(self.noise_function, as_name="noise_function")

        self.noise_args = kwargs.get("noise_function_args")
        self.feedback_gain = feedback_gain

    def set_feedback_gain(self, gain):
        """set_feedback_gain

        set feedback gain. Only needed if the gain needs to be changed after initialization, otherwise it is recommended to set the gain during initialiation of the policy.

        :param gain: feedback gain matrix
        :type gain: numpy.ndarray
        """
        self.feedback_gain = gain

    def sample(self, observation=None):
        """sample

        Applies the gain.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """
        if observation is None:
            observation = self.observation

        if isinstance(self.slice, list):
            raise NotImplementedError

        substate = observation
        for key in self.state_indicator:
            substate = substate[key]
        if self.slice is not None:
            substate = substate[self.slice]

        if isinstance(self.feedback_gain, str):
            if self.feedback_gain == "identity":
                self.feedback_gain = -numpy.eye(max(substate["values"][0].shape))

        noiseless_feedback = -self.feedback_gain @ substate.reshape((-1, 1))
        noise = self.noise_function(noiseless_feedback, observation, *self.noise_args)
        action = self.action
        action["values"] = noiseless_feedback + noise.reshape((-1, 1))
        # if not hasattr(noise, '__iter__'):
        #     noise = [noise]
        # header = ['action', 'noiseless', 'noise']
        # rows = [action, noiseless_feedback , noise]
        # logger.info('Policy {} selected action\n{})'.format(self.__class__.__name__, tabulate(rows, header) ))
        return action, 0
