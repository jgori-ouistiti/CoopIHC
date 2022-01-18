import numpy
from coopihc.policy.BasePolicy import BasePolicy

# ============== General Policies ===============


class LinearFeedback(BasePolicy):
    """LinearFeedback

    Linear Feedback policy, which applies a feedback gain to a given state of the observation, and passes it to a function.

    For example with:

        + ``state_indicator = ('user_state', 'substate1', slice(0,2,1))``
        + ``feedback_gain = -numpy.eye(2)``
        + ``noise_function = f(action, observation, *args)``
        + ``noise_func_args = (1,2)``

    You will get

    .. code-block:: python

        obs = observation['user_state']['substate_1'][slice(0,2,1)]
        action = - -numpy.eye(2 @ obs)
        return f(action, observation, *(1,2))


    You can change the feedback gain online via the ``set_feedback_gain()`` method


    :param action_state: see `BasePolicy<coopihc.policy.BasePolicy.BasePolicy`
    :type action_state: `State<coopihc.space.State.State`
    :param state_indicator: specifies which component is used as feedback information e.g. ``('user_state', 'substate1', slice(0,2,1))``
    :type state_indicator: iterable
    :param feedback_gain: Feedback gain matrix, defaults to "identity", which creates a negative identity matrix.
    :type feedback_gain: str or numpy.ndarray, optional
    :param noise_function: a function that produces a noise sample to add to the generated action, defaults to None
    :type noise_function: function, optional
    :param noise_func_args: arguments to the function above, defaults to ()
    :type noise_func_args: tuple, optional

    """

    def __init__(
        self,
        action_state,
        state_indicator,
        *args,
        feedback_gain="identity",
        noise_function=None,
        noise_func_args=(),
        **kwargs
    ):
        super().__init__(*args, action_state=action_state, **kwargs)
        self.state_indicator = state_indicator

        self.feedback_gain = feedback_gain
        self.noise_function = noise_function
        self.noise_args = noise_func_args

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

        output = observation
        for item in self.state_indicator:
            output = output[item]

        output = output.view(numpy.ndarray)

        if isinstance(
            self.feedback_gain, str
        ):  # Checking type is needed to suppress FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison in Python 3.8 + NumPy 1.22.0
            if self.feedback_gain == "identity":
                self.feedback_gain = -numpy.eye(max(output.shape))

        noiseless_feedback = -self.feedback_gain @ output.reshape((-1, 1))
        noisy_action = self.noise_function(
            noiseless_feedback, observation, *self.noise_args
        )
        action = self.action
        action[:] = noisy_action

        return action, 0
