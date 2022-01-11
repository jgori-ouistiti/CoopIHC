import numpy
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.space.Space import Space


class BadlyDefinedLikelihoodError(Exception):
    pass


class ELLDiscretePolicy(BasePolicy):
    """ELLDiscretePolicy

    Explicitly defined Likelihood Policy. A policy which is described by an explicit probabilistic model.

    This policy expects you to bind the actual likelihood model. An example is as follows:

     .. code-block:: python

        se = StateElement(
            1, autospace([0, 1, 2, 3, 4, 5, 6]), seed=_seed
        )
        action_state = State(**{"action": se})
        policy = ELLDiscretePolicy(action_state, seed=_seed)

        # Define the likelihood model
        def likelihood_model(self, action, observation, *args, **kwargs):
            if action == 0:
                return 1 / 7
            elif action == 1:
                return 1 / 7 + 0.05
            elif action == 2:
                return 1 / 7 - 0.05
            elif action == 3:
                return 1 / 7 + 0.1
            elif action == 4:
                return 1 / 7 - 0.1
            elif action == 5:
                return 1 / 7 + 0.075
            elif action == 6:
                return 1 / 7 - 0.075
            else:
                raise RuntimeError(
                    "warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition"
                )

        # Attach it
        policy.attach_likelihood_function(likelihood_model)

    .. note::

        The signature of the likelihood model should be the same signature as a bound method (i.e. the first argument is self)

    :param action_state: See the BasePolicy keyword argument with the same name
    :type action_state: See the BasePolicy keyword argument with the same name
    :param seed: seed for the RNG
    :type seed: int, optional

    """

    def __init__(self, action_state, *args, seed=None, **kwargs):
        super().__init__(*args, action_state=action_state, **kwargs)
        self.explicit_likelihood = True
        self.rng = numpy.random.default_rng(seed)

    def attach_likelihood_function(self, _function):
        """attach_likelihood_function

        Bind the likelihood model by calling BasePolicy's _bind method.

        :param _function: likelihood model to bind to the policy
        :type _function: function
        """
        self._bind(_function, "compute_likelihood")

    def sample(self, observation=None):
        """sample from likelihood model.

        Select an action according to its probability as defined by the likelihood model. You can pass an observation as well, in which case the policy will not look up he actual observation but use the observation you passed. This is useful e.g. when debugging the policy.

        :param observation: if passed, this is the observation upon which action selection is based upon. Otherwise, the policy will look at the actual agent observation, defaults to None
        :type observation: `State<coopihc.space.State.State>`, optional
        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        if observation is None:
            observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[self.rng.choice(len(llh), p=llh)]
        self.action_state["action"][:] = action
        return self.action_state["action"], 0

    def forward_summary(self, observation):
        """forward_summary

        Compute the likelihood of each action, given the current observation

        :param observation: current agent observation
        :type observation: `State<coopihc.space.State.State>`
        :return: [description]
        :rtype: [type]
        """
        llh, actions = [], []
        action_stateelement = self.action_state["action"]
        action_space = action_stateelement.spaces

        for action in Space.cartesian_product(action_space)[0]:
            llh.append(self.compute_likelihood(action, observation))
            actions.append(action)
        ACCEPTABLE_ERROR = 1e-13
        error = abs(1 - sum(llh))
        if error > ACCEPTABLE_ERROR:
            raise BadlyDefinedLikelihoodError(
                "Likelihood does not sum to 1: {}".format(llh)
            )
        return actions, llh
