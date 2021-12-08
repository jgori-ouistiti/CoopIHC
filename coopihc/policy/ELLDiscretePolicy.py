import numpy
from coopihc.policy.BasePolicy import BasePolicy


class BadlyDefinedLikelihoodError(Exception):
    pass


class ELLDiscretePolicy(BasePolicy):
    """ELLDiscretePolicy

    Explicitly defined Likelihood Policy. A policy which is described by an explicit probabilistic model.

    """

    def __init__(self, *args, **kwargs):
        action_state = kwargs.pop("action_state")
        super().__init__(action_state, *args, **kwargs)
        self.explicit_likelihood = True
        self.rng = numpy.random.default_rng(kwargs.get("seed"))

    @classmethod
    def attach_likelihood_function(cls, _function):
        """attach_likelihood_function

        Attach the probabilistic model (likelihood function) to the class

        :param _function: likelihood function
        :type _function: function
        """
        cls.compute_likelihood = _function

    def sample(self):
        """sample

        Select the action according to its probability

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[self.rng.choice(len(llh), p=llh)]
        self.action_state["action"] = action
        return action, 0

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
        for action in action_stateelement.cartesian_product():
            llh.append(self.compute_likelihood(action, observation))
            actions.append(action)
        ACCEPTABLE_ERROR = 1e-13
        error = 1 - sum(llh)
        if error > ACCEPTABLE_ERROR:
            raise BadlyDefinedLikelihoodError(
                "Likelihood does not sum to 1: {}".format(llh)
            )
        return actions, llh
