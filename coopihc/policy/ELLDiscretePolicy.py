import numpy
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.space.Space import Space


class BadlyDefinedLikelihoodError(Exception):
    pass


class ELLDiscretePolicy(BasePolicy):
    """ELLDiscretePolicy

    Explicitly defined Likelihood Policy. A policy which is described by an explicit probabilistic model.

    """

    def __init__(self, action_state, *args, seed=None, **kwargs):
        super().__init__(*args, action_state=action_state, **kwargs)
        self.explicit_likelihood = True
        self.rng = numpy.random.default_rng(seed)

    # @classmethod
    # def attach_likelihood_function(cls, _function):
    #     """attach_likelihood_function

    #     Attach the probabilistic model (likelihood function) to the class

    #     :param _function: likelihood function
    #     :type _function: function
    #     """

    #     cls.compute_likelihood = _function

    def attach_likelihood_function(self, _function):
        self._bind(_function, "compute_likelihood")

    @staticmethod
    def test():
        pass

    def sample(self, observation=None):
        """sample

        Select the action according to its probability

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """
        if observation is None:
            observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[self.rng.choice(len(llh), p=llh)]
        self.action_state["action"][:] = action
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
