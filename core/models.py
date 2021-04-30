from abc import ABC, abstractmethod
import numpy

from core.helpers import sort_two_lists

class DiscreteOperatorModelMax(ABC):
    """ A Discrete Operator Model.

    An operator model is a policy, described by a likelihood model. By default the policy selects the action with highest likelihood (greedy) given the operator's current observation.

    :param actions: (list) list of possible actions.

    :meta public:
    """
    @abstractmethod
    def __init__(self, actions):
        self.actions = actions
        return

    def forward_summary(self, observation):
        """ Compute the likelihood of each action, given the current observation

        :param observation: (OrderedDict) operator observation

        :return: actions, likelihood. list of actions and associated likelihoods

        :meta public:
        """
        llh = []
        for i in self.actions:
            llh.append(self.compute_likelihood(i, observation))
        return self.actions, llh

    @abstractmethod
    def compute_likelihood(self, action, observation):
        """ This is what specifies the operator model.

        Redefine this when subclassing the Discrete operator model.

        :param action: (float) the action of which we want to evaluate the likelihood
        :param observation: (OrderedDict) observation used for context in the likelihood computation.

        :return: likelihood (float) of that particular action given the observation.

        :meta public:
        """
        return

    def sample(self, observation):
        """ Select the most likely action.

        :param observation: (OrderedDict)
        :return: action. most likely action.
        """
        actions, llh = self.forward_summary(observation)
        llh, actions = sort_two_lists(llh, actions)
        return actions[-1]


class GoalDrivenBinaryOperatorModel(DiscreteOperatorModelMax):
    """ A Model for a Binary Operator.

    A Binary Operator only has two actions (opposite of each other)

    :param action: the set of actions for the binary operator is ``[-action, action]`` If the actions are discrete actions, enter the real value of actions.
    """
    def __init__(self, action):
        super().__init__([-action, action])

    #
    def compute_likelihood(self, action, observation):
        """ This is what specifies the operator model.

        .. warning::
            Here the action is expected to be in range (0,N) if it is Discrete (You may need to convert it in your likelihood expression)

        :param action: (float) the action of which we want to evaluate the likelihood
        :param observation: (OrderedDict) observation used for context in the likelihood computation.

        :return: likelihood (float) of that particular action given the observation.

        :meta public:
        """
        # Write down all possible cases (4)
        # (1) Goal to the right, positive action
        if observation['operator_state']['Goal'][0] > observation['task_state']['Position'][0] and action > 0 :
            return .99
        # (2) Goal to the right, negative action
        elif observation['operator_state']['Goal'][0] > observation['task_state']['Position'][0] and action < 0 :
            return .01
        # (3) Goal to the left, positive action
        if observation['operator_state']['Goal'][0] < observation['task_state']['Position'][0] and action > 0 :
            return .01
        # (4) Goal to the left, negative action
        elif observation['operator_state']['Goal'][0] < observation['task_state']['Position'][0] and action < 0 :
            return .99
        elif observation['operator_state']['Goal'][0] == observation['task_state']['Position'][0]:
            return 0
        else:
            print("warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition")
            raise NotImplementedError


class ContinuousGaussianOperatorModel(ABC):
    def __init__(self, Sigma):
        self.Sigma = Sigma

    @abstractmethod
    def compute_mu(self, observation):
        return numpy.array([1])

    def compute_likelihood(self, action, observation):
        mu = self.compute_mu(observation)
        k = max(mu.shape)
        prefactor = (2*numpy.pi)**(-k/2)/numpy.sqrt(numpy.linalg.det(Sigma))
        expfactor = -1/2*(action-mu).T @ numpy.linalg.inv(Sigma) @ (action-mu)
        return prefactor*numpy.exp(expfactor)

    def sample(self, observation):
        mu = self.compute_mu(observation)
        return numpy.random.multivariate_normal(mu, self.Sigma)

class LinearEstimatedFeedback(ContinuousGaussianOperatorModel):
    """ A linear Feedback from the estimated state.

    Expects an operator with a state called 'xhat'. Produces the action u = -L @ Xhat

    :param L: (numpy.ndarray) Kalman Feedback Gain matrix
    :param Sigma: (numpy.ndarray) covariance matrix of noise used to specify the likelihood :math:`\mathcal{N}(-L \hat{x}, \Sigma)`
    """
    def __init__(self, L, Gamma):
        self.L = L
        super().__init__(Gamma)
        return

    def compute_mu(self, observation):
        return -self.L @ observation["operator_state"]['xhat']
