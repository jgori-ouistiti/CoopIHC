from core.agents import BaseAgent, DiscreteBayesianBeliefAssistant
from core.helpers import sort_two_lists

import gym
import numpy
from collections import OrderedDict
import math

class ConstantCDGain(BaseAgent):
    """ A Constant CD Gain transfer function.

    Here the assistant just picks a fixed modulation.

    :param gain: (float) constant CD gain.

    :meta public:
    """
    def __init__(self, gain):
        self.gain = gain
        super().__init__(   "assistant",
                            [gym.spaces.Box(low = numpy.array([-10]), high = numpy.array([10]))],
                            [None]
                            )

    def sample(self):
        """ Sample Assistant policy.

        Return the constant CD gain.

        :meta public:
        """
        return [self.gain]

    def finit(self):
        return

    def reset(self):
        return


class BIGGain(DiscreteBayesianBeliefAssistant):
    """ A Bayesian Information Gain Gain Assistant.

    Here the gain is selected so as to maximize the information gain for the assistant with regards to the user's goal. See Liu, Wanyu, et al. "Bignav: Bayesian information gain for guiding multiscale navigation." Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems. 2017.

    :param operator_model: (core.model) An operator model. It can be the same model as the one used by the operator, or not (it should have the same possible actions, but not necessarily the same likelihood).
    :param threshold: (float) When one potential target's posterior probability goes above this threshold, go directly towards it.

    :meta public:
    """
    def __init__(self, operator_model, threshold = 0.5):
        # If one posterior probability goes above threshold, then the gain will be computed so as to reach it immediately.

        super().__init__(   [gym.spaces.Box(low = numpy.array([-10]), high = numpy.array([10]))],
                            [None],
                            operator_model,
                            observation_engine = None)
        self.operator_model = operator_model
        self.threshold = threshold

    def PYy_Xx(self, operator_action, position, targets, beliefs):
        r""" Compute the conditional probability :math:`P(Y=y|X=x)`

        :param operator_action: given operator action y for which the condition is computed
        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the conditional :math:`P(Y=y|X=x)`

        :meta public:
        """
        pYy__Xx = 0
        for t,b in zip(targets, beliefs):
            observation = self.generate_candidate_observation(t, position)
            try:
                pYy__Xx += self.operator_model.compute_likelihood(operator_action, observation)*b
            except TypeError:
                print(operator_action, observation, b)
        return pYy__Xx


    def HY__Xx(self, position, targets, beliefs):
        r""" Computes the conditional entropy :math:`H(Y |X=x) = -\mathbb{E}[\log(p(Y|X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |X=x)`

        :meta public:
        """
        H = 0
        for operator_action in self.operator_model.actions:
            pYy_Xx = self.PYy_Xx(operator_action, position, targets, beliefs)
            if pYy_Xx != 0:
                H += -pYy_Xx * math.log(pYy_Xx,2)
        return H

    def HY__OoXx(self, position, targets, beliefs):
        r""" Computes the conditional entropy :math:`H(Y |\Theta = \theta, X=x) = -\mathbb{E}[\log(p(Y|\Theta = \theta, X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |\Theta = \theta, X=x)`

        :meta public:
        """
        H = 0
        for operator_action in self.operator_model.actions:
            for t,b in zip(targets, beliefs):
                observation = self.generate_candidate_observation(t, position)
                pYy__OoXx = self.operator_model.compute_likelihood(operator_action, observation)
                if pYy__OoXx != 0:
                    H += -b*pYy__OoXx*math.log(pYy__OoXx,2)
        return H

    def IG(self, position, targets, beliefs):
        r""" Computes the expected information gain :math:`\mathrm{IG}(X=x) = H(Y |X=x) - H(Y |\Theta = \theta, X=x)` for a future position.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the information gain  :math:`\mathrm{IG}(X=x)`

        :meta public:
        """
        return self.HY__Xx(position, targets, beliefs) - self.HY__OoXx(position, targets, beliefs)

    def find_best_pos(self):
        """ Finds expected information gain associated with each possible future cursor position and ranks them in order from the most to  less informative.

        :return: pos, IG. Future cursor position and associated expected information gain.

        :meta public:
        """
        targets = self.targets
        beliefs = self.state['Beliefs']
        hp, hp_target = max(beliefs), targets[beliefs.index(max(beliefs))]
        if hp > self.threshold:
            return [hp_target], [None]
        else:
            IG_storage = [self.IG( pos, targets, beliefs) for pos in range(self.bundle.task.state['Gridsize'][0])]
            _IG, pos = sort_two_lists(IG_storage, list(range(self.bundle.task.state['Gridsize'][0])))
            pos.reverse(), _IG.reverse()
            return pos, _IG


    def sample_discrete(self):
        """ Get the position with maximum expected information gain.

        :return: position.

        :meta public:
        """
        new_pos, probabilities = self.find_best_pos()
        return new_pos[0]


    def generate_candidate_observation(self, target, position):
        """ Used to generate candidate observations, compatible with the operator model.

        :param target: a target, which is a potential goal
        :param position: the cursor's position

        :return: (OrderedDict) an observation compatible with the operator model.
        """
        observation = OrderedDict({})
        observation['operator_state'] = OrderedDict({'Goal': [target]})
        observation['task_state'] = OrderedDict({'Position': [position]})
        return observation



    def sample(self, mode = 'gain'):
        r""" Redefine the sample method of the assistant to use BIG.

        :meta public:
        """
        ret = self.sample_discrete()
        if mode == 'gain':
            operator_action = self.state['OperatorAction'][0]
            position = self.bundle.task.state['Position'][0]
            ret =  (ret - position)/operator_action
        else:
            pass
        return [ret]
