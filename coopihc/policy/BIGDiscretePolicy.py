import copy
import math
import copy

from coopihc.space.State import State
from coopihc.space.Space import Space
from coopihc.helpers import sort_two_lists
from coopihc.policy.BasePolicy import BasePolicy

import numpy

# ============= Discrete Policies =================

# ----------- Bayesian Information Gain Policy ---------


class BIGDiscretePolicy(BasePolicy):
    """BIGDiscretePolicy [summary]

    Bayesian Information Gain Policy, adapted from [1]_.

    The main ideas/assumptions are:

        * A user wants the task to go to some goal state :math:`\\Theta`
        * The assistant can put the task in a number of states (X)
        * The user can perform a given set of action Y
        * A model :math:`p(Y=y|X=X, \\Theta = \\theta)` exists for user behavior

    Make sure to call:

        * attach_set_theta, to specify the potential goal states
        * attach_transition_function, to specify how the task state evolves after an assistant action




    .. [1] Liu, Wanyu, et al. "Bignav: Bayesian information gain for guiding multiscale navigation." Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems. 2017.

    :param assistant_action_state: action state of the assistant
    :type assistant_action_state: `State<coopihc.space.State.State>`
    :param user_policy_model: user policy model. This may be the real policy of the user, but realistically has to be a model of the user policy. This policy must currently be an `ELLDiscretePolicy<coopihc.policy.ELLDiscretePolicy.ELLDiscretePolicy>`.
    :type user_policy_model: ELLDiscretePolicy<coopihc.policy.ELLDiscretePolicy.ELLDiscretePolicy>`
    """

    def __init__(
        self, assistant_action_state, user_policy_model, *args, threshold=0.8, **kwargs
    ):
        self.threshold = threshold
        super().__init__(*args, action_state=assistant_action_state, **kwargs)

        self.assistant_action_set = Space.cartesian_product(
            self.action_state["action"].spaces
        )[0]

        self.user_policy_model = user_policy_model
        self.user_action_set = Space.cartesian_product(
            user_policy_model.action_state["action"].spaces
        )[0]

        self.user_policy_likelihood_function = user_policy_model.compute_likelihood

    def attach_set_theta(self, set_theta):
        self.set_theta = set_theta

    def attach_transition_function(self, trans_func):
        self.transition_function = trans_func

    def PYy_Xx(self, user_action, assistant_action, potential_states, beliefs):
        """:math:`P(Y=y|X=x)`

        Computes the conditional probability :math:`P(Y=y|X=x)`, where X is the assistant outcome and Y the user's response.

        :param user_action: user action y for which the condition is computed
        :type user_action: `StateElement<coopihc.space.StateElement.StateElement>`
        :param assistant_action: assistant action to be evaluated
        :type assistant_action: `StateElement<coopihc.space.StateElement.StateElement>`
        :param potential_states: collection of potential goal states
        :type potential_states: iterable
        :param beliefs: (list) beliefs for each target
        :type beliefs: (list) beliefs for each target

        :return: the conditional :math:`P(Y=y|X=x)`
        :rtype: list
        """
        pYy__Xx = 0
        for potential_state, belief in zip(potential_states, beliefs):
            pYy__Xx += (
                self.user_policy_likelihood_function(user_action, potential_state)
                * belief
            )
        return pYy__Xx

    def HY__Xx(self, potential_states, assistant_action, beliefs):
        """:math:`H(Y |X=x)`

        Computes the conditional entropy :math:`H(Y |X=x) = -\mathbb{E}[\log(p(Y|X=x))]`.




        :param assistant_action: assistant action to be evaluated
        :type assistant_action: `StateElement<coopihc.space.StateElement.StateElement>`
        :param potential_states: collection of potential goal states
        :type potential_states: iterable
        :param beliefs: (list) beliefs for each target
        :type beliefs: (list) beliefs for each target
        :return: :math:`H(Y |X=x)`
        :rtype: float
        """
        H = 0
        for user_action in self.user_action_set:
            pYy_Xx = self.PYy_Xx(
                user_action, assistant_action, potential_states, beliefs
            )
            if pYy_Xx != 0:
                H += -pYy_Xx * math.log(pYy_Xx, 2)
        return H

    def HY__OoXx(self, potential_states, beliefs):
        """:math:`H(Y |\Theta = \theta, X=x)`

        Computes the conditional entropy :math:`H(Y |\Theta = \theta, X=x) = -\mathbb{E}[\log(p(Y|\Theta = \theta, X=x))]`.

        :param potential_states: collection of potential goal states
        :type potential_states: iterable
        :param beliefs: (list) beliefs for each target
        :type beliefs: (list) beliefs for each target
        :return: :math:`H(Y |\Theta = \theta, X=x)`
        :rtype: float
        """
        H = 0
        for user_action in self.user_action_set:
            for potential_state, belief in zip(potential_states, beliefs):

                pYy__OoXx = self.user_policy_likelihood_function(
                    user_action, potential_state
                )

                if pYy__OoXx != 0:  # convention: 0 log 0 = 0
                    H += -belief * pYy__OoXx * math.log(pYy__OoXx, 2)

        return H

    def IG(self, assistant_action, observation, beliefs):
        """Information Gain :math:`\mathrm{IG}(X=x)`

        Computes the expected information gain :math:`\mathrm{IG}(X=x) = H(Y |X=x) - H(Y |\Theta = \theta, X=x)` for a potential assistant action x.

        :param assistant_action: assistant action to be evaluated
        :type assistant_action: `StateElement<coopihc.space.StateElement.StateElement>`
        :param observation: current assistant observation
        :type observation: `State<coopihc.space.State.State>`
        :param beliefs: (list) beliefs for each target
        :type beliefs: (list) beliefs for each target
        :return: [description]
        :rtype: [type]
        """

        observation = self.transition_function(assistant_action, observation)
        potential_states = []
        for nt, t in enumerate(self.set_theta):
            # Deepcopy would be safer, but copy should do. Deepcopy is much more expensive to produce.
            # potential_state = copy.deepcopy(observation)
            potential_state = copy.copy(observation)
            for key, value in t.items():
                try:
                    potential_state[key[0]][key[1]] = value
                except KeyError:  # key[0] is not in observation
                    _state = State()
                    _state[key[1]] = value
                    potential_state[key[0]] = _state
            potential_states.append(potential_state)

        return self.HY__Xx(potential_states, assistant_action, beliefs) - self.HY__OoXx(
            potential_states, beliefs
        )

    def find_best_action(self):
        """find_best_action

        Finds expected information gain associated with each possible future cursor position and ranks them in order from the most to less informative.



        :return: (assistant actions, associated information gain)
        :rtype: tuple(list, list)
        """

        beliefs = self.host.state["beliefs"]
        index = numpy.argmax(beliefs)
        hp = beliefs[index]
        if hp > self.threshold:
            targets = self.observation["task_state"]["targets"]
            hp_target = targets[index]
            return [hp_target], [None]
        else:
            observation = self.observation

        IG_storage = [
            self.IG(action, observation, beliefs.squeeze().tolist())
            for action in self.assistant_action_set
        ]

        _IG, action = sort_two_lists(
            IG_storage, self.assistant_action_set, lambda pair: pair[0]
        )
        action.reverse(), _IG.reverse()
        return action, _IG

    def sample(self, observation=None):
        """sample

        Choose action (select the action with highest expected information gain)

        :return: (assistant action, associated reward)
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """
        self._actions, self._IG = self.find_best_action()
        new_action = self.new_action
        new_action[:] = self._actions[0]

        return new_action, 0
