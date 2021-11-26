from coopihc.space import State
from coopihc.helpers import sort_two_lists
import copy
import math
import copy

from .BasePolicy import BasePolicy

# ============= Discrete Policies =================

# ----------- Bayesian Information Gain Policy ---------


class BIGDiscretePolicy(BasePolicy):
    def __init__(self, assistant_action_state, user_policy_model):

        super().__init__(assistant_action_state)

        self.assistant_action_set = self.action_state["action"].cartesian_product()
        self.user_policy_model = user_policy_model
        self.user_action_set = user_policy_model.action_state[
            "action"
        ].cartesian_product()
        self.user_policy_likelihood_function = user_policy_model.compute_likelihood

    def attach_set_theta(self, set_theta):
        self.set_theta = set_theta

    def attach_transition_function(self, trans_func):
        self.transition_function = trans_func

    #
    # def generate_candidate_next_state(self, observation, assistant_action):
    #     print(observation, assistant_action)
    #     return candidate_next_state
    #
    # def generate_candidate_next_observation(self, candidate_next_state):
    #     # do something
    #     return observation

    def PYy_Xx(self, user_action, assistant_action, potential_states, beliefs):
        r"""Compute the conditional probability :math:`P(Y=y|X=x)`

        :param user_action: given user action y for which the condition is computed
        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the conditional :math:`P(Y=y|X=x)`

        :meta public:
        """
        pYy__Xx = 0
        for potential_state, belief in zip(potential_states, beliefs):
            pYy__Xx += (
                self.user_policy_likelihood_function(user_action, potential_state)
                * belief
            )
        return pYy__Xx

    def HY__Xx(self, potential_states, assistant_action, beliefs):
        r"""Computes the conditional entropy :math:`H(Y |X=x) = -\mathbb{E}[\log(p(Y|X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |X=x)`

        :meta public:
        """
        H = 0
        for user_action in self.user_action_set:
            pYy_Xx = self.PYy_Xx(
                user_action, assistant_action, potential_states, beliefs
            )
            if pYy_Xx != 0:
                H += -pYy_Xx * math.log(pYy_Xx, 2)
        return H

    def HY__OoXx(self, potential_states, assistant_action, beliefs):
        r"""Computes the conditional entropy :math:`H(Y |\Theta = \theta, X=x) = -\mathbb{E}[\log(p(Y|\Theta = \theta, X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |\Theta = \theta, X=x)`

        :meta public:
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
        r"""Computes the expected information gain :math:`\mathrm{IG}(X=x) = H(Y |X=x) - H(Y |\Theta = \theta, X=x)` for a future position.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the information gain  :math:`\mathrm{IG}(X=x)`

        :meta public:
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
            potential_states, assistant_action, beliefs
        )

    def find_best_action(self):
        """Finds expected information gain associated with each possible future cursor position and ranks them in order from the most to  less informative.

        :return: pos, IG. Future cursor position and associated expected information gain.

        :meta public:
        """
        beliefs = self.host.state["beliefs"]["values"][0].squeeze().tolist()
        # hp, hp_target = max(beliefs), targets[beliefs.index(max(beliefs))]
        # if hp > self.threshold:
        #     return [hp_target], [None]
        # else:
        observation = self.host.inference_engine.buffer[-1]

        IG_storage = [
            self.IG(action, observation, beliefs)
            for action in self.assistant_action_set
        ]

        _IG, action = sort_two_lists(
            IG_storage, self.assistant_action_set, lambda pair: pair[0]
        )
        action.reverse(), _IG.reverse()
        return action, _IG

    def sample(self):
        self._actions, self._IG = self.find_best_action()
        # logger.info('Actions and associated expected information gain:\n{}'.format(tabulate(list(zip(self._actions['values'], self._IG)), headers = ['action', 'Expected Information Gain']) ))
        return self._actions[0], 0
