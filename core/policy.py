import gym
from core.space import State, StateElement
import itertools
from core.helpers import sort_two_lists
import numpy
import copy
import math

import time

class Policy:
    """Policy to subclass. Provide either an action state used for initialization, or specify action_spaces and action_sets
    """
    def __init__(self, *args, **kwargs):
        # If a state is provided, use it; else create one (important not to lose the reference w/r the game_state)
        if args:
            self.action_state = args[0]
        else:
            action_state = State()
            action_state['action'] = StateElement()
            self.action_state = action_state
        if kwargs:
            spaces = kwargs.get('action_space')
            if spaces is not None:
                self.action_state['action']['spaces'] = spaces
            set = kwargs.get('action_set')
            if set is not None:
                self.action_state['action']['possible_values'] = set
            values = kwargs.get('action_values')
            if values is not None:
                self.action_state['action']['values'] = values

        self.host = None



    def reset(self):
        pass

    def sample(self):
        return StateElement(values = [u.sample() for u in self.action_state['action'].spaces], spaces = self.action_state['action'].spaces, possible_values = self.action_state['action'].possible_values)


# class DummyPolicy:
#     def __init__(self):
#         super().__init__()
#
#     def reset(self):
#         pass
#
#     def sample(self):
#         return [u.sample() for u in self.action_state[0][1]]

# ============= Discrete Policies =================

# ----------- Bayesian Information Gain Policy ---------
class BIGDiscretePolicy(Policy):
    def __init__(self, action_state, assistant_action_space, assistant_action_possible_values, operator_policy_model):
        super().__init__(action_state, action_space = assistant_action_space, action_set = assistant_action_possible_values)

        self.assistant_action_set = self.action_state['action'].cartesian_product()
        self.operator_policy_model = operator_policy_model
        self.operator_action_set = operator_policy_model.action_state['action'].cartesian_product()
        self.operator_policy_likelihood_function = operator_policy_model.compute_likelihood

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


    def PYy_Xx(self, operator_action, assistant_action, potential_states, beliefs):
        r""" Compute the conditional probability :math:`P(Y=y|X=x)`

        :param operator_action: given operator action y for which the condition is computed
        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the conditional :math:`P(Y=y|X=x)`

        :meta public:
        """
        pYy__Xx = 0
        for potential_state, belief in zip(potential_states, beliefs):
            pYy__Xx += self.operator_policy_likelihood_function(operator_action, potential_state)*belief
        return pYy__Xx


    def HY__Xx(self, potential_states, assistant_action, beliefs):
        r""" Computes the conditional entropy :math:`H(Y |X=x) = -\mathbb{E}[\log(p(Y|X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |X=x)`

        :meta public:
        """
        H = 0
        for operator_action in self.operator_action_set:
            pYy_Xx = self.PYy_Xx(operator_action, assistant_action, potential_states, beliefs)
            if pYy_Xx != 0:
                H += -pYy_Xx * math.log(pYy_Xx,2)
        return H

    def HY__OoXx(self, potential_states, assistant_action, beliefs):
        r""" Computes the conditional entropy :math:`H(Y |\Theta = \theta, X=x) = -\mathbb{E}[\log(p(Y|\Theta = \theta, X=x))]`.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: The conditional entropy :math:`H(Y |\Theta = \theta, X=x)`

        :meta public:
        """
        H = 0
        for operator_action in self.operator_action_set:
            for potential_state, belief in zip(potential_states, beliefs):
                pYy__OoXx = self.operator_policy_likelihood_function(operator_action, potential_state)
                if pYy__OoXx != 0: # convention: 0 log 0 = 0
                    H += -belief*pYy__OoXx*math.log(pYy__OoXx,2)
        return H





    def IG(self, assistant_action, observation, beliefs):
        r""" Computes the expected information gain :math:`\mathrm{IG}(X=x) = H(Y |X=x) - H(Y |\Theta = \theta, X=x)` for a future position.

        :param position: the future position
        :param targets: (list) possible targets
        :param beliefs: (list) priors for each target

        :return: the information gain  :math:`\mathrm{IG}(X=x)`

        :meta public:
        """
        observation = self.transition_function(assistant_action, observation)
        potential_states = []
        t0 = time.time()
        for nt,t in enumerate(self.set_theta):
            # potential_state = observation
            potential_state = copy.deepcopy(observation)
            for key, value in t.items():
                try:
                    potential_state[key[0]][key[1]] = value
                except KeyError: # key[0] is not in observation
                    _state = State()
                    _state[key[1]] = value
                    potential_state[key[0]] = _state
            potential_states.append(potential_state)

        t1 = time.time()
        print("elapsed time")
        print(t1-t0)

        return self.HY__Xx(potential_states, assistant_action, beliefs) - self.HY__OoXx(potential_states, assistant_action, beliefs)

    def find_best_action(self):
        """ Finds expected information gain associated with each possible future cursor position and ranks them in order from the most to  less informative.

        :return: pos, IG. Future cursor position and associated expected information gain.

        :meta public:
        """
        # targets = self.targets
        beliefs = self.host.state['Beliefs']['values']
        # hp, hp_target = max(beliefs), targets[beliefs.index(max(beliefs))]
        # if hp > self.threshold:
        #     return [hp_target], [None]
        # else:
        observation = self.host.inference_engine.buffer[-1]

        IG_storage = [self.IG(action, observation, beliefs) for action in self.assistant_action_set]

        # IG_storage = [self.IG( pos, targets, beliefs) for pos in range(self.bundle.task.state['Gridsize'][0])]
        _IG, action = sort_two_lists(IG_storage, self.assistant_action_set, lambda pair: pair[0])
        action.reverse(), _IG.reverse()
        return action, _IG

    def sample(self):
        self._actions, self._IG = self.find_best_action()
        return self._actions[0]


# ----------- Explicit Likelihood Discrete Policy
class ELLDiscretePolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_likelihood = True

    @classmethod
    def attach_likelihood_function(cls, _function):
        cls.compute_likelihood = _function
        print('Attached {} to {}'.format(_function, cls.__name__))

    def sample(self):
        """ Select the most likely action.

        :param observation: (OrderedDict)
        :return: action. most likely action.
        """

        observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[numpy.random.choice(len(llh), p = llh)]
        self.action_state['action'] = action
        return action

    def forward_summary(self, observation):
        """ Compute the likelihood of each action, given the current observation

        :param observation: (OrderedDict) operator observation

        :return: actions, likelihood. list of actions and associated likelihoods

        :meta public:
        """
        llh, actions = [], []
        for action in self.action_state["action"].cartesian_product():
            llh.append(self.compute_likelihood(action, observation))
            actions.append(action)
        return actions, llh

        # llh = []
        # actions = []
        # values, spaces, possible_values = self.action_state['action']
        #
        # if possible_values is None:
        #     human_value_elements = itertools.product(*[list(range(s.n)) for ns,s in enumerate(spaces)])
        #     elements = human_value_elements
        # else:
        #     elements = list(itertools.product(*[list(range(s.n)) for ns,s in enumerate(spaces)]))
        #     human_value_elements = list(itertools.product(*[list(range(s.n)) if None in possible_values[ns] else possible_values[ns] for ns,s in enumerate(spaces)]))
        #
        #
        # for e, hve in zip(elements, human_value_elements):
        #     llh.append(self.compute_likelihood(hve, observation))
        #     actions.append(list(e))
        #
        # return actions, llh


class LinearFeedback(Policy):
    """ host_state_key = tuple(substate, subsubstate) e.g. ('task_state', 'x')
    """
    def __init__(self, host_state_key, action_space, action_set = None, **kwargs):

        self.noise = kwargs.get('noise')
        self.Gamma = kwargs.get('Gamma')

        action_state = State()
        action_state['action'] = [None, action_space, action_set]
        self.action_state = action_state
        self.host = None
        self.feedback_gain = None
        self.host_state_key = host_state_key

    def sample(self):
        if self.feedback_gain is None:
            raise ValueError('{} attribute "feedback_gain" is None. You have to set the feedback gain on this policy before using it.')

        noiseless_feedback = - self.feedback_gain @ self.host.inference_engine.buffer[-1][self.host_state_key[0]]['_value_{}'.format(self.host_state_key[1])]

        if self.noise and self.Gamma:
            gamma = numpy.random.normal(0, numpy.sqrt(self.host.timestep))
            return noiseless_feedback + self.Gamma * gamma
        else:
            return noiseless_feedback


class BundlePolicy(Policy):
    """
    """
    def __init__(self, bundle, *args):
        super().__init__(None, None)
        self.bundle = bundle
        self.substate_list = args
        substate = bundle.game_state
        for arg in args:
            substate = substate[arg]
        self.action_state['action'] = substate
        self.host = None

    def sample(self, nsteps, reset):
        if reset:
            self.reset(reset)
            print(self.bundle.game_state)

        if nsteps == 'end':
            total_rewards = []
            while True:
                observation, rewards, is_done, breakdown_rewards = self.bundle.step()
                total_rewards += [rewards]
                if is_done:
                    break



        elif isinstance(nsteps, int):
            print("passing here")
            total_rewards = []
            for i in range(nsteps):
                observation, rewards, is_done, breakdown_rewards = self.bundle.step()
                total_rewards += [rewards]
                if is_done:
                    break

        else:
            raise NotImplementedError('nsteps should be either "end" or an interfer specifying the number of steps')

        substate = self.bundle.game_state
        print(substate)
        for arg in self.substate_list:
            substate = substate[arg]
        self.action_state['action'] = substate

    def reset(self, dic):
        self.bundle.reset(dic)
