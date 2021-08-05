import gym
from core.space import State, StateElement
import itertools
from core.helpers import sort_two_lists
import numpy
import copy
import math

from loguru import logger
from tabulate import tabulate
from core.core import Handbook
import time
import importlib
from collections import OrderedDict

# ============== General Policies ===============

class BasePolicy:
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
            clipping_mode = kwargs.get('clipping_mode')
            if clipping_mode is not None:
                self.action_state['action']['clipping_mode'] = clipping_mode


        self.host = None
        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})

    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    def _bind(self, func, as_name=None):
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(self, self.__class__)
        setattr(self, as_name, bound_method)
        return bound_method

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        return self.action_state['action']

    @property
    def unwrapped(self):
        return self

    def reset(self):
        pass

    def sample(self):
        return  StateElement(
            values = [u.sample() for u in self.action_state['action'].spaces],
            spaces = self.action_state['action'].spaces,
            possible_values = self.action_state['action'].possible_values,
            clipping_mode = self.action_state['action'].clipping_mode), 0


class LinearFeedback(BasePolicy):
    def __init__(self, state_indicator, index, action_state, *args, feedback_gain = 'identity', **kwargs):
        super().__init__(action_state, *args, **kwargs)
        self.state_indicator = state_indicator
        self.index = index
        self.noise_function = kwargs.get('noise_function')
        # bind the noise function
        if self.noise_function is not None:
            self._bind(self.noise_function, as_name = 'noise_function')

        self.noise_args = kwargs.get('noise_function_args')
        self.feedback_gain = feedback_gain




        _state_indicator = {"name": 'state_indicator', "value": state_indicator, "meaning": 'Indicates which substate of the internal state will be used as action'}
        _index =  {"name": 'index', "value": index, "meaning": 'Index that is applied to the state_indicator'}
        _feedback_gain =  {"name": 'feedback_gain', "value": feedback_gain, "meaning": 'Gain (K) applied to the selected substate: output of sample = -K @ state[state_indicator][index].'}
        _noise_function = {"name": 'noise_function', "value": self.noise_function.__name__, "meaning": 'A function that is applied to the action to produce noisy actions. The signature of the function is "def noise_function(self, action, observation, *args):" and it should return a noise vector'}
        _noise_args = {"name": 'noise_args', "value": self.noise_args, "meaning": 'args that need to be supplied to the noise function'}

        self.handbook['parameters'].extend([_state_indicator, _index, _feedback_gain, _noise_function, _noise_args])

    def set_feedback_gain(self, gain):
        self.feedback_gain = gain


    def sample(self):
        if isinstance(self.index, list):
            raise NotImplementedError
        substate = self.observation
        for key in self.state_indicator:
            substate = substate[key]
        substate = substate[self.index]

        if isinstance(self.feedback_gain, str):
            if self.feedback_gain == 'identity':
                self.feedback_gain = -numpy.eye(max(substate['values'][0].shape))

        noiseless_feedback = - self.feedback_gain @ substate
        noise = self.noise_function(noiseless_feedback, self.observation, *self.noise_args)
        action = noiseless_feedback + noise
        # if not hasattr(noise, '__iter__'):
        #     noise = [noise]
        # header = ['action', 'noiseless', 'noise']
        # rows = [action, noiseless_feedback , noise]
        # logger.info('Policy {} selected action\n{})'.format(self.__class__.__name__, tabulate(rows, header) ))
        return action, 0

class WrapAsPolicy(BasePolicy):
    def __init__(self, action_bundle, action_state, *args, **kwargs):
        super().__init__(action_state, *args, **kwargs)
        self.bundle = action_bundle

    def __content__(self):
        return { 'Name': self.__class__.__name__, 'Bundle': self.bundle.__content__() }
    @property
    def unwrapped(self):
        return self.bundle.unwrapped

    @property
    def game_state(self):
        return self.bundle.game_state

    def reset(self, *args, **kwargs):
        return self.bundle.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.bundle.step(*args, **kwargs)

    def sample(self):
        pass
        # Do something
        # return action, rewards

    def __str__(self):
        return '{} <[ {} ]>'.format(self.__class__.__name__, self.bundle.__str__())

    def __repr__(self):
        return self.__str__()

# ============= Discrete Policies =================

# ----------- Bayesian Information Gain Policy ---------
class BIGDiscretePolicy(BasePolicy):
    def __init__(self, action_state, assistant_action_space, operator_policy_model):
        assistant_action_possible_values = list(range(assistant_action_space[0].n))
        super().__init__(action_state, action_space = assistant_action_space, action_set = assistant_action_possible_values)

        self.assistant_action_set = self.action_state['action'].cartesian_product()
        self.operator_policy_model = operator_policy_model
        self.operator_action_set = operator_policy_model.action_state['action'].cartesian_product()
        self.operator_policy_likelihood_function = operator_policy_model.compute_likelihood


        _action_state = {"name": 'action_state', "value": action_state, "meaning": 'Reference to the action_state, usually from the game_state'}
        _assistant_action_space =  {"name": 'assistant_action_space', "value": assistant_action_space, "meaning": 'The space in which the assistant can take actions'}
        _operator_policy_model = {"name": 'operator_policy_model', "value": operator_policy_model, "meaning": 'An instance of a Policy class (or Subclass)'}

        self.handbook['render_mode'].extend(['plot', 'text'])
        self.handbook['parameters'].extend([_action_state, _assistant_action_space, _operator_policy_model])


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
        for nt,t in enumerate(self.set_theta):
            # Deepcopy would be safer, but copy should do. Deepcopy is much more expensive to produce.
            # potential_state = copy.deepcopy(observation)
            potential_state = copy.copy(observation)
            for key, value in t.items():
                try:
                    potential_state[key[0]][key[1]] = value
                except KeyError: # key[0] is not in observation
                    _state = State()
                    _state[key[1]] = value
                    potential_state[key[0]] = _state
            potential_states.append(potential_state)


        return self.HY__Xx(potential_states, assistant_action, beliefs) - self.HY__OoXx(potential_states, assistant_action, beliefs)

    def find_best_action(self):
        """ Finds expected information gain associated with each possible future cursor position and ranks them in order from the most to  less informative.

        :return: pos, IG. Future cursor position and associated expected information gain.

        :meta public:
        """
        beliefs = self.host.state['Beliefs']['values']
        # hp, hp_target = max(beliefs), targets[beliefs.index(max(beliefs))]
        # if hp > self.threshold:
        #     return [hp_target], [None]
        # else:
        observation = self.host.inference_engine.buffer[-1]

        IG_storage = [self.IG(action, observation, beliefs) for action in self.assistant_action_set]

        _IG, action = sort_two_lists(IG_storage, self.assistant_action_set, lambda pair: pair[0])
        action.reverse(), _IG.reverse()
        return action, _IG

    def sample(self):
        self._actions, self._IG = self.find_best_action()
        # logger.info('Actions and associated expected information gain:\n{}'.format(tabulate(list(zip(self._actions['values'], self._IG)), headers = ['action', 'Expected Information Gain']) ))
        return self._actions[0], 0


# ----------- Explicit Likelihood Discrete Policy

class BadlyDefinedLikelihoodError(Exception):
    pass

class ELLDiscretePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_likelihood = True

    @classmethod
    def attach_likelihood_function(cls, _function):
        cls.compute_likelihood = _function
        logger.info('Attached {} to {}'.format(_function, cls.__name__))

    def sample(self):
        """ Select the most likely action.

        :param observation: (OrderedDict)
        :return: action. most likely action.
        """

        observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[numpy.random.choice(len(llh), p = llh)]
        self.action_state['action'] = action
        return action, 0

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
        if sum(llh) != 1:
            print('Warning, llh does not sum to 1')
            print(llh)
            print(observation)
        return actions, llh








# ======================= Continuous Policies

class RLPolicy(BasePolicy):
    """ Code works as proof of concept, but should be tested and augmented to deal with arbitrary wrappers. Possibly the wrapper class should be augmented with a reverse method, or something like that.

    """
    def __init__(self, *args, **kwargs):
        self.role = args[0]
        model_path = kwargs.get('model_path')
        learning_algorithm = kwargs.get('learning_algorithm')
        library = kwargs.get('library')
        self.training_env = kwargs.get('training_env')
        self.wrappers = kwargs.get('wrappers')



        if library != 'stable_baselines3':
            raise NotImplementedError('The Reinforcement Learning Policy currently only supports policies obtained via stables baselines 3.')
        import stable_baselines3
        learning_algorithm = getattr(stable_baselines3, learning_algorithm)
        self.model = learning_algorithm.load(model_path)


        # Recovering action space
        action_state = State()
        action_state['action'] = copy.deepcopy(getattr(getattr(getattr(self.training_env.unwrapped.bundle, 'operator'), 'policy'), 'action_state')['action'])

        super().__init__(action_state, *args, **kwargs)

    def sample(self):
        # observation = self.host.inference_engine.buffer[-1]
        observation = self.observation
        nn_obs = self.training_env.unwrapped.convert_observation(observation)
        _action = self.model.predict(nn_obs)[0]
        for wrappers_name, (_cls, _args) in reversed(self.wrappers['actionwrappers'].items()):
            aw = _cls(self.training_env.unwrapped, *_args)
            _action = aw.action(_action)
        action = self.action_state['action']
        action['values'] = _action
        return action, 0
