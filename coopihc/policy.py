import gym
from coopihc.space import State, StateElement, Space
import itertools
from coopihc.helpers import sort_two_lists
import numpy
import copy
import math

from tabulate import tabulate
import time
import importlib
from collections import OrderedDict
import copy

# ============== General Policies ===============


class BasePolicy:
    """Policy to subclass. Provide either an action state used for initialization, or specify action_spaces and action_sets"""

    def __init__(self, *args, **kwargs):
        # If a state is provided, use it; else create one (important not to lose the reference w/r the game_state)
        if args:
            self.action_state = args[0]
        else:
            action_state = State()
            action_state["action"] = StateElement(
                values=None,
                spaces=Space([numpy.array([None], dtype=numpy.object)]),
            )
            self.action_state = action_state
        # if kwargs:
        #     spaces = kwargs.get('action_space')
        #     if spaces is not None:
        #         self.action_state['action']['spaces'] = spaces
        #     set = kwargs.get('action_set')
        #     if set is not None:
        #         self.action_state['action']['possible_values'] = set
        #     values = kwargs.get('action_values')
        #     if values is not None:
        #         self.action_state['action']['values'] = values
        #     clipping_mode = kwargs.get('clipping_mode')
        #     if clipping_mode is not None:
        #         self.action_state['action']['clipping_mode'] = clipping_mode

        self.host = None

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
        return self.action_state["action"]

    @property
    def new_action(self):
        return copy.copy(self.action_state["action"])

    @property
    def unwrapped(self):
        return self

    def reset(self):
        pass

    def sample(self):
        self.action.reset()
        return self.action, 0


class LinearFeedback(BasePolicy):
    def __init__(
        self,
        state_indicator,
        index,
        action_state,
        *args,
        feedback_gain="identity",
        **kwargs
    ):
        super().__init__(action_state, *args, **kwargs)
        self.state_indicator = state_indicator
        self.index = index
        self.noise_function = kwargs.get("noise_function")
        # bind the noise function
        if self.noise_function is not None:
            self._bind(self.noise_function, as_name="noise_function")

        self.noise_args = kwargs.get("noise_function_args")
        self.feedback_gain = feedback_gain

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
            if self.feedback_gain == "identity":
                self.feedback_gain = -numpy.eye(max(substate["values"][0].shape))

        noiseless_feedback = -self.feedback_gain @ substate.reshape((-1, 1))
        noise = self.noise_function(
            noiseless_feedback, self.observation, *self.noise_args
        )
        action = self.action
        action["values"] = noiseless_feedback + noise.reshape((-1, 1))
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
        return {
            "Name": self.__class__.__name__,
            "Bundle": self.bundle.__content__(),
        }

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
        return "{} <[ {} ]>".format(self.__class__.__name__, self.bundle.__str__())

    def __repr__(self):
        return self.__str__()


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


# ----------- Explicit Likelihood Discrete Policy


class BadlyDefinedLikelihoodError(Exception):
    pass


class ELLDiscretePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        action_state = kwargs.pop("action_state")
        super().__init__(action_state, *args, **kwargs)
        self.explicit_likelihood = True
        self.rng = numpy.random.default_rng(kwargs.get("seed"))

    @classmethod
    def attach_likelihood_function(cls, _function):
        cls.compute_likelihood = _function

    def sample(self):
        """Select the most likely action.

        :param observation: (OrderedDict)
        :return: action. most likely action.
        """

        observation = self.host.inference_engine.buffer[-1]
        actions, llh = self.forward_summary(observation)
        action = actions[self.rng.choice(len(llh), p=llh)]
        self.action_state["action"] = action
        return action, 0

    def forward_summary(self, observation):
        """Compute the likelihood of each action, given the current observation

        :param observation: (OrderedDict) user observation

        :return: actions, likelihood. list of actions and associated likelihoods

        :meta public:
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


# ======================= RL Policy


class RLPolicy(BasePolicy):
    """Code works as proof of concept, but should be tested and augmented to deal with arbitrary wrappers. Possibly the wrapper class should be augmented with a reverse method, or something like that."""

    def __init__(self, *args, **kwargs):
        self.role = args[0]
        model_path = kwargs.get("model_path")
        learning_algorithm = kwargs.get("learning_algorithm")
        library = kwargs.get("library")
        self.training_env = kwargs.get("training_env")
        self.wrappers = kwargs.get("wrappers")

        if library != "stable_baselines3":
            raise NotImplementedError(
                "The Reinforcement Learning Policy currently only supports policies obtained via stables baselines 3."
            )
        import stable_baselines3

        learning_algorithm = getattr(stable_baselines3, learning_algorithm)
        self.model = learning_algorithm.load(model_path)

        # Recovering action space
        action_state = State()
        action_state["action"] = copy.deepcopy(
            getattr(
                getattr(
                    getattr(self.training_env.unwrapped.bundle, "user"),
                    "policy",
                ),
                "action_state",
            )["action"]
        )

        super().__init__(action_state, *args, **kwargs)

    def sample(self):
        # observation = self.host.inference_engine.buffer[-1]
        observation = self.observation
        nn_obs = self.training_env.unwrapped.convert_observation(observation)
        _action = self.model.predict(nn_obs)[0]
        for wrappers_name, (_cls, _args) in reversed(
            self.wrappers["actionwrappers"].items()
        ):
            aw = _cls(self.training_env.unwrapped, *_args)
            _action = aw.action(_action)
        action = self.action_state["action"]
        action["values"] = _action
        return action, 0


# ================= Examples ==============
class ExamplePolicy(BasePolicy):
    """A simple policy which assumes that the agent using it has a goal state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        if (
            self.observation["task_state"]["x"]
            < self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = 1
        elif (
            self.observation["task_state"]["x"]
            > self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = -1
        else:
            _action_value = 0

        new_action = self.new_action["values"] = numpy.array(_action_value)
        return new_action, 0