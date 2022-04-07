import numpy
import copy

from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.elements import cat_element


# ============== General Policies ===============


class BasePolicy:
    """BasePolicy

    Base Policy class. Randomly samples from the action state. You have can provide an action state as an argument (args[0]). If no action state is provided, the policy is initialized with an action state with a single 'None' action.
    """

    def __init__(self, *args, action_state=None, **kwargs):

        self._action_keys = None  # For actionkeys property

        # If a state is provided, use it; else create one (important not to lose the reference w/r the game_state)

        if action_state is None:
            action_state = State()
            action_state["action"] = cat_element(N=2, init=0)

        self.action_state = action_state
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
    def parameters(self):
        try:
            return self.host.parameters
        except AttributeError:
            raise AttributeError(
                "This policy has not been connected to an agent yet -- You can't access this agent's parameters"
            )

    @property
    def state(self):
        try:
            return self.host.state
        except AttributeError:
            raise AttributeError(
                "This policy has not been connected to an agent yet -- You can't access this agent's state"
            )

    @property
    def observation(self):
        """observation

        Return the last observation.

        :return: last observation
        :rtype: `State<coopihc.base.State.State>`
        """
        try:
            return self.host.observation
        except AttributeError:
            raise AttributeError(
                "This policy has not been connected to an agent yet -- You can't access this agent's observation"
            )

    @property
    def action_keys(self):
        if self._action_keys is None:
            self._action_keys = self.action_state.keys()

        return self._action_keys

    @property
    def action(self):
        """action

        Return the last action.

        :return: last action
        :rtype: `State<coopihc.base.StateElement.StateElement>`
        """
        actions = tuple(self.action_state.values())
        if len(actions) == 1:
            return next(iter(actions))
        return actions

    @action.setter
    def action(self, item):
        try:
            next(iter(item))
        except TypeError:
            item = (item,)
        for _action, key in zip(item, self.action_keys):
            self.action_state[key][...] = _action

    @property
    def unwrapped(self):
        return self

    def default_value(func):
        """Apply this decorator to use bundle.game_state as default value to observe if game_state = None"""

        def wrapper_default_value(self, agent_observation=None, agent_state=None):
            if agent_observation is None:
                agent_observation = self.host.observation
            if agent_state is None:
                agent_state = self.state
            return func(
                self, agent_observation=agent_observation, agent_state=agent_state
            )

        return wrapper_default_value

    def reset(self, random=True):
        """reset

        Reset the policy

        :param random: reset the policy, defaults to True. Here in case of subclassing BasePolicy.
        :type random: bool, optional
        """
        if random:
            self.action_state.reset()

    def _base_sample(self, agent_observation=None, agent_state=None):
        action, reward = self.sample(
            agent_observation=agent_observation, agent_state=agent_state
        )
        self.action = action
        return self.action, reward

    @default_value
    def sample(self, agent_observation=None, agent_state=None):
        """sample

        (Randomly) Sample from the policy

        :return: (action, action reward)
        :rtype: (StateElement<coopihc.base.StateElement.StateElement>, float)
        """
        try:
            _ = [_action.reset() for _action in self.action]
        except TypeError:
            self.action.reset()
        return self.action, 0

    def __repr__(self):
        try:
            return self.action_state.__str__()
        except AttributeError:
            return "Policy--unreadable"
