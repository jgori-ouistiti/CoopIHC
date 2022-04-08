from coopihc.base.StateElement import StateElement
from coopihc.base.Space import Numeric, CatSet

import numpy
import gym
from collections import OrderedDict
from abc import ABC, abstractmethod


class TrainGym2SB3ActionWrapper(gym.ActionWrapper):
    """TrainGym2SB3ActionWrapper

    Wrapper that flatten all spaces to boxes, using one-hot encoding for discrete spaces.

    While this wrapper will likely work for all cases, it may sometimes be more effective to code your own actionwrapper to avoid one-hot encoding.

    :param gym: [description]
    :type gym: [type]
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(action)


class TrainGym(gym.Env):
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a Wrapper to make a Bundle compatible with gym.Env. Read more on the Train class.


    :param bundle: bundle to convert to a gym.Env
    :type bundle: `Bundle <coopihc.bundle.Bundle.Bundle>`
    :param train_user: whether to train the user, defaults to True
    :type train_user: bool, optional
    :param train_assistant: whether to train the assistant, defaults to True
    :type train_assistant: bool, optional
    :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:

    .. code-block:: python

        filterdict = OrderedDict(
            {
                "user_state": OrderedDict({"goal": 0}),
                "task_state": OrderedDict({"x": 0}),
            }
        )

    You can always filter out observations later using an ObservationWrapper. Difference in performance between the two approaches is unknown.

    :type observation_dict: collections.OrderedDict, optional
    :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here if needed (see Bundle reset mechanism), defaults to {}
    :type reset_dic: dict, optional
    :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see Bundle reset_turn mechanism), defaults to None, which selects either 1 if the user is trained else 3
    :type reset_turn: int, optional
    """

    def __init__(
        self,
        bundle,
        *args,
        train_user=False,
        train_assistant=False,
        observation_dict=None,
        reset_dic={},
        reset_turn=None,
        filter_observation=None,
        **kwargs,
    ):
        self.train_user = train_user
        self.train_assistant = train_assistant
        self.bundle = bundle
        self.observation_dict = observation_dict
        self.reset_dic = reset_dic
        self.filter_observation = filter_observation

        if reset_turn is None:
            if self.train_user:
                self.reset_turn = 1
            if train_assistant:  # override reset_turn if train_assistant is True
                self.reset_turn = 3

        else:
            self.reset_turn = reset_turn

        self._convertor = GymConvertor(filter_observation=filter_observation)

        # The asymmetry of these two should be resolved. Currently, some fiddling is needed due to Issue # 58 https://github.com/jgori-ouistiti/CoopIHC/issues/58 . It is expected that when issue 58 is resolved, this code can be cleaned up.
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # Below: Using ordereddict here is forced due to Open AI gym's behavior: when initializing the Dict space, it tries to order the dict by keys, which may change the order of the dict entries. This is actually useless since Python 3.7 because dicts are ordered by default.

    def get_action_space(self):
        """get_action_space

        Create a gym.spaces.Dict out of the action states of the Bundle.

        """
        # ----- Init Action space -------
        action_dict = OrderedDict({})
        if self.train_user:
            try:
                for i, _action in enumerate(self.bundle.user.action):
                    action_dict.update(
                        {f"user_action_{i}": self.convert_space(_action)}
                    )
            except TypeError:  # Catch single actions
                action_dict.update(
                    {f"user_action": self.convert_space(self.bundle.user.action)}
                )

        if self.train_assistant:
            try:
                for i, _action in enumerate(self.bundle.assistant.action):
                    action_dict.update(
                        {f"assistant_action_{i}": self.convert_space(_action)}
                    )
            except TypeError:  # Catch single actions
                action_dict.update(
                    {
                        f"assistant_action": self.convert_space(
                            self.bundle.assistant.action
                        )
                    }
                )
        return gym.spaces.Dict(action_dict)

    def get_observation_space(self):
        """get_observation_space

        Same as get_action_space for observations.


        """
        self.bundle.reset(go_to=self.reset_turn)
        # ------- Init Observation space

        if self.train_user and self.train_assistant:
            raise NotImplementedError(
                "Currently this wrapper can not deal with simultaneous training of users and assistants."
            )

        if self.train_user:
            return self.get_agent_observation_space("user")
        if self.train_assistant:
            return self.get_agent_observation_space("assistant")

    def get_agent_observation_space(self, agent):
        observation_dict = OrderedDict({})

        observation = getattr(self.bundle, agent).observation
        if self.filter_observation is not None:
            observation = observation.filter(
                mode="stateelement", filterdict=self.filter_observation
            )

        for key, value in observation.items():
            # for key, value in getattr(self.bundle, agent).observation.items():
            if key == "user_action" or key == "assistant_action":
                observation_dict.update({key: self.convert_space(value["action"])})

            else:
                observation_dict.update(
                    {
                        __key: self.convert_space(__value)
                        for __key, __value in value.items()
                    }
                )

        return gym.spaces.Dict(observation_dict)

    def reset(self):
        self.bundle.reset(dic=self.reset_dic, go_to=self.reset_turn)
        if self.train_user and self.train_assistant:
            raise NotImplementedError
        if self.train_user:
            return self._convertor.filter_gamestate(self.bundle.user.observation)
        if self.train_assistant:
            return self._convertor.filter_gamestate(self.bundle.assistant.observation)

    def step(self, action):

        user_action = action.get("user_action", None)
        assistant_action = action.get("assistant_action", None)

        obs, rewards, flag = self.bundle.step(
            user_action=user_action, assistant_action=assistant_action
        )

        if self.train_user and self.train_assistant:
            raise NotImplementedError
        if self.train_user:
            obs = self._convertor.filter_gamestate(self.bundle.user.observation)
        if self.train_assistant:
            obs = self._convertor.filter_gamestate(self.bundle.assistant.observation)

        return (
            obs,
            float(sum(rewards.values())),
            flag,
            {"name": "CoopIHC Bundle {}".format(str(self.bundle))},
        )

    def convert_space(self, object):
        if isinstance(object, StateElement):
            object = object.space
        return self._convertor.convert_space(object)

    def render(self, mode):
        """See Bundle and gym API

        :meta public:
        """
        self.bundle.render(mode)

    def close(self):
        """See Bundle and gym API

        :meta public:
        """
        self.bundle.close()


class RLConvertor(ABC):
    """RLConvertor

    An object, who should be subclassed that helps convert spaces from Bundles to another library.

    :param interface: API target for conversion, defaults to "gym"
    :type interface: str, optional
    """

    def __init__(self, interface="gym", **kwargs):

        self.interface = interface
        if self.interface != "gym":
            raise NotImplementedError

    @abstractmethod
    def convert_space(self, space):
        pass

    @abstractmethod
    def filter_gamestate(self, gamestate, observation_mapping):
        pass


class GymConvertor(RLConvertor):
    """GymConvertor

    Convertor to convert spaces from Bundle to Gym.

    .. note::

        Code is a little messy. Refactoring together with Train and TrainGym would be beneficial.

    :param RLConvertor: [description]
    :type RLConvertor: [type]
    """

    def __init__(self, filter_observation=None, **kwargs):
        super().__init__(interface="gym", **kwargs)
        self._filter_observation = filter_observation

    def convert_space(self, space):
        if isinstance(space, Numeric):
            return gym.spaces.Box(
                low=numpy.atleast_1d(space.low),
                high=numpy.atleast_1d(space.high),
                dtype=space.dtype,
            )
        elif isinstance(space, CatSet):
            return gym.spaces.Discrete(space.N)

    def filter_gamestate(self, gamestate):
        """filter_gamestate

        converts a CoopIHC observation to a valid Gym observation


        """

        dic = OrderedDict({})
        for k, v in gamestate.filter(
            mode="array-Gym", filterdict=self._filter_observation
        ).items():
            # Hack, see Issue # 58  https://github.com/jgori-ouistiti/CoopIHC/issues/58

            try:
                _key, _value = next(iter(v.items()))
            except StopIteration:
                continue

            if k == "user_action" or k == "assistant_action":
                v[k] = v.pop("action")
                _key = k

            dic.update(v)

        return dic
