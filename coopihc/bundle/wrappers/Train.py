from coopihc.helpers import hard_flatten
from coopihc.base.convertor import GymConvertor, GymForceConvertor
from coopihc.base.elements import discrete_array_element, array_element, cat_element

import gym
import numpy
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

    ..code-block:: python

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
        **kwargs
    ):
        self.train_user = train_user
        self.train_assistant = train_assistant
        self.bundle = bundle
        self.observation_dict = observation_dict
        self.reset_dic = reset_dic

        if reset_turn is None:
            if self.train_user:
                self.reset_turn = 1
            if train_assistant:  # override reset_turn if train_assistant is True
                self.reset_turn = 3

        else:
            self.reset_turn = reset_turn

        self._convertor = GymConvertor()

        # The asymmetry of these two should be resolved. Currently, some fiddling is needed due to Issue # 58 https://github.com/jgori-ouistiti/CoopIHC/issues/58 . It is expected that when issue 58 is resolved, this code can be cleaned up.
        self.action_space = self.get_action_space()
        self.observation_space, self.observation_mapping = self.get_observation_space()

        # Below: Using ordereddict here is forced due to Open AI gym's behavior: when initializing the Dict space, it tries to order the dict by keys, which may change the order of the dict entries. This is actually useless since Python 3.7 because dicts are ordered by default.

    def get_action_space(self):
        """get_action_space

        Create a gym.spaces.Dict out of the action states of the Bundle.

        """
        # ----- Init Action space -------
        action_dict = OrderedDict({})
        if self.train_user:
            action_dict.update(
                {"user_action": self.convert_space(self.bundle.user.action)[0]}
            )
        if self.train_assistant:
            action_dict.update(
                {
                    "assistant_action": self.convert_space(
                        self.bundle.assistant.action
                    )[0]
                }
            )
        return gym.spaces.Dict(action_dict)

    def get_observation_space(self):
        """get_observation_space

        Same as get_action_space for observations.


        """
        self.bundle.reset(self.reset_turn)
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
        observation_mapping = OrderedDict({})

        observation_dict.update(
            {
                "turn_index": gym.spaces.Discrete(4),
                "round_index": gym.spaces.Discrete(
                    1000
                ),  # Maximum of N=1000 rounds. Should not affect performance and be big enough for most cases.
            }
        )
        observation_mapping.update({"turn_index": None, "round_index": None})

        for key, value in getattr(self.bundle, agent).observation.items():
            if key == "game_info":
                continue
            elif key == "user_action" or key == "assistant_action":
                observation_dict.update({key: self.convert_space(value["action"])[0]})
                observation_mapping.update(
                    {key: self.convert_space(value["action"])[1]}
                )
            else:
                observation_dict.update(
                    {
                        __key: self.convert_space(__value)[0]
                        for __key, __value in value.items()
                    }
                )
                observation_mapping.update(
                    {
                        __key: self.convert_space(__value)[1]
                        for __key, __value in value.items()
                    }
                )

        return gym.spaces.Dict(observation_dict), observation_mapping

    def reset(self):
        self.bundle.reset(dic=self.reset_dic, turn=self.reset_turn)
        if self.train_user and self.train_assistant:
            raise NotImplementedError
        if self.train_user:
            return self._convertor.filter_gamestate(
                self.bundle.user.observation, self.observation_mapping
            )
        if self.train_assistant:
            return self._convertor.filter_gamestate(
                self.bundle.assistant.observation, self.observation_mapping
            )

    def step(self, action):
        ## Hack for now
        # action = self._convertor.adapt_discrete_and_multidiscrete_action(
        #     self._convertor, action, self
        # )
        action = GymConvertor.adapt_discrete_and_multidiscrete_action(
            self._convertor, action, self
        )
        user_action = action.get("user_action", None)
        assistant_action = action.get("assistant_action", None)

        obs, rewards, flag = self.bundle.step(
            user_action=user_action, assistant_action=assistant_action
        )

        if self.train_user and self.train_assistant:
            raise NotImplementedError
        if self.train_user:
            obs = self._convertor.filter_gamestate(
                self.bundle.user.observation, self.observation_mapping
            )
        if self.train_assistant:
            obs = self._convertor.filter_gamestate(
                self.bundle.assistant.observation, self.observation_mapping
            )

        return (
            obs,
            float(sum(rewards.values())),
            flag,
            {"name": "CoopIHC Bundle {}".format(str(self.bundle))},
        )

    def convert_space(self, object):
        if isinstance(object, StateElement):
            object = object.spaces
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

    def __init__(self, **kwargs):
        super().__init__(interface="gym", **kwargs)

    def convert_space(self, space):
        """convert_space

        CoopIHC continuous spaces are simply cast to Gym boxes. CoopIHC Discrete and Multidiscrete are cast to Gym Discrete and Multidiscrete, with a transformation of the arrays (since in Gym, discrete values live in :math:`\mathbb{N}`)

        :param space: CoopIHC space to convert
        :type space: `Space <coopihc.base.Space.Space>`
        :return: (Gym space, conversion_list)
        :rtype: tuple(gym.spaces, list)
        """
        if space.space_type == "continuous":
            return (gym.spaces.Box(low=space.low, high=space.high), None)
        elif space.space_type == "discrete":
            if space.low == 0 and space.high == space.N - 1:
                return (gym.spaces.Discrete(space.N), None)
            else:
                return (gym.spaces.Discrete(space.N), space._array_bound)
        elif space.space_type == "multidiscrete":
            convert_list = []
            for _space in space:
                if _space.low == 0 and _space.high == _space.N - 1:
                    convert_list.append(None)
                else:
                    convert_list.append(_space._array_bound)
            return (gym.spaces.MultiDiscrete(space.N), convert_list)

    def filter_gamestate(self, gamestate, observation_mapping):
        """filter_gamestate

        converts a CoopIHC observation to a valid Gym observation


        """
        dic = OrderedDict({})
        for k, v in gamestate.filter(mode="array-Gym").items():
            # Hack, see Issue # 58  https://github.com/jgori-ouistiti/CoopIHC/issues/58

            try:
                _key, _value = next(iter(v.items()))
            except StopIteration:
                continue

            if k == "user_action" or k == "assistant_action":
                v[k] = v.pop("action")
                _key = k

            # Also convert discrete spaces to gym discrete spaces
            if observation_mapping[_key] is not None:
                if isinstance(observation_mapping[_key], numpy.ndarray):
                    v[_key] = int(numpy.where(observation_mapping[_key] == _value)[0])

                elif isinstance(observation_mapping[_key], list):  # Multidiscrete
                    for n, (mapping, __v) in enumerate(
                        zip(observation_mapping[_key], _value)
                    ):  # could verify here that iterables in zip are equal length. This can be done with strict = True starting from Python 3.10
                        v[_key][n] = numpy.where(observation_mapping[_key] == __v)[0][0]
            dic.update(v)
        return dic

    def adapt_discrete_and_multidiscrete_action(self, action, traingym):
        """_adapt_discrete_and_multidiscrete_action

        Transforms a Gym action to a CoopIHC valid action.

        """

        for key, value in action.items():
            if isinstance(traingym.action_space.spaces[key], gym.spaces.Box):
                pass
            elif isinstance(
                traingym.action_space.spaces[key],
                (gym.spaces.Discrete, gym.spaces.MultiDiscrete),
            ):
                atr1, atr2 = key.split("_")
                action[key] = getattr(
                    getattr(traingym.bundle, atr1), atr2
                ).spaces._array_bound[value]
        return action
