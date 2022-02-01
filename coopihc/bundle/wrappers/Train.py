from coopihc.helpers import hard_flatten
from coopihc.space.utils import GymConvertor, GymForceConvertor
from coopihc.space.StateElement import StateElement

import gym
import numpy
from collections import OrderedDict
from abc import ABC, abstractmethod


class TrainGym2SB3ActionWrapper(gym.ActionWrapper):
    def __init__(self, env_action_dict):
        super().__init__(env_action_dict)

        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(action)


class TrainGym(gym.Env):
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a generic Wrapper to make bundles compatibles with gym.Env. Read more on the Train class.

    :param bundle: bundle to wrap
    :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
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

    :type observation_dict: collections.OrderedDict, optional
    :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
    :type reset_dic: dict, optional
    :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
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
        reset_turn=0,
        **kwargs
    ):

        self.train_user = train_user
        self.train_assistant = train_assistant
        self.bundle = bundle
        self.observation_dict = observation_dict
        self.reset_dic = reset_dic
        self.reset_turn = reset_turn

        self._convertor = GymConvertor()

        # The asymmetry of these two should be resolved. Currently, some fiddling is needed due to Issue # 58 https://github.com/jgori-ouistiti/CoopIHC/issues/58 . It is expected that when issue 58 is resolved, this code can be cleaned up.
        self.action_space = self.get_action_space()
        self.observation_space, self.observation_mapping = self.get_observation_space()

        # Below: Using ordereddict here is forced due to Open AI gym's behavior: when initializing the Dict space, it tries to order the dict by keys, which may change the order of the dict entries. This is actually useless since Python 3.7 because dicts are ordered by default.

    def get_action_space(self):
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
        # ------- Init Observation space
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
        observation_dict.update(
            {
                key: self.convert_space(value)[0]
                for key, value in self.bundle.task.state.items()
            }
        )
        observation_mapping.update(
            {
                key: self.convert_space(value)[1]
                for key, value in self.bundle.task.state.items()
            }
        )

        if self.bundle.user.state:
            observation_dict.update(
                {
                    key: self.convert_space(value)[0]
                    for key, value in self.bundle.user.state.items()
                }
            )
            observation_mapping.update(
                {
                    key: self.convert_space(value)[1]
                    for key, value in self.bundle.user.state.items()
                }
            )
        if self.bundle.assistant.state:
            observation_dict.update(
                {
                    key: self.convert_space(value)[0]
                    for key, value in self.bundle.assistant.state.items()
                }
            )
            observation_mapping.update(
                {
                    key: self.convert_space(value)[1]
                    for key, value in self.bundle.assistant.state.items()
                }
            )

        observation_dict.update(
            {"user_action": self.convert_space(self.bundle.user.action)[0]}
        )
        observation_mapping.update(
            {"user_action": self.convert_space(self.bundle.user.action)[1]}
        )
        observation_dict.update(
            {"assistant_action": self.convert_space(self.bundle.assistant.action)[0]}
        )
        observation_mapping.update(
            {"assistant_action": self.convert_space(self.bundle.assistant.action)[1]}
        )
        return gym.spaces.Dict(observation_dict), observation_mapping

    def reset(self):
        game_state = self.bundle.reset(dic=self.reset_dic, turn=self.reset_turn)
        return self._filter_bundle_gamestate(game_state)

    def step(self, action):
        action = self._adapt_discrete_and_multidiscrete_action(action)
        user_action = action.get("user_action", None)
        assistant_action = action.get("assistant_action", None)
        obs, rewards, flag = self.bundle.step(
            user_action=user_action, assistant_action=assistant_action
        )
        obs = self._filter_bundle_gamestate(obs)

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

    def _filter_bundle_gamestate(self, gamestate):
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
            if self.observation_mapping[_key] is not None:
                if isinstance(self.observation_mapping[_key], numpy.ndarray):
                    v[_key] = int(
                        numpy.where(self.observation_mapping[_key] == _value)[0]
                    )

                elif isinstance(self.observation_mapping[_key], list):  # Multidiscrete
                    for n, (mapping, __v) in enumerate(
                        zip(self.observation_mapping[_key], _value)
                    ):  # could verify here that iterables in zip are equal length. This can be done with strict = True starting from Python 3.10
                        v[_key][n] = numpy.where(self.observation_mapping[_key] == __v)[
                            0
                        ][0]
            dic.update(v)
        return dic

    def _adapt_discrete_and_multidiscrete_action(self, action):
        """_adapt_discrete_and_multidiscrete_action

        Could maybe be rewritten as an actionwrapper

        :param action: [description]
        :type action: [type]
        :return: [description]
        :rtype: [type]
        """
        for key, value in action.items():
            if isinstance(self.action_space.spaces[key], gym.spaces.Box):
                pass
            elif isinstance(
                self.action_space.spaces[key],
                (gym.spaces.Discrete, gym.spaces.MultiDiscrete),
            ):
                atr1, atr2 = key.split("_")
                action[key] = getattr(
                    getattr(self.bundle, atr1), atr2
                ).spaces._array_bound[value]
        return action


# class Train(ABC):
#     """Generic Wrapper to make bundles compatibles with gym.Env

#     This is a generic Wrapper to make bundles compatibles with gym.Env. It is mainly here to be subclassed by other wrappers.

#     Depending on the spaces you are using, you might need to provide a wrapper to accustom the fact that coopihc spaces can take any values whereas e.g. gym discrete spaces have to be unit-spaced values.

#     .. note::

#         Experimental: This class automatically build wrappers to account for the transformation between a bundle and an env, but we offer no guarantees that it will work in all cases. It might also likely be faster (computationnally) to hard code your own wrappers.


#     :param bundle: bundle to wrap
#     :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
#     :param train_user: whether to train the user, defaults to True
#     :type train_user: bool, optional
#     :param train_assistant: whether to train the assistant, defaults to True
#     :type train_assistant: bool, optional
#     :param API: API with which the bundle will be made compatible for, defaults to "gym-force". In gym force, a limited gym compatible environment is created, which casts everything to float32 and boxes.
#     :type API: str, optional
#     :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:
#     :type observation_dict: collections.OrderedDict, optional

#     ..code-block:: python

#         filterdict = OrderedDict(
#             {
#                 "user_state": OrderedDict({"goal": 0}),
#                 "task_state": OrderedDict({"x": 0}),
#             }
#         )

#     :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
#     :type reset_dic: dict, optional
#     :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
#     :type reset_turn: int, optional
#     """

#     def __init__(
#         self,
#         bundle,
#         *args,
#         train_user=False,
#         train_assistant=False,
#         observation_dict=None,
#         reset_dic={},
#         API="gym",
#         reset_turn=0,
#         **kwargs
#     ):
#         self.train_user = train_user
#         self.train_assistant = train_assistant
#         self.bundle = bundle
#         self.observation_dict = observation_dict
#         self.reset_dic = reset_dic
#         self.reset_turn = reset_turn

#         self.action_space = self.get_action_space()
#         self.observation_space = self.get_observation_space()

#     @abstractmethod
#     def get_action_space(self):
#         pass

#     @abstractmethod
#     def get_observation_space(self):
#         pass

#     def reset(self):
#         game_state = self.bundle.reset(dic=self.reset_dic, turn=self.reset_turn)
#         dic = OrderedDict({})
#         for v in game_state.filter(mode="array").values():
#             dic.update(v)
#         return dic

#     def step(self, action):
#         user_action = action.get("user_action", None)
#         assistant_action = action.get("assistant_action", None)
#         obs, rewards, flag = self.bundle.step(
#             user_action=user_action, assistant_action=assistant_action
#         )
#         dic = OrderedDict({})
#         for v in obs.filter(mode="array").values():
#             dic.update(v)

#         return (
#             dic,
#             float(sum(rewards.values())),
#             flag,
#             {"CoopIHC Bundle {}".format(str(self.bundle))},
#         )

#     def convert(self, object):
#         if isinstance(object, StateElement):
#             object = object.spaces
#         return self._convertor.convert(object)


# class TrainGym(gym.Env, Train):
#     """Generic Wrapper to make bundles compatibles with gym.Env

#     This is a generic Wrapper to make bundles compatibles with gym.Env. Read more on the Train class.

#     :param bundle: bundle to wrap
#     :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
#     :param train_user: whether to train the user, defaults to True
#     :type train_user: bool, optional
#     :param train_assistant: whether to train the assistant, defaults to True
#     :type train_assistant: bool, optional
#     :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:

#     ..code-block:: python

#         filterdict = OrderedDict(
#         {
#             "user_state": OrderedDict({"goal": 0}),
#             "task_state": OrderedDict({"x": 0}),
#         }
#     )

#     :type observation_dict: collections.OrderedDict, optional
#     :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
#     :type reset_dic: dict, optional
#     :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
#     :type reset_turn: int, optional
#     """

#     def __init__(
#         self,
#         bundle,
#         *args,
#         train_user=False,
#         train_assistant=False,
#         observation_dict=None,
#         reset_dic={},
#         reset_turn=0,
#         **kwargs
#     ):

#         # Call Gym Env's constructor
#         # gym.Env.__init__()

#         self._convertor = GymConvertor()

#         # Call Train's constructor
#         Train.__init__(
#             self,
#             bundle,
#             *args,
#             train_user=train_user,
#             train_assistant=train_assistant,
#             API="gym",
#             observation_dict=observation_dict,
#             reset_dic=reset_dic,
#             reset_turn=reset_turn,
#             **kwargs,
#         )

#         # Below: Using ordereddict here is forced due to Open AI gym's behavior: when initializing the Dict space, it tries to order the dict by keys, which may change the order of the dict entries. This is actually useless since Python 3.7 because dicts are ordered by default.

#     def get_action_space(self):
#         # ----- Init Action space -------
#         action_dict = OrderedDict({})
#         if self.train_user:
#             action_dict.update({"user_action": self.convert(self.bundle.user.action)})
#         if self.train_assistant:
#             action_dict.update(
#                 {"assistant_action": self.convert(self.bundle.assistant.action)}
#             )
#         return gym.spaces.Dict(action_dict)

#     def get_observation_space(self):
#         # ------- Init Observation space
#         observation_dict = OrderedDict({})
#         observation_dict.update(
#             {
#                 "turn_index": gym.spaces.Discrete(4),
#                 "round_index": gym.spaces.Discrete(
#                     1000
#                 ),  # Maximum of N=1000 rounds. Should not affect performance and be big enough for most cases.
#             }
#         )
#         observation_dict.update(
#             {key: self.convert(value) for key, value in self.bundle.task.state.items()}
#         )
#         if self.bundle.user.state:
#             observation_dict.update(
#                 {
#                     key: self.convert(value)
#                     for key, value in self.bundle.user.state.items()
#                 }
#             )
#         if self.bundle.assistant.state:
#             observation_dict.update(
#                 {
#                     key: self.convert(value)
#                     for key, value in self.bundle.assistant.state.items()
#                 }
#             )

#         observation_dict.update({"user_action": self.convert(self.bundle.user.action)})
#         observation_dict.update(
#             {"assistant_action": self.convert(self.bundle.assistant.action)}
#         )
#         return gym.spaces.Dict(observation_dict)

#     def reset(self):
#         return Train.reset(self)

#     def step(self, action):
#         return Train.step(self, action)

#     def render(self, mode):
#         """See Bundle and gym API

#         :meta public:
#         """
#         self.bundle.render(mode)

#     def close(self):
#         """See Bundle and gym API

#         :meta public:
#         """
#         self.bundle.close()


class RLConvertor:
    def __init__(self, interface="gym", **kwargs):
        self.interface = interface
        if self.interface != "gym":
            raise NotImplementedError


class GymConvertor(RLConvertor):
    def __init__(self, **kwargs):
        super().__init__(interface="gym", **kwargs)

    def convert_space(self, space):
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

    # def convert_value(self, value, space):


# class Train:
#     """Generic Wrapper to make bundles compatibles with gym.Env

#     This is a generic Wrapper to make bundles compatibles with gym.Env. It is mainly here to be subclassed by other wrappers.

#     Depending on the spaces you are using, you might need to provide a wrapper to accustom the fact that coopihc spaces can take any values whereas e.g. gym discrete spaces have to be unit-spaced values.

#     .. note::

#         Experimental: This class automatically build wrappers to account for the transformation between a bundle and an env, but we offer no guarantees that it will work in all cases. It might also likely be faster (computationnally) to hard code your own wrappers.


#     :param bundle: bundle to wrap
#     :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
#     :param train_user: whether to train the user, defaults to True
#     :type train_user: bool, optional
#     :param train_assistant: whether to train the assistant, defaults to True
#     :type train_assistant: bool, optional
#     :param API: API with which the bundle will be made compatible for, defaults to "gym-force". In gym force, a limited gym compatible environment is created, which casts everything to float32 and boxes.
#     :type API: str, optional
#     :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:
#     :type observation_dict: collections.OrderedDict, optional

#     ..code-block:: python

#         filterdict = OrderedDict(
#             {
#                 "user_state": OrderedDict({"goal": 0}),
#                 "task_state": OrderedDict({"x": 0}),
#             }
#         )

#     :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
#     :type reset_dic: dict, optional
#     :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
#     :type reset_turn: int, optional
#     """

#     def __init__(
#         self,
#         bundle,
#         *args,
#         train_user=True,
#         train_assistant=True,
#         API="gym-force",
#         observation_dict=None,
#         reset_dic={},
#         reset_turn=0,
#         **kwargs
#     ):

#         self.bundle = bundle
#         self.train_user = train_user
#         self.train_assistant = train_assistant
#         self.observation_dict = observation_dict
#         self.reset_dic = reset_dic
#         self.reset_turn = reset_turn
#         self._convertor = None

#         if API == "gym":
#             self._convertor = GymConvertor
#         elif API == "gym-force":
#             self._convertor = GymForceConvertor
#         else:
#             raise NotImplementedError

#         (
#             self.action_space,
#             self.action_wrappers,
#         ) = self._get_action_spaces_and_wrappers()

#         (
#             self.observation_space,
#             self.observation_wrappers,
#         ) = self._get_observation_spaces_and_wrappers()

#     def _get_observation_spaces_and_wrappers(self):
#         """_get_observation_spaces_and_wrappers

#         Get obs spaces and wrappers by querying an observation from the bundle and calling the convertor."""
#         obs = self.bundle.reset()
#         if self.observation_dict is None:
#             filter = obs
#         else:
#             filter = self.observation_dict
#         spaces = hard_flatten(obs.filter("spaces", filter))
#         return self.convertor.get_spaces_and_wrappers(spaces, "observation")[:2]

#     def _get_action_spaces_and_wrappers(self):
#         """_get_action_spaces_and_wrappers [summary]

#         Get action spaces and wrappers. Checks who should be trained, and calls the convertor

#         """
#         action_spaces = []
#         if self.train_user:
#             user_action_space = self.bundle.game_state["user_action"]["action"][
#                 "spaces"
#             ]
#             action_spaces.extend(user_action_space)
#         else:
#             user_action_space = None

#         if self.train_assistant:
#             assistant_action_space = self.bundle.game_state["assistant_action"][
#                 "action"
#             ]["spaces"]
#             action_spaces.extend(assistant_action_space)
#         else:
#             assistant_action_space = None

#         self.bundle_user_action_space = user_action_space
#         self.bundle_assistant_action_space = assistant_action_space

#         self.convertor = self._convertor(bundle_action_spaces=action_spaces)
#         return self.convertor.get_spaces_and_wrappers(action_spaces, "action")[
#             :2
#         ]  # no wrapper flags returned

#     def _convert_observation(self, observation):
#         """_convert_observation

#         Hard flattens the bundle observation and casts to int or array


#         """
#         if isinstance(self.observation_space, gym.spaces.Discrete):
#             return int(
#                 hard_flatten(observation.filter("values", self.observation_dict))[0]
#             )
#         else:
#             return numpy.array(
#                 hard_flatten(observation.filter("values", self.observation_dict))
#             )
