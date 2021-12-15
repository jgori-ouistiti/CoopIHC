from coopihc.helpers import hard_flatten

from coopihc.space.Space import Space
from coopihc.space.utils import NotASpaceError, GymConvertor

import gym
import numpy


class Train:
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a generic Wrapper to make bundles compatibles with gym.Env. It is mainly here to be subclassed by other wrappers

    :param bundle: bundle to wrap
    :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
    :param train_user: whether to train the user, defaults to True
    :type train_user: bool, optional
    :param train_assistant: whether to train the assistant, defaults to True
    :type train_assistant: bool, optional
    :param convertor: lib to which the bundle will be made compatible for, defaults to "gym"
    :type convertor: str, optional
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
        train_user=True,
        train_assistant=True,
        convertor="gym",
        observation_dict=None,
        reset_dic={},
        reset_turn=0,
        **kwargs
    ):

        self.bundle = bundle
        self.train_user = train_user
        self.train_assistant = train_assistant
        self.observation_dict = observation_dict
        self.reset_dic = reset_dic
        self.reset_turn = reset_turn

        if convertor == "gym":
            self.convertor = GymConvertor()
        else:
            raise NotImplementedError

        (
            self.action_space,
            self.action_wrappers,
        ) = self._get_action_spaces_and_wrappers()

        (
            self.observation_space,
            self.observation_wrappers,
        ) = self._get_observation_spaces_and_wrappers()

    def _get_observation_spaces_and_wrappers(self):
        obs = self.bundle.reset()
        if self.observation_dict is None:
            filter = obs
        else:
            filter = self.observation_dict
        spaces = hard_flatten(obs.filter("spaces", filter))
        return self.convertor.get_spaces_and_wrappers(spaces, "observation")[:2]

    def _get_action_spaces_and_wrappers(self):
        gc = self.convertor
        action_spaces = []
        if self.train_user:
            user_action_space = self.bundle.game_state["user_action"]["action"][
                "spaces"
            ]
            action_spaces.extend(user_action_space)
        else:
            user_action_space = None

        if self.train_assistant:
            assistant_action_space = self.bundle.game_state["assistant_action"][
                "action"
            ]["spaces"]
            action_spaces.extend(assistant_action_space)
        else:
            assistant_action_space = None

        self.bundle_user_action_space = user_action_space
        self.bundle_assistant_action_space = assistant_action_space

        return gc.get_spaces_and_wrappers(action_spaces, "action")[
            :2
        ]  # no wrapper flags returned

    def _convert_observation(self, observation):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return int(
                hard_flatten(observation.filter("values", self.observation_dict))[0]
            )
        else:
            return numpy.array(
                hard_flatten(observation.filter("values", self.observation_dict))
            )


class TrainGym(Train, gym.Env):
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a generic Wrapper to make bundles compatibles with gym.Env. It is mainly here to be subclassed by other wrappers

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
        train_user=True,
        train_assistant=True,
        observation_dict=None,
        reset_dic={},
        reset_turn=0,
        **kwargs
    ):
        super().__init__(
            bundle,
            *args,
            train_user=train_user,
            train_assistant=train_assistant,
            convertor="gym",
            observation_dict=observation_dict,
            reset_dic=reset_dic,
            reset_turn=reset_turn,
            **kwargs
        )

    def reset(self):
        """Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        """
        obs = self.bundle.reset(turn=self.reset_turn, dic=self.reset_dic)
        return self._convert_observation(obs)

    def step(self, action):
        """Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """

        user_action = action[: len(self.bundle_user_action_space)]
        assistant_action = action[len(self.bundle_user_action_space) :]
        obs, rewards, is_done = self.bundle.step(user_action, assistant_action)

        return (
            self._convert_observation(obs),
            float(sum(rewards.values())),
            is_done,
            rewards,
        )

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
