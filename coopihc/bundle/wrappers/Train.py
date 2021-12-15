from coopihc.helpers import hard_flatten

from coopihc.space.Space import Space
from coopihc.space.utils import NotASpaceError, GymConvertor

import gym
import numpy


class Train:
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

        return gc.get_spaces_and_wrappers(action_spaces, "action")[
            :2
        ]  # no wrapper flags returned

    # def convert_observation(self, observation):
    #     if self.observation_mode is None:
    #         return observation
    #     elif self.observation_mode == "tuple":
    #         return self.convert_observation_tuple(observation)
    #     elif self.observation_mode == "multidiscrete":
    #         return self.convert_observation_multidiscrete(observation)
    #     elif self.observation_mode == "dict":
    #         return self.convert_observation_dict(observation)
    #     else:
    #         raise NotImplementedError

    # def convert_observation_tuple(self, observation):
    #     return tuple(hard_flatten(observation.filter("values", self.observation_dict)))

    # def convert_observation_multidiscrete(self, observation):
    #     return numpy.array(
    #         hard_flatten(observation.filter("values", self.observation_dict))
    #     )

    # def convert_observation_dict(self, observation):
    #     return observation.filter("values", self.observation_dict)

    def convert_observation(self, observation):
        return hard_flatten(observation.filter("values", self.observation_dict))

    def reset(self):
        """Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(turn=self.reset_turn, dic=self.reset_dic)
        return self.convert_observation(obs)

    def step(self, action):
        """Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """

        obs, sum_reward, is_done, rewards = self.bundle.step(action)

        return (
            self.convert_observation(obs),
            sum_reward,
            is_done,
            {"rewards": rewards},
        )

    def render(self, mode):
        """See Bundle

        :meta public:
        """
        self.bundle.render(mode)

    def close(self):
        """See Bundle

        :meta public:
        """
        self.bundle.close()


class TrainGym(Train, gym.Env):
    """Train

    Use this class to wrap a Bundle, making it compatible with the gym API. From there, it can be trained with off-the-shelf RL algorithms.


    :param bundle: bundle to wrap
    :type bundle: :py:class:`Bundle<coopihc.bundle.Bundle.Bundle>`
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
