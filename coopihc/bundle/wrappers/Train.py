from coopihc.helpers import hard_flatten

from coopihc.space.Space import Space
from coopihc.space.utils import NotASpaceError

import gym
import numpy


class Train(gym.Env):
    """Train

    Use this class to wrap a Bundle, making it compatible with the gym API. From there, it can be trained with off-the-shelf RL algorithms.


    :param bundle: bundle to wrap
    :type bundle: :py:class:`Bundle<coopihc.bundle.Bundle.Bundle>`
    """

    def __init__(self, bundle, *args, train_user=True, train_assistant=True, **kwargs):

        self.bundle = bundle
        self.train_user = train_user
        self.train_assistant = train_assistant
        self.user_action_space, self.assistant_action_space = self._get_action_spaces()
        #  = self._convert_action_space(
        #     action_spaces
        # )
        # self.action_space = gym.spaces.Tuple(bundle.action_space)

        # obs = bundle.reset()

        # self.observation_mode = kwargs.get("observation_mode")
        # self.observation_dict = kwargs.get("observation_dict")

        # if self.observation_mode is None:
        #     self.observation_space = obs.filter("spaces", obs)
        # elif self.observation_mode == "tuple":
        #     self.observation_space = gym.spaces.Tuple(
        #         hard_flatten(obs.filter("spaces", self.observation_dict))
        #     )
        # elif self.observation_mode == "multidiscrete":
        #     self.observation_space = gym.spaces.MultiDiscrete(
        #         [i.n for i in hard_flatten(obs.filter("spaces", self.observation_dict))]
        #     )
        # elif self.observation_mode == "dict":
        #     self.observation_space = obs.filter("spaces", self.observation_dict)
        # else:
        #     raise NotImplementedError

    def _get_action_spaces(self):

        self.bundle.reset()
        if self.train_user:
            user_action_space = self.bundle.game_state["user_action"]["action"][
                "spaces"
            ]
        else:
            user_action_space = None

        if self.train_assistant:
            assistant_action_space = self.bundle.game_state["assistant_action"][
                "action"
            ]["spaces"]
        else:
            assistant_action_space = None

        return self._convert_action_space(
            user_action_space
        ), self._convert_action_space(assistant_action_space)

    def _convert_action_space(self, action_space):
        # spaces = []
        discrete = False
        continuous = False
        for asp in action_space:
            if asp.continuous:
                continuous = True
            else:
                discrete = True
            if not isinstance(asp, Space):
                raise NotASpaceError
            # else:
            #     spaces.append(asp.convert_to_gym())

        # continuous and discrete are flags that indicate if at least one C or D space is to be converted.
        if continuous and discrete:
            raise NotImplementedError
        if discrete:
            for sp in action_space:
                if not (
                    numpy.mean(numpy.diff(sp.range)) == 1.0
                    and numpy.std(numpy.diff(sp.range)) == 0.0
                ):
                    raise NotImplementedError(
                        "Only works currently for action_spaces which increment by 1, but you have {} (mu = {}, std = {})".format(
                            sp.range,
                            numpy.mean(numpy.diff(sp.range)),
                            numpy.std(numpy.diff(sp.range)),
                        )
                    )

            if len(action_space) == 1:
                if gym.__version__ < "0.21":
                    if action_space[0].low[0] != 0:
                        raise NotImplementedError
                    else:
                        return gym.spaces.Discrete(action_space[0].N)
                else:
                    return gym.spaces.Discrete(
                        action_space[0].N, start=action_space[0].low[0]
                    )
            else:
                return gym.spaces.MultiDiscrete([s.N for s in action_space])
        else:
            if len(action_space) != 1:
                raise NotImplementedError
            else:
                return gym.spaces.Box(
                    action_space[0].low,
                    action_space[0].high,
                    dtype=action_space[0].dtype,
                    seed=action_space[0].seed,
                )

    def convert_observation(self, observation):
        if self.observation_mode is None:
            return observation
        elif self.observation_mode == "tuple":
            return self.convert_observation_tuple(observation)
        elif self.observation_mode == "multidiscrete":
            return self.convert_observation_multidiscrete(observation)
        elif self.observation_mode == "dict":
            return self.convert_observation_dict(observation)
        else:
            raise NotImplementedError

    def convert_observation_tuple(self, observation):
        return tuple(hard_flatten(observation.filter("values", self.observation_dict)))

    def convert_observation_multidiscrete(self, observation):
        return numpy.array(
            hard_flatten(observation.filter("values", self.observation_dict))
        )

    def convert_observation_dict(self, observation):
        return observation.filter("values", self.observation_dict)

    def reset(self, dic={}, **kwargs):
        """Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(dic=dic, **kwargs)
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
