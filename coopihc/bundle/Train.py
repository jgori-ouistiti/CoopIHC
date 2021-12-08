from coopihc.helpers import hard_flatten
import gym
import numpy


class Train(gym.Env):
    """Train

    Use this class to wrap a Bundle up, so that it is compatible with the gym API and can be trained with off-the-shelf RL algorithms.

    .. warning::

        outdated


    :param bundle: bundle to wrap
    :type bundle: :py:class:`Bundle<coopihc.bundle.Bundle.Bundle>`
    """

    def __init__(self, bundle, *args, **kwargs):

        self.bundle = bundle
        self.action_space = gym.spaces.Tuple(bundle.action_space)

        obs = bundle.reset()

        self.observation_mode = kwargs.get("observation_mode")
        self.observation_dict = kwargs.get("observation_dict")

        if self.observation_mode is None:
            self.observation_space = obs.filter("spaces", obs)
        elif self.observation_mode == "tuple":
            self.observation_space = gym.spaces.Tuple(
                hard_flatten(obs.filter("spaces", self.observation_dict))
            )
        elif self.observation_mode == "multidiscrete":
            self.observation_space = gym.spaces.MultiDiscrete(
                [i.n for i in hard_flatten(obs.filter("spaces", self.observation_dict))]
            )
        elif self.observation_mode == "dict":
            self.observation_space = obs.filter("spaces", self.observation_dict)
        else:
            raise NotImplementedError

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
