from coopihc.observation.BaseObservationEngine import BaseObservationEngine


class WrapAsObservationEngine(BaseObservationEngine):
    """WrapAsObservationEngine

    Wrap a bundle as an Observation Engine

    :param obs_bundle: bundle that simulates an observation process
    :type obs_bundle: `Bundle :py:mod:<coopihc.bundle>`
    """

    def __init__(self, obs_bundle):

        self.bundle = obs_bundle
        self.bundle.reset()

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

    def observe(self, game_state):
        pass
        # Do something
        # return observation, rewards

    def __str__(self):
        return "{} <[ {} ]>".format(self.__class__.__name__, self.bundle.__str__())

    def __repr__(self):
        return self.__str__()
