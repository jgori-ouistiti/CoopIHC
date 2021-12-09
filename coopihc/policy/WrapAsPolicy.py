from coopihc.policy.BasePolicy import BasePolicy

# ============== General Policies ===============


class WrapAsPolicy(BasePolicy):
    """WrapAsPolicy

    Wrap a bundle as a policy.

    :param action_bundle: bundle to wrap
    :type action_bundle: `Bundle<coopihc.bundle>`
    :param action_state: which bundle game state should serve as action state
    :type action_state: `State<coopihc.space.State.State>`
    """

    def __init__(self, action_bundle, action_state, *args, **kwargs):
        super().__init__(*args, action_state=action_state, **kwargs)
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

    def sample(self, observation=None):
        pass
        # Do something
        # return action, rewards

    def __str__(self):
        return "{} <[ {} ]>".format(self.__class__.__name__, self.bundle.__str__())

    def __repr__(self):
        return self.__str__()
