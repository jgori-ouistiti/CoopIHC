from coopihc.base.State import State
from coopihc.base.elements import cat_element
from coopihc.policy.BasePolicy import BasePolicy


# ============== General Policies ===============


class DualPolicy(BasePolicy):
    def __init__(
        self, primary_policy, dual_policy, primary_kwargs={}, dual_kwargs={}, **kwargs
    ):
        if type(primary_policy).__name__ == "type":
            self.primary_policy = primary_policy(**primary_kwargs)
        else:
            self.primary_policy = primary_policy

        if type(dual_policy).__name__ == "type":
            self.dual_policy = dual_policy(**dual_kwargs)
        else:
            self.dual_policy = dual_policy

        super().__init__()

        self._host = None
        self._action_state = None
        delattr(self, "action_state")
        delattr(self, "host")

        self._mode = "primary"

    @property
    def mode(self):
        return self._mode

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        self.primary_policy.host = value
        self.dual_policy.host = value

    @property
    def action_state(self):
        if self._mode == "primary":
            return self.primary_policy.action_state
        else:
            return self.dual_policy.action_state

    def _base_sample(self):
        action, reward = self.sample(observation=None)
        self.action = action
        return self.action, reward

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):
        if self.mode == "primary":
            return self.primary_policy.sample(
                agent_observation=agent_observation, agent_state=agent_state
            )
        else:
            return self.dual_policy.sample(
                agent_observation=agent_observation, agent_state=agent_state
            )
