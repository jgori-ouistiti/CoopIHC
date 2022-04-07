import numpy
from coopihc.policy.BasePolicy import BasePolicy
import copy


class ExamplePolicy(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.


    """

    def __init____init__(self, *args, action_state=None, **kwargs):
        super().__init__(*args, action_state=None, **kwargs)

    @BasePolicy.default_value
    def sample(self, agent_observation=None, agent_state=None):
        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.base.StateElement.StateElement>`, float)
        """

        if (
            agent_observation["task_state"]["x"]
            < agent_observation["user_state"]["goal"]
        ):
            _action_value = 1
        elif (
            agent_observation["task_state"]["x"]
            > agent_observation["user_state"]["goal"]
        ):
            _action_value = -1
        else:
            _action_value = 0

        return _action_value, 0


class PseudoRandomPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, agent_observation=None, agent_state=None):
        if agent_observation is None:
            agent_observation = self.observation

        x = agent_observation.task_state.x

        _action_value = (
            8 + self.state.p0 * x + self.state.p1 * x * x + self.state.p2 * x * x * x
        ) % 10

        return _action_value, 0


class CoordinatedPolicy(BasePolicy):
    @property
    def simulation_bundle(self):
        return self.host.simulation_bundle

    def sample(self, agent_observation=None, agent_state=None):
        if agent_observation is None:
            agent_observation = self.observation

        reset_dic = {"task_state": agent_observation.task_state}

        self.simulation_bundle.reset(dic=reset_dic)
        self.simulation_bundle.step(turn=2)

        _action_value = self.simulation_bundle.user.action

        return _action_value, 0


class CoordinatedPolicyWithParams(CoordinatedPolicy):
    def sample(self, agent_observation=None, agent_state=None):
        if agent_observation is None:
            agent_observation = self.observation

        reset_dic = {
            "task_state": copy.deepcopy(agent_observation.task_state),
            "user_state": {
                "p0": copy.deepcopy(agent_observation.assistant_state.user_p0)
            },
        }

        self.simulation_bundle.reset(dic=reset_dic)
        self.simulation_bundle.step(turn=2)

        _action_value = copy.copy(self.simulation_bundle.user.action)

        return _action_value, 0
