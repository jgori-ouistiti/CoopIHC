import numpy
from coopihc.policy.BasePolicy import BasePolicy
import copy


class ExamplePolicy(BasePolicy):
    """ExamplePolicy

    A simple policy which assumes that the agent using it has a 'goal' state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal.


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, observation=None):
        """sample

        Compares 'x' to goal and issues +-1 accordingly.

        :return: action, reward
        :rtype: tuple(`StateElement<coopihc.space.StateElement.StateElement>`, float)
        """

        if observation is None:
            observation = self.observation

        if (
            observation["task_state"]["x"]
            < observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = 1
        elif (
            observation["task_state"]["x"]
            > observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = -1
        else:
            _action_value = 0

        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, 0


class PseudoRandomPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, observation=None):
        if observation is None:
            observation = self.observation

        x = observation.task_state.x

        _action_value = (
            8 + self.state.p0 * x + self.state.p1 * x * x + self.state.p2 * x * x * x
        ) % 10

        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, 0


class CoordinatedPolicy(BasePolicy):
    @property
    def simulation_bundle(self):
        return self.host.simulation_bundle

    def sample(self, observation=None):
        if observation is None:
            observation = self.observation

        reset_dic = {"task_state": observation.task_state}

        self.simulation_bundle.reset(dic=reset_dic)
        self.simulation_bundle.step(turn=2)

        _action_value = self.simulation_bundle.game_state.user_action.action.view(
            numpy.ndarray
        )
        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, 0


class CoordinatedPolicyWithParams(CoordinatedPolicy):
    def sample(self, observation=None):
        if observation is None:
            observation = self.observation

        reset_dic = {
            "task_state": copy.deepcopy(observation.task_state),
            "user_state": {"p0": copy.deepcopy(observation.assistant_state.user_p0)},
        }

        self.simulation_bundle.reset(dic=reset_dic)
        self.simulation_bundle.step(turn=2)

        _action_value = copy.copy(
            self.simulation_bundle.game_state.user_action.action[:]
        )

        new_action = self.new_action
        new_action[:] = _action_value

        return new_action, 0
