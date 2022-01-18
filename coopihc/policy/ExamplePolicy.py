import numpy
from coopihc.policy.BasePolicy import BasePolicy


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
