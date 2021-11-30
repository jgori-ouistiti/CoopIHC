import numpy
from coopihc.policy.BasePolicy import BasePolicy


class ExamplePolicy(BasePolicy):
    """A simple policy which assumes that the agent using it has a goal state nd that the task has an 'x' state. x is compared to the goal and appropriate ction is taken to make sure x reaches the goal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        if (
            self.observation["task_state"]["x"]
            < self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = 1
        elif (
            self.observation["task_state"]["x"]
            > self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = -1
        else:
            _action_value = 0

        new_action = self.new_action["values"] = numpy.array(_action_value)
        return new_action, 0
