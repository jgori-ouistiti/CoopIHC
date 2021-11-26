from coopihc.agents import BaseAgent
from coopihc.space import StateElement
import copy


### Goal could be defined as a target state of the task, in a more general description.
class GoalDrivenDiscreteUser(BaseAgent):
    """A User that is driven by a Goal and uses Discrete actions. It expects to be used with a task that has a substate named targets. Its internal state includes a 'goal' substate, whose value is either one of the task's targets."""

    def finit(self):
        """Appends a Goal substate to the agent's internal state (with dummy values).

        :meta public:
        """
        target_space = self.bundle.task.state["targets"]["spaces"][0]
        self.state["goal"] = StateElement(values=None, spaces=copy.copy(target_space))

        return

    def render(self, *args, **kwargs):
        """Similar to BaseAgent's render, but prints the "Goal" state in addition.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axuser, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """

        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                self.ax = axuser
                self.ax.text(0, 0, "Goal: {}".format(self.state["goal"][0][0]))
                self.ax.set_xlim([-0.5, 0.5])
                self.ax.set_ylim([-0.5, 0.5])
                self.ax.axis("off")
                self.ax.set_title(type(self).__name__ + " Goal")
        if "text" in mode:
            print(type(self).__name__ + " Goal")
            print(self.state["Goal"][0][0])
