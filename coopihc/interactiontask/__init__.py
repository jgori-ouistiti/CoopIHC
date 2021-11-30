import numpy
from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.interactiontask.InteractionTask import InteractionTask


class ExampleTask(InteractionTask):
    """ExampleTask with two agents. The task is finished when 'x' reaches . The user and assistants can both do -1, +0, +1 to 'x'."""

    def __init__(self, *args, **kwargs):
        # Call super().__init__() beofre anything else, which initializes ome useful attributes, including a State (self.state) for the task
        super().__init__(*args, **kwargs)

        self.state["x"] = StateElement(
            values=0,
            spaces=Space(
                [numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.nt16)]
            ),
            clipping_mode="clip",
        )

    def reset(self, dic=None):
        self.state["x"]["values"] = numpy.array([0])
        return

    def user_step(self, *args, **kwargs):
        is_done = False
        self.state["x"] += self.user_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def assistant_step(self, *args, **kwargs):
        is_done = False
        self.state["x"] += self.assistant_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def render(self, *args, mode="text"):
        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number.squeeze().olist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError


if __name__ == "__main__":
    et = ExampleTask()
