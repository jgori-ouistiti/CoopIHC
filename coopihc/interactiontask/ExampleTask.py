import numpy
from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.interactiontask.InteractionTask import InteractionTask


class ExampleTask(InteractionTask):
    """ExampleTask

    An example algebraic task which a single task state 'x', which finishes when x = 4.

    """

    def __init__(self, *args, scale=1, **kwargs):
        # Call super().__init__() beofre anything else, which initializes ome useful attributes, including a State (self.state) for the task
        super().__init__(*args, **kwargs)

        self.state["x"] = StateElement(
            values=0,
            spaces=Space(
                [numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)]
            ),
            clipping_mode="clip",
        )

    def reset(self, dic=None):
        """reset

        .. warning ::

            Verify signature, dic mechanism normally taken care of.

        :param dic: [description], defaults to None
        :type dic: [type], optional
        """
        self.state["x"]["values"] = numpy.array([0])
        return

    def user_step(self, *args, **kwargs):
        """user_step

        Add the user action to 'x'

        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float, boolean, dictionnary)
        """
        is_done = False
        self.state["x"] += self.user_action

        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def assistant_step(self, *args, **kwargs):
        """assistant_step

        Add the assistant action to 'x'

        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float, boolean, dictionnary)
        """
        is_done = False

        self.state["x"] += self.assistant_action
        if int(self.state["x"]["values"][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def render(self, *args, mode="text"):
        """render

        text mode: prints turn number and state.

        :param mode: [description], defaults to "text"
        :type mode: str, optional
        """
        if "text" in mode:
            print("\n")
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError("Only 'text' mode implemented for this task")
