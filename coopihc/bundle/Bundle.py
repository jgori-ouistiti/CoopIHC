from coopihc.bundle._Bundle import _Bundle
from coopihc.agents.BaseAgent import BaseAgent


class Bundle(_Bundle):
    """Bundle

    Modifies the interface of the _Bundle class.

    A bundle combines a task with a user and an assistant. The bundle creates the ``game_state`` by combining the task, user and assistant states with the turn index and both agent's actions.

    The bundle takes care of all the messaging between classes, making sure the gamestate and all individual states are synchronized at all times.

    The bundle implements a forced reset mechanism, where each state of the bundle can be forced to a particular state via a dictionnary mechanism (see :py:func:reset)

    The bundle also takes care of rendering each of the three component in a single place.


    :param task: A task that inherits from ``InteractionTask``, defaults to None
    :type task: :py:class:`coopihc.interactiontask.InteractionTask.InteractionTask`, optional
    :param user: a user which inherits from ``BaseAgent``, defaults to None
    :type user: :py:class:`coopihc.agents.BaseAgent.BaseAgent`, optional
    :param assistant: an assistant which inherits from ``BaseAgent``, defaults to None
    :type assistant: :py:class:`coopihc.agents.BaseAgent.BaseAgent`, optional
    """

    def __init__(self, *args, task=None, user=None, assistant=None, **kwargs):
        if task is None:
            task_bit = "0"
            raise NotImplementedError
        else:
            task_bit = "1"
        if user is None:
            user = BaseAgent("user")
            user_bit = "0"
        else:
            user_bit = "1"
        if assistant is None:
            assistant = BaseAgent("assistant")
            assistant_bit = "0"
        else:
            assistant_bit = "1"

        self.bundle_bits = task_bit + user_bit + assistant_bit

        if user_bit + assistant_bit == "00":
            name = "no-user--no-assistant"
        elif user_bit + assistant_bit == "01":
            name = "no-user"
        elif user_bit + assistant_bit == "10":
            name = "no-assistant"
        else:
            name = "full"

        super().__init__(task, user, assistant, *args, name=name, **kwargs)
