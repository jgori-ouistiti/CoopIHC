from core.bundle import _Bundle
from core.agents import BaseAgent


class Bundle(_Bundle):
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
