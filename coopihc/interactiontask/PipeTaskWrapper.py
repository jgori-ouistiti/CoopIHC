from abc import ABC, abstractmethod
from coopihc.interactiontask.InteractionTask import InteractionTask


class PipeTaskWrapper(InteractionTask, ABC):
    def __init__(self, task, pipe):
        self.__dict__ = task.__dict__
        self.task = task
        self.pipe = pipe
        self.pipe.send({"type": "init", "parameters": self.parameters})
        is_done = False
        while True:
            self.pipe.poll(None)
            received_state = self.pipe.recv()
            # This assumes that the final message sent by the client is a task_state message. Below should be changed to remove that assumption (i.e. client can send whatever order)
            if received_state["type"] == "task_state":
                is_done = True
            self.update_state(received_state)
            if is_done:
                break

    def __getattr__(self, attr):
        if self.__dict__:
            return getattr(self.__dict__["task"], attr)
        else:
            # should never happen
            pass

    def __setattr__(self, name, value):
        if name == "__dict__" or name == "task":
            super().__setattr__(name, value)
            return
        if self.__dict__:
            setattr(self.__dict__["task"], name, value)

    def update_state(self, state):
        print("updating state")
        if state["type"] == "task_state":
            del state["type"]
            self.update_task_state(state)
        elif state["type"] == "user_state":
            del state["type"]
            self.update_user_state(state)

    @abstractmethod
    def update_task_state(self, state):
        pass

    @abstractmethod
    def update_user_state(self, state):
        pass

    def user_step(self, *args, **kwargs):
        super().user_step(*args, **kwargs)
        user_action_msg = {
            "type": "user_action",
            "value": self.bundle.game_state["user_action"]["action"].serialize(),
        }
        self.pipe.send(user_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic["state"]
        self.update_state(received_state)
        return self.state, received_dic["reward"], received_dic["is_done"], {}

    def assistant_step(self, *args, **kwargs):
        super().assistant_step(*args, **kwargs)
        assistant_action_msg = {
            "type": "assistant_action",
            "value": self.bundle.game_state["assistant_action"]["action"].serialize(),
        }
        self.pipe.send(assistant_action_msg)
        self.pipe.poll(None)
        received_dic = self.pipe.recv()
        received_state = received_dic["state"]
        self.update_state(received_state)
        return self.state, received_dic["reward"], received_dic["is_done"], {}

    def reset(self, dic=None):
        super().reset(dic=dic)
        reset_msg = {"type": "reset", "reset_dic": dic}
        self.pipe.send(reset_msg)
        self.pipe.poll(None)
        received_state = self.pipe.recv()
        self.update_state(received_state)
        self.bundle.reset(task=False)
        return self.state
