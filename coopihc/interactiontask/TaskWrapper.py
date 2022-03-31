from coopihc.interactiontask.InteractionTask import InteractionTask


class TaskWrapper(InteractionTask):
    """TaskWrapper

    Unused ?

    """

    def __init__(self, task):
        self.task = task
        self.__dict__.update(task.__dict__)

    def on_user_action(self, *args, **kwargs):
        return self.task.on_user_action(*args, **kwargs)

    def on_assistant_action(self, *args, **kwargs):
        return self.task.on_assistant_action(*args, **kwargs)

    def reset(self, dic=None):
        return self.task.reset(dic=dic)

    def render(self, *args, **kwargs):
        return self.task.render(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.task.unwrapped
