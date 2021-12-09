from coopihc.interactiontask.InteractionTask import InteractionTask


class TaskWrapper(InteractionTask):
    """TaskWrapper

    Unused ?

    """

    def __init__(self, task):
        self.task = task
        self.__dict__.update(task.__dict__)

    def user_step(self, *args, **kwargs):
        return self.task.user_step(*args, **kwargs)

    def assistant_step(self, *args, **kwargs):
        return self.task.assistant_step(*args, **kwargs)

    def reset(self, dic=None):
        return self.task.reset(dic=dic)

    def render(self, *args, **kwargs):
        return self.task.render(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.task.unwrapped
