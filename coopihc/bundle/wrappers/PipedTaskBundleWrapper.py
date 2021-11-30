from coopihc.bundle._Bundle import _Bundle
import time


class PipedTaskBundleWrapper(_Bundle):
    # Wrap it by taking over bundles attribute via the instance __dict__. Methods can not be taken like that since they belong to the class __dict__ and have to be called via self.bundle.method()
    def __init__(self, bundle, taskwrapper, pipe):
        self.__dict__ = bundle.__dict__  # take over bundles attributes
        self.bundle = bundle
        self.pipe = pipe
        pipedtask = taskwrapper(bundle.task, pipe)
        self.bundle.task = pipedtask  # replace the task with the piped task
        bundle_kwargs = bundle.kwargs
        bundle_class = self.bundle.__class__
        self.bundle = bundle_class(
            pipedtask, bundle.user, bundle.assistant, **bundle_kwargs
        )

        self.framerate = 1000
        self.iter = 0

        self.run()

    def run(self, reset_dic={}, **kwargs):
        reset_kwargs = kwargs.get("reset_kwargs")
        if reset_kwargs is None:
            reset_kwargs = {}
        self.bundle.reset(dic=reset_dic, **reset_kwargs)
        time.sleep(1 / self.framerate)
        while True:
            obs, sum_reward, is_done, rewards = self.bundle.step()
            time.sleep(1 / self.framerate)
            if is_done:
                break
        self.end()

    def end(self):
        self.pipe.send("done")
