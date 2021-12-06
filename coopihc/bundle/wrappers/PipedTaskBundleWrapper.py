from coopihc.bundle._Bundle import _Bundle
import time


class PipedTaskBundleWrapper(_Bundle):
    """PipedTaskBundleWrapper

    Wrap a Bundle so that its task gets replaced by a version of the task which emits messages via a pipe.

    :param bundle: bundle to wrap
    :type bundle: :py:mod:`Bundle<coopihc.bundle>`
    :param taskwrapper: Task wrapper
    :type taskwrapper: :py:class:`PipeTaskWrapper<coopihc.interactiontask.PipeTaskWrapper.PipeTaskWrapper>`
    :param pipe: Pipe through which messages are sent
    :type pipe: :py:class:`Pipe <subprocess.Pipe>`
    """

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
        """run the bundle

        Play the game.

        :param reset_dic: reset dictionnary, see :py:mod:`Bundle API<coopihc.bundle>`, defaults to {}
        :type reset_dic: dict, optional
        """
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
        """end

        Send 'done' over the pipe to signal that the game is over
        """
        self.pipe.send("done")
