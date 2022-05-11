from coopihc.bundle.BaseBundle import BaseBundle
import copy


class Watcher:
    def __init__(self, bundle, *args, **kwargs):
        self.watching = bundle
        self.container = []

    def __getattr__(self, key):
        return getattr(self.watching, key)

    def reset(self, *args, **kwargs):
        ret = self.watching.reset(*args, **kwargs)
        self.container.append(copy.deepcopy(ret))
        return ret

    def step(self, *args, **kwargs):
        ret = self.watching.step(*args, **kwargs)
        self.container.append(copy.deepcopy(ret[0]))
        return ret
