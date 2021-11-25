from coopihc.bundle import _Bundle


class BundleWrapper(_Bundle):
    def __init__(self, bundle):
        self.__class__ = type(
            bundle.__class__.__name__, (self.__class__, bundle.__class__), {}
        )
        self.__dict__ = bundle.__dict__
