from coopihc.bundle._Bundle import _Bundle


class BundleWrapper(_Bundle):
    """BundleWrapper [summary]

    .. warning::

        Outdated/unused

    :param bundle: bundle to wrap
    :type bundle: :py:mod:`Bundle<coopihc.bundle>`
    """

    def __init__(self, bundle):

        self.__class__ = type(
            bundle.__class__.__name__, (self.__class__, bundle.__class__), {}
        )
        self.__dict__ = bundle.__dict__
