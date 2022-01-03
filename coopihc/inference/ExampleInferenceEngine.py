from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ExampleInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Example class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, *args, **kwargs):
        """infer

        Do nothing. Same behavior as parent ``BaseInferenceEngine``

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        return self.state, 0
