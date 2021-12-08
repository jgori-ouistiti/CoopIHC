from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ExampleInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Example class

    """

    def infer(self):
        """infer

        Do nothing. Same behavior as parent ``BaseInferenceEngine``

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        return self.state, 0
