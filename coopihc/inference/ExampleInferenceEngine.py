from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine

# [start-infeng-subclass]
class ExampleInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Example class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, user_state=None):
        """infer

        Do nothing. Same behavior as parent ``BaseInferenceEngine``

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        if user_state is None:
            user_state = self.state

        reward = 0
        # Do something
        # user_state = ..
        # reward = ...

        return user_state, reward


ExampleInferenceEngine(buffer_depth=5)
# [end-infeng-subclass]
