from .BaseInferenceEngine import BaseInferenceEngine
from .GoalInferenceWithUserPolicyGiven import GoalInferenceWithUserPolicyGiven
from .LinearGaussianContinuous import LinearGaussianContinuous
from .ContinuousKalmanUpdate import ContinuousKalmanUpdate


# ================ Examples ============


class ExampleInferenceEngine(BaseInferenceEngine):
    def infer(self):
        return self.state, 0
