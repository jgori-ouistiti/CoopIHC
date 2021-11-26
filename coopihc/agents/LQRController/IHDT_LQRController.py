from coopihc.agents import LQRController
import scipy.linalg


# Infinite Horizon Discrete Time Controller
# Uses Discrete Algebraic Ricatti Equation to get P


class IHDT_LQRController(LQRController):
    def __init__(self, role, Q, R, Gamma):
        super().__init__(role, Q, R, gamma=Gamma)
        self.timespace = "discrete"

    def finit(self):
        task = self.bundle.task
        A, B = task.A, task.B
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        invPart = scipy.linalg.inv((self.R + B.T @ P @ B))
        K = invPart @ B.T @ P @ A
        self.policy.set_feedback_gain(K)
