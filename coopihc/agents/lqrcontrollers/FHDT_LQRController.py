from .LQRController import LQRController
import scipy.linalg


# Finite Horizon Discrete Time Controller
# Outdated
class FHDT_LQRController(LQRController):
    def __init__(self, N, role, Q, R, Gamma):
        self.N = N
        self.i = 0
        super().__init__(role, Q, R, gamma=Gamma)
        self.timespace = "discrete"

    def reset(self, dic=None):
        self.i = 0
        super().reset(dic)

    def finit(self):
        self.K = []
        task = self.bundle.task
        A, B = task.A, task.B
        # Compute P(k) matrix for k in (N:-1:1)
        self.P = [self.Q]
        for k in range(self.N - 1, 0, -1):
            Pcurrent = self.P[0]
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            Pnext = (
                self.Q
                + A.T @ Pcurrent @ A
                - A.T @ Pcurrent @ B @ invPart @ B.T @ Pcurrent @ A
            )
            self.P.insert(0, Pnext)

        # Compute Kalman Gain
        for Pcurrent in self.P:
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            K = -invPart @ B.T @ Pcurrent @ A
            self.K.append(K)
