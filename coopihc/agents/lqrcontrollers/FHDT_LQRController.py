from coopihc.agents.lqrcontrollers.LQRController import LQRController
import scipy.linalg


# Finite Horizon Discrete Time Controller
# Outdated
class FHDT_LQRController(LQRController):
    """Finite Horizon Discrete Time LQR

    A Finite Horizon (i.e. planning for N steps) Discrete Time implementation of the LQR controller.

    :param N: Horizon (steps)
    :type N: int
    :param role: "user" or "assistant"
    :type role: string
    :param Q: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type Q: numpy.ndarray
    :param R: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type R: numpy.ndarray
    :param Gamma: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type Gamma: float
    """

    def __init__(self, N, role, Q, R, Gamma):
        self.N = N
        self.i = 0
        super().__init__(role, Q, R, gamma=Gamma)
        self.timespace = "discrete"

    # untested, old version below
    def reset(self):
        """reset"""
        self.i = 0

    # def reset(self, dic=None):
    #
    #     self.i = 0
    #     super().reset(dic)

    def finit(self):
        """finit

        Compute feedback gain from A, B, Q, R matrices.
        """
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
