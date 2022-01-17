from coopihc.agents.lqrcontrollers.LQRController import LQRController
import scipy.linalg


class IHDT_LQRController(LQRController):
    """Infinite Horizon Discrete Time LQR

    An Infinite Horizon (i.e. planning for unicode:: U+221E .. steps) Discrete Time implementation of the LQR controller.


    :param role: "user" or "assistant"
    :type role: string
    :param Q: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type Q: numpy.ndarray
    :param R: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type R: numpy.ndarray
    :param Gamma: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type Gamma: float
    """

    def __init__(self, role, Q, R, Gamma):

        super().__init__(role, Q, R, gamma=Gamma)

    def finit(self):
        """finit

        Uses Discrete Algebraic Ricatti Equation to get P

        :meta public:
        """
        task = self.bundle.task
        A, B = task.A, task.B
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        invPart = scipy.linalg.inv((self.R + B.T @ P @ B))
        K = invPart @ B.T @ P @ A
        self.policy.set_feedback_gain(K)
