from coopihc.agents.lqrcontrollers.LQRController import LQRController
import scipy.linalg


class IHDT_LQRController(LQRController):
    """Infinite Horizon Discrete Time LQR

    An Infinite Horizon (i.e. planning for unicode:: U+221E .. steps) Discrete Time implementation of the LQR controller. The controller is computed to minimize costs :math: `X^tQX + u^t R u`, where X is the state of the system and u is the linear feedback command :math:`u = -K X`, where the feedback gain :math:`K` is given by solving the discrete ARE

    .. math::

        \\begin{align}
            K = (R + B^tPB)^{-1}B^TPA \\text{ (gain)}\\
            P = Q + A^tPA - A^tPB(R + B^tPB)^{-1}B^TPA \\text{Discrete ARE}
        \\end{align}


    :param role: "user" or "assistant"
    :type role: string
    :param Q: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type Q: numpy.ndarray
    :param R: see :py:class:`LQRController <coopihc.agents.lqrcontrollers.LQRController.LQRController>`
    :type R: numpy.ndarray
    :param Acontroller: Model of A used by the controller to compute K
    :type Acontroller: numpy.ndarray
    :param Bcontroller: Model of B used by the controller to compute K
    :type Bcontroller: numpy.ndarray



    """

    def __init__(self, role, Q, R, Acontroller=None, Bcontroller=None):
        self.Acontroller = Acontroller
        self.Bcontroller = Bcontroller
        super().__init__(role, Q, R)

    def finit(self):
        """finit

        Uses Discrete Algebraic Ricatti Equation to get P

        :meta public:
        """
        task = self.bundle.task
        if self.Acontroller is None:
            self.Acontroller = task.A
        if self.Bcontroller is None:
            self.Bcontroller = task.B
        A, B = self.Acontroller, self.Bcontroller
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        invPart = scipy.linalg.inv((self.R + B.T @ P @ B))
        K = invPart @ B.T @ P @ A
        self.policy.set_feedback_gain(K)
