import numpy
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ContinuousKalmanUpdate(BaseInferenceEngine):
    """ContinuousKalmanUpdate

    An inference engine which estimates the new state according to a continuous kalman filter, where state transition dynamics and kalman gains are provided externally.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fmd_flag = False
        self.K_flag = False

    def set_forward_model_dynamics(self, A, B, C):
        """set forward model dynamics

        Call this externally to supply the linear dynamic matrices that describe the deterministic part of the state transitions:

        .. math::

            \\begin{align*}
            d\\hat{x} = A\\hat{x}dt + Budt \\\\
            dy = C \\hat{x} dt
            \\end{align*}

        :param A: see equation above
        :type A: numpy.ndarray
        :param B: see equation above
        :type B: numpy.ndarray
        :param C: see equation above
        :type C: numpy.ndarray
        """
        self.fmd_flag = True
        self.A = A
        self.B = B
        self.C = C

    def set_K(self, K):
        """set_K

        Set the Kalman gain

        :param K: Kalman Gain
        :type K: numpy.ndarray
        """
        self.K_flag = True
        self.K = K

    def infer(self):
        """infer

        Infer the state based on the observation.

        :return: (new state, reward)
        :rtype: tuple(py:class:`State<coopihc.space.State.State>`, float)
        """
        if not self.fmd_flag:
            raise RuntimeError(
                "You have to set the forward model dynamics, by calling the set_forward_model_dynamics() method with inference engine {}  before using it".format(
                    type(self).__name__
                )
            )
        if not self.K_flag:
            raise RuntimeError(
                "You have to set the K Matrix, by calling the set_K() method with inference engine {}  before using it".format(
                    type(self).__name__
                )
            )
        observation = self.observation
        dy = observation["task_state"]["x"]["values"][0] * self.host.timestep

        if isinstance(dy, list):
            dy = dy[0]
        if not isinstance(dy, numpy.ndarray):
            raise TypeError(
                "Substate Xhat of {} is expected to be of type numpy.ndarray".format(
                    type(self.host).__name__
                )
            )

        state = observation["{}_state".format(self.host.role)]
        u = self.action["values"][0]

        xhat = state["xhat"]["values"][0]

        xhat = xhat.reshape(-1, 1)
        u = u.reshape(-1, 1)
        deltaxhat = (self.A @ xhat + self.B @ u) * self.host.timestep + self.K @ (
            dy - self.C @ xhat * self.host.timestep
        )
        xhat += deltaxhat
        state["xhat"]["values"] = xhat

        # Here, we use the classical definition of rewards in the LQG setup, but this requires having the true value of the state. This may or may not realistic...
        # ====================== Rewards ===============

        x = self.host.bundle.task.state["x"]["values"][0]
        reward = (x - xhat).T @ self.host.U @ (x - xhat)

        return state, reward
