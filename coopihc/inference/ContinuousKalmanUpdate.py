import numpy
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ContinuousKalmanUpdate(BaseInferenceEngine):
    def __init__(self):
        super().__init__()
        self.fmd_flag = False
        self.K_flag = False

    def set_forward_model_dynamics(self, A, B, C):
        self.fmd_flag = True
        self.A = A
        self.B = B
        self.C = C

    def set_K(self, K):
        self.K_flag = True
        self.K = K

    def infer(self):
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
