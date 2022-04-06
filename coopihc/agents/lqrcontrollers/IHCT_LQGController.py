import numpy
import copy
import warnings
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.policy.LinearFeedback import LinearFeedback
from coopihc.inference.ContinuousKalmanUpdate import ContinuousKalmanUpdate


# Infinite Horizon Continuous Time LQG controller, based on Phillis 1985
class IHCT_LQGController(BaseAgent):
    """Infinite Horizon Continuous Time LQ Gaussian Controller.

    An Infinite Horizon (Steady-state) LQG controller, based on [Phillis1985]_ and [Qian2013]_.

    For the a task where state 'x' follows a linear noisy dynamic:

    .. math::

        \\begin{align}
            x(+.) = (Ax(.) + Bu(.))dt + Fx(.).d\\beta + G.d\\omega + Hu(.)d\\gamma \\\\
        \\end{align}

    the LQG controller produces the following observations dy and commands u minimizing cost J:

    .. math::

        \\begin{align*}
        dy & = Cxdt + Dd\\xi \\\\
        d\\hat{x} & = (A \\hat{x} + Bu) dt + K (dy - C\\hat{x}dt) \\\\
        u & = - L\\hat{x} \\\\
        \\tilde{x} & = x - \\hat{x} \\\\
        J & \simeq \\mathbb{E} [\\tilde{x}^T U \\tilde{x} + x^TQx + u^TRu]
        \\end{align*}

    .. [Phillis1985] Phillis, Y. "Controller design of systems with multiplicative noise." IEEE Transactions on Automatic Control 30.10 (1985): 1017-1019. `Link <https://ieeexplore.ieee.org/abstract/document/1103828>`_
    .. [Qian2013] Qian, Ning, et al. "Movement duration, Fitts's law, and an infinite-horizon optimal feedback control model for biological motor systems." Neural computation 25.3 (2013): 697-724. `Link <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.4312&rep=rep1&type=pdf>`_

    :param role: "user" or "assistant"
    :type role: string
    :param timestep: duration of timestep
    :type timestep: float
    :param Q: State cost
    :type Q: numpy.ndarray
    :param R: Control cost
    :type R: numpy.ndarray
    :param U: Estimation error cost
    :type U: numpy.ndarray
    :param C: Observation matrix
    :type C: numpy.ndarray
    :param D: Observation noise matrix
    :type D: numpy.ndarray
    :param noise: whether or not to have, defaults to "on"
    :type noise: str, optional
    :param Acontroller: Representation of A for the agent. If None, the agent representation of A is equal to the task A, defaults to None.
    :type Acontroller: numpy.ndarray, optional
    :param Bcontroller: Representation of B for the agent. If None, the agent representation of B is equal to the task B, defaults to None.
    :type Bcontroller: numpy.ndarray, optional
    :param Fcontroller: Representation of F for the agent. If None, the agent representation of F is equal to the task F, defaults to None.
    :type Fcontroller: numpy.ndarray, optional
    :param Gcontroller: Representation of G for the agent. If None, the agent representation of G is equal to the task G, defaults to None.
    :type Gcontroller: numpy.ndarray, optional
    :param Hcontroller: Representation of H for the agent. If None, the agent representation of H is equal to the task H, defaults to None.
    :type Hcontroller: numpy.ndarray, optional

    
    
    
    """

    def __init__(
        self,
        role,
        timestep,
        Q,
        R,
        U,
        C,
        D,
        *args,
        noise="on",
        Acontroller=None,
        Bcontroller=None,
        F=None,
        G=None,
        H=None,
        **kwargs
    ):
        self.C = C
        self.Q = Q
        self.R = R
        self.U = U
        self.D = D
        self.timestep = timestep
        self.role = role

        # Initialize Random Kalmain gains
        self.K = numpy.random.rand(*C.T.shape)
        self.L = numpy.random.rand(1, Q.shape[1])

        self.noise = noise

        self.Acontroller = Acontroller
        self.Bcontroller = Bcontroller
        self.Fcontroller = F
        self.Gcontroller = G
        self.Hcontroller = H

        # =================== Linear Feedback Policy ==========

        action_state = State()
        action_state["action"] = array_element(shape=(1, 1))

        # StateElement(
        #     numpy.zeros((1, 1)),
        #     Space(
        #         [numpy.full((1, 1), -numpy.inf), numpy.full((1, 1), numpy.inf)],
        #         "continuous",
        #     ),
        # )

        # Linear Feedback with LQ reward
        class LFwithLQreward(LinearFeedback):
            def __init__(self, R, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.R = R

            def sample(self, agent_observation=None, agent_state=None):
                action, _ = super().sample(agent_observation=agent_observation)
                return (
                    action,
                    (action.T @ self.R @ action).squeeze().tolist(),
                )

        agent_policy = LFwithLQreward(
            self.R,
            action_state,
            ("user_state", "xhat"),
        )

        # =========== Observation Engine ==============
        # Rule Observation Engine with LQ reward
        class RuleObswithLQreward(RuleObservationEngine):
            def __init__(self, Q, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Q = Q

            def observe(self, game_state=None):
                observation, _ = super().observe(game_state=game_state)
                x = observation["task_state"]["x"].view(numpy.ndarray)
                reward = x.T @ self.Q @ x
                return observation, reward

        # Base Spec
        user_engine_specification = [
            ("game_info", "all"),
            ("task_state", "x"),
            ("user_state", "all"),
            ("assistant_state", None),
            ("user_action", "all"),
            ("assistant_action", None),
        ]

        # Add rule for matrix observation y += Cx
        def observation_linear_combination(_obs, game_state, C):
            return C @ _obs

        C_rule = {
            ("task_state", "x"): (
                observation_linear_combination,
                (C,),
            )
        }
        extradeterministicrules = {}
        extradeterministicrules.update(C_rule)

        # Add rule for noisy observation y += D * epsilon ~ N(mu, sigma)

        def additive_gaussian_noise(_obs, gamestate, D, *args):
            try:
                mu, sigma = args
            except ValueError:
                mu, sigma = numpy.zeros(_obs.shape), numpy.eye(max(_obs.shape))
            return _obs + D @ numpy.random.multivariate_normal(
                mu, sigma, size=1
            ).reshape(-1, 1)

        # Instantiate previous rule so that epsilon ~ N(0, sqrt(dt))
        agn_rule = {
            ("task_state", "x"): (
                additive_gaussian_noise,
                (
                    D,
                    numpy.zeros((C.shape[0], 1)).reshape(
                        -1,
                    ),
                    numpy.sqrt(timestep) * numpy.eye(C.shape[0]),
                ),
            )
        }

        extraprobabilisticrules = {}
        extraprobabilisticrules.update(agn_rule)

        observation_engine = RuleObswithLQreward(
            self.Q,
            deterministic_specification=user_engine_specification,
            extradeterministicrules=extradeterministicrules,
            extraprobabilisticrules=extraprobabilisticrules,
        )

        # ======================= Inference Engine
        inference_engine = ContinuousKalmanUpdate()

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
        )

    def finit(self):
        """Get and compute needed matrices.

        0. Take A, B, F, G, H from task if not provided by the end-user
        1. Create an :math:`\\hat{x}` state;
        2. attach the model dynamics to the inference engine if needed
        3. compute K and L;
        4. set K and L in inference engine and policy
        """

        task = self.bundle.task

        for elem, taskelem in zip(
            [
                "Acontroller",
                "Bcontroller",
                "Fcontroller",
                "Gcontroller",
                "Hcontroller",
            ],
            [task.A, task.B, task.F, task.G, task.H],
        ):
            if getattr(self, elem) == None:
                setattr(self, elem, taskelem)

        # ---- init xhat state
        self.state["xhat"] = copy.deepcopy(self.bundle.task.state["x"])

        # ---- Attach the model dynamics to the inference engine.
        if not self.inference_engine.fmd_flag:

            self.inference_engine.set_forward_model_dynamics(
                self.Acontroller, self.Bcontroller, self.C
            )

        # ---- Set K and L up
        mc = self._MContainer(
            self.Acontroller,
            self.Bcontroller,
            self.C,
            self.D,
            self.Gcontroller,
            self.Hcontroller,
            self.Q,
            self.R,
            self.U,
        )
        self.K, self.L = self._compute_Kalman_matrices(mc.pass_args())
        self.inference_engine.set_K(self.K)
        self.policy.set_feedback_gain(self.L)

    class _MContainer:
        """Matrix container

        The purpose of this container is to facilitate common manipulations of the matrices of the LQG problem, as well as potentially storing their evolution. (not implemented yet)
        """

        def __init__(self, A, B, C, D, G, H, Q, R, U):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.G = G
            self.H = H
            self.Q = Q
            self.R = R
            self.U = U
            self._check_matrices()

        def _check_matrices(self):
            # Not implemented yet
            pass

        def pass_args(self):
            return (
                self.A,
                self.B,
                self.C,
                self.D,
                self.G,
                self.H,
                self.Q,
                self.R,
                self.U,
            )

    def _compute_Kalman_matrices(self, matrices, N=20):
        """Compute K and L

        K and L are computed according to the algorithm described in [Qian2013]_ with some minor tweaks. K and L are obtained recursively, where more and more precise estimates are obtained. At first N iterations are performed, if that fails to converge, N is grown as :math:`N^{1.3}` and K and L are recomputed.

        :param matrices: (A, B, C, D, G, H, Q, R, U)
        :type matrices: tuple(numpy.ndarray)
        :param N: max iterations of the algorithm on first try, defaults to 20
        :type N: int, optional
        :return: (K, L)
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        A, B, C, D, G, H, Q, R, U = matrices
        Y = B @ H.reshape(1, -1)
        Lnorm = []
        Knorm = []
        K = numpy.random.rand(*C.T.shape)
        L = numpy.random.rand(1, A.shape[1])
        for i in range(N):
            Lnorm.append(numpy.linalg.norm(L))
            Knorm.append(numpy.linalg.norm(K))

            n, m = A.shape
            Abar = numpy.block([[A - B @ L, B @ L], [numpy.zeros((n, m)), A - K @ C]])

            Ybar = numpy.block([[-Y @ L, Y @ L], [-Y @ L, Y @ L]])

            Gbar = numpy.block(
                [[G, numpy.zeros((G.shape[0], D.shape[1]))], [G, -K @ D]]
            )

            V = numpy.block(
                [
                    [Q + L.T @ R @ L, -L.T @ R @ L],
                    [-L.T @ R @ L, L.T @ R @ L + U],
                ]
            )

            P, p_res = self._LinRicatti(Abar, Ybar, Gbar @ Gbar.T)
            S, s_res = self._LinRicatti(Abar.T, Ybar.T, V)

            P22 = P[n:, n:]
            S11 = S[:n, :n]
            S22 = S[n:, n:]

            K = P22 @ C.T @ numpy.linalg.pinv(D @ D.T)
            L = numpy.linalg.pinv(R + Y.T @ (S11 + S22) @ Y) @ B.T @ S11

        K, L = self._check_KL(Knorm, Lnorm, K, L, matrices)
        return K, L

    def _LinRicatti(self, A, B, C):
        """_LinRicatti [summary]

        Returns norm of an equation of the form

        .. math ::

            \\begin{align}
            AX + XA.T + BXB.T + C = 0
            \\end{align}



        :param A: See Equation above
        :type A: numpy.ndarray
        :param B: See Equation above
        :type B: numpy.ndarray
        :param C: See Equation above
        :type C: numpy.ndarray
        :return: X, residue
        :rtype: tuple(numpy.ndarray, float)
        """
        #
        n, m = A.shape
        nc, mc = C.shape
        if n != m:
            print("Matrix A has to be square")
            return -1
        M = (
            numpy.kron(numpy.identity(n), A)
            + numpy.kron(A, numpy.identity(n))
            + numpy.kron(B, B)
        )
        C = C.reshape(-1, 1)
        X = -numpy.linalg.pinv(M) @ C
        X = X.reshape(n, n)
        C = C.reshape(nc, mc)
        res = numpy.linalg.norm(A @ X + X @ A.T + B @ X @ B.T + C)
        return X, res

    # Counting decorator

    def counted_decorator(f):
        """counted_decorator

        Decorator that counts the number of times function f has been called

        :param f: decorated function
        :type f: function
        """

        def wrapped(*args, **kwargs):
            wrapped.calls += 1
            return f(*args, **kwargs)

        wrapped.calls = 0
        return wrapped

    @counted_decorator
    def _check_KL(self, Knorm, Lnorm, K, L, matrices):
        """Check K and L convergence

        Checks whether K and L have converged, by looking at the variations over the last 5 iterations.

        :param Knorm: list of the norms of K on each iteration
        :type Knorm: list(numpy.array)
        :param Lnorm: list of the norms of L on each iteration
        :type Lnorm: list(numpy.array)
        :param K: See Equation in class docstring
        :type K: numpy.array
        :param L: See Equation in class docstring
        :type L: numpy.array
        :param matrices: Matrix container
        :type matrices: _MContainer
        :return: K and L
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        average_delta = numpy.convolve(
            numpy.diff(Lnorm) + numpy.diff(Knorm),
            numpy.ones(5) / 5,
            mode="full",
        )[-5]
        if average_delta > 0.01:  # Arbitrary threshold
            print(
                "Warning: the K and L matrices computations did not converge. Retrying with different starting point and a N={:d} search".format(
                    int(20 * 1.3 ** self._check_KL.calls)
                )
            )
            K, L = self._compute_Kalman_matrices(
                matrices, N=int(20 * 1.3 ** self.check_KL.calls)
            )
        else:
            return K, L
