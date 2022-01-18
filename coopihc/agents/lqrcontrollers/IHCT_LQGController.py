import numpy
import copy
import warnings
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.policy.LinearFeedback import LinearFeedback
from coopihc.inference.ContinuousKalmanUpdate import ContinuousKalmanUpdate


# Infinite Horizon Continuous Time LQG controller, based on Phillis 1985
class IHCT_LQGController(BaseAgent):
    """Infinite Horizon Continuous Time LQ Gaussian Controller.

    An Infinite Horizon (Steady-state) LQG controller, based on Phillis 1985 [Phillis1985]_ and Qian 2013 [Qian2013]_.

    For the a task where state 'x' follows a linear noisy dynamic:

    .. math::

        \\begin{align*}
        dx & = (Ax + Bu)dt + Fxd \\beta + G d\omega, \\\\
        \\end{align*}

    the LQG controller produces the following observations dy and commands u with cost J:

    .. math::

        \\begin{align*}
        dy & = Cxdt + Dd\\xi \\\\
        d\\hat{x} & = (A \\hat{x} + Bu) dt + K (dy - C\\hat{x}dt) \\\\
        u & = -\\Gamma  L\\hat{x} d\\gamma \\\\
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
    :param Gamma: Additive independent Noise weight shaping matrix
    :type Gamma: numpy.ndarray
    :param mu: mean of :math:`d\\gamma`, defaults to 0
    :type mu: int, optional
    :param sigma: stdev of :math:`d\\gamma`, defaults to 0
    :type sigma: int, optional
    :param Acontroller: Representation of A for the agent. If None, the agent representation of A is equal to the task A, defaults to None.
    :type Acontroller: numpy.ndarray, optional
    :param Bcontroller: Representation of B for the agent. If None, the agent representation of B is equal to the task B, defaults to None.
    :type Bcontroller: numpy.ndarray, optional
    :param Ccontroller: Representation of C for the agent. If None, the agent representation of C is equal to the task C, defaults to None.
    :type Ccontroller: numpy.ndarray, optional

    
    
    
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
        Gamma="id",
        mu=0,
        sigma=0,
        Acontroller=None,
        Bcontroller=None,
        Ccontroller=None,
        **kwargs
    ):
        """__init__ [summary]

        [extended_summary]

        :param role: [description]
        :type role: [type]
        :param timestep: [description]
        :type timestep: [type]
        :param Q: [description]
        :type Q: [type]
        :param R: [description]
        :type R: [type]
        :param U: [description]
        :type U: [type]
        :param C: [description]
        :type C: [type]
        :param D: [description]
        :type D: [type]
        :param noise: [description], defaults to "on"
        :type noise: str, optional
        :param Gamma: [description], defaults to "id"
        :type Gamma: str, optional
        :param mu: [description], defaults to 0
        :type mu: int, optional
        :param sigma: [description], defaults to 0
        :type sigma: int, optional
        :param Acontroller: [description], defaults to None
        :type Acontroller: [type], optional
        :param Bcontroller: [description], defaults to None
        :type Bcontroller: [type], optional
        :param Ccontroller: [description], defaults to None
        :type Ccontroller: [type], optional
        :return: [description]
        :rtype: [type]
        """

        self.C = C
        self.Gamma = numpy.array(Gamma)
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
        self.Gamma = Gamma
        self.mu = mu
        self.sigma = sigma

        self.Acontroller = Acontroller
        self.Bcontroller = Bcontroller
        self.Ccontroller = Ccontroller

        # =================== Linear Feedback Policy ==========

        action_state = State()
        action_state["action"] = StateElement(
            numpy.zeros((1, 1)),
            Space([numpy.full((1, 1), -numpy.inf), numpy.full((1, 1), numpy.inf)]),
        )
        # Gaussian noise on action
        def shaped_gaussian_noise(action, observation, Gamma, mu, sigma):
            if Gamma is None:
                return action
            if sigma is None:
                sigma = numpy.sqrt(self.host.timestep)  # Wiener process
            if mu is None:
                mu = 0
            noisy_action = Gamma @ action * numpy.random.normal(mu, sigma)
            return noisy_action

        # Linear Feedback with LQ reward
        class LFwithLQreward(LinearFeedback):
            def __init__(self, R, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.R = R

            def sample(self, observation=None):
                action, _ = super().sample(observation=observation)
                return (
                    action,
                    (action.T @ self.R @ action).squeeze().tolist(),
                )

        agent_policy = LFwithLQreward(
            self.R,
            action_state,
            ("user_state", "xhat"),
            noise_function=shaped_gaussian_noise,
            noise_func_args=(self.Gamma, self.mu, self.sigma),
        )

        # =========== Observation Engine ==============
        # Rule Observation Engine with LQ reward
        class RuleObswithLQreward(RuleObservationEngine):
            def __init__(self, Q, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Q = Q

            def observe(self, game_state=None):
                observation, _ = super().observe(game_state=game_state)
                x = observation["task_state"]["x"]
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
        if all(
            [self.Acontroller, self.Bcontroller, self.Ccontroller] != [None, None, None]
        ):
            inference_engine.set_forward_model_dynamics(self.A_c, self.B_c, self.C)
        else:
            if any(
                [self.Acontroller, self.Bcontroller, self.Ccontroller]
                != [None, None, None]
            ):
                warnings.warn(
                    Warning(
                        "The controller matrices A, B, C you provided are not accounted for. You have to define all three of them to account for them."
                    )
                )

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
        )

    def finit(self):
        """finit

        1. Create an :math:`\\hat{x}` state;
        2. attach the model dynamics to the inference engine if needed
        3. compute K and L;
        4. set K and L in inference engine and policy
        """
        task = self.bundle.task
        # ---- init xhat state
        self.state["xhat"] = copy.deepcopy(self.bundle.task.state["x"])

        # ---- Attach the model dynamics to the inference engine.
        if not self.inference.engine.fmd_flag:
            self.A_c, self.B_c, self.G = task.A_c, task.B_c, task.G
            self.inference_engine.set_forward_model_dynamics(self.A_c, self.B_c, self.C)

        # ---- Set K and L up
        mc = self._MContainer(
            self.A_c,
            self.B_c,
            self.C,
            self.D,
            self.G,
            self.Gamma,
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

        def __init__(self, A, B, C, D, G, Gamma, Q, R, U):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.G = G
            self.Gamma = Gamma
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
                self.Gamma,
                self.Q,
                self.R,
                self.U,
            )

    def _compute_Kalman_matrices(self, matrices, N=20):
        """Compute K and L

        K and L are computed according to the algorithm described in [Qian2013]_ with some minor tweaks. K and L are obtained recursively, where more and more precise estimates are obtained. At first N iterations are performed, if that fails to converge, N is grown as :math:`N^{1.3}` and K and L are recomputed.

        :param matrices: (A, B, C, D, G, Gamma, Q, R, U)
        :type matrices: tuple(numpy.ndarray)
        :param N: max iterations of the algorithm on first try, defaults to 20
        :type N: int, optional
        :return: (K, L)
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        A, B, C, D, G, Gamma, Q, R, U = matrices
        Y = B @ numpy.array(Gamma).reshape(1, -1)
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
