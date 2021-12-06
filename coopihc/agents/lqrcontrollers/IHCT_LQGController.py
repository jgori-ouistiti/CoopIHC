import numpy
import copy
import gym.spaces

from coopihc.agents.BaseAgent import BaseAgent
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import observation_linear_combination
from coopihc.observation.utils import additive_gaussian_noise
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.policy.LinearFeedback import LinearFeedback
from coopihc.inference.ContinuousKalmanUpdate import ContinuousKalmanUpdate


# Infinite Horizon Continuous Time LQG controller, based on Phillis 1985
class IHCT_LQGController(BaseAgent):
    """Infinite Horizon Continuous Time LQ Gaussian Controller.

    An Infinite Horizon (Steady-state) LQG controller, based on Phillis 1985 [Phillis1985]_, using notations from Qian 2013 [Qian2013]_.

    .. math::

        \\begin{align*}
        dx & = (Ax + Bu)dt + Fxd \\beta + Yud \\gamma + \\Gamma d\omega \\\\
        dy & = Cxdt + Dd\\xi \\\\
        d\\hat{x} & = (A \\hat{x} + Bu) dt + K (dy - C\\hat{x}dt) \\\\
        u & = -L\\hat{x} \\\\
        \\tilde{x} & = x - \\hat{x} \\\\
        J & \simeq \\mathbb{E} [\\tilde{x}^T U \\tilde{x} + x^TQx + u^TRu]
        \\end{align*}

    .. [Phillis1985] Phillis, Y. "Controller design of systems with multiplicative noise." IEEE Transactions on Automatic Control 30.10 (1985): 1017-1019. `Doc here<https://ieeexplore.ieee.org/abstract/document/1103828>`
    .. [Qian2013] Qian, Ning, et al. "Movement duration, Fitts's law, and an infinite-horizon optimal feedback control model for biological motor systems." Neural computation 25.3 (2013): 697-724. `Doc here<https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.4312&rep=rep1&type=pdf>`

    .. warning:: 

        watch out for :math:`\\gamma` and :math:`\\Gamma`

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
    :param Gamma: Additive independent Noise weight
    :type Gamma: numpy.ndarray
    :param D: Observation noise matrix
    :type D: numpy.ndarray
    :param noise: whether or not to have, defaults to "on"
    :type noise: str, optional
    """

    def __init__(
        self, role, timestep, Q, R, U, C, Gamma, D, *args, noise="on", **kwargs
    ):

        self.C = C
        self.Gamma = numpy.array(Gamma)
        self.Q = Q
        self.R = R
        self.U = U
        self.D = D
        self.timestep = timestep
        self.role = role
        self.timespace = "continuous"

        # Initialize Random Kalmain gains
        self.K = numpy.random.rand(*C.T.shape)
        self.L = numpy.random.rand(1, Q.shape[1])
        self.noise = noise

        # =================== Linear Feedback Policy ==========
        self.gamma = kwargs.get("Gamma")
        self.mu = kwargs.get("Mu")
        self.sigma = kwargs.get("Sigma")

        agent_policy = kwargs.get("agent_policy")
        if agent_policy is None:
            action_state = State()
            action_state["action"] = StateElement(
                values=[None],
                spaces=[gym.spaces.Box(-numpy.inf, numpy.inf, shape=(1,))],
                possible_values=[[None]],
            )

            def shaped_gaussian_noise(self, action, observation, *args):
                gamma, mu, sigma = args[:3]
                if gamma is None:
                    return 0
                if sigma is None:
                    sigma = numpy.sqrt(self.host.timestep)  # Wiener process
                if mu is None:
                    mu = 0
                noise = gamma * numpy.random.normal(mu, sigma)
                return noise

            class LFwithreward(LinearFeedback):
                def __init__(self, R, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.R = R

                def sample(self):
                    action, _ = super().sample()
                    return (
                        action,
                        action["values"][0].T @ action["values"][0] * 1e3,
                    )

            agent_policy = LFwithreward(
                self.R,
                ("user_state", "xhat"),
                0,
                action_state,
                noise_function=shaped_gaussian_noise,
                noise_function_args=(self.gamma, self.mu, self.sigma),
            )

            # agent_policy = LinearFeedback(
            #     ('user_state','xhat'),
            #     0,
            #     action_state,
            #     noise_function = shaped_gaussian_noise,
            #     noise_function_args = (self.gamma, self.mu, self.sigma)
            #             )

        # =========== Observation Engine: Task state unobservable, internal estimates observable ============

        observation_engine = kwargs.get("observation_engine")

        class RuleObswithLQreward(RuleObservationEngine):
            def __init__(self, Q, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Q = Q

            def observe(self, game_state):
                observation, _ = super().observe(game_state)
                x = observation["task_state"]["x"]["values"][0]
                reward = x.T @ self.Q @ x
                return observation, reward

        if observation_engine is None:
            user_engine_specification = [
                ("turn_index", "all"),
                ("task_state", "all"),
                ("user_state", "all"),
                ("assistant_state", None),
                ("user_action", "all"),
                ("assistant_action", "all"),
            ]

            obs_matrix = {
                ("task_state", "x"): (
                    observation_linear_combination,
                    (C,),
                )
            }
            extradeterministicrules = {}
            extradeterministicrules.update(obs_matrix)

            # extraprobabilisticrule
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
            # observation_engine = RuleObservationEngine(deterministic_specification = user_engine_specification, extradeterministicrules = extradeterministicrules, extraprobabilisticrules = extraprobabilisticrules)

        inference_engine = kwargs.get("inference_engine")
        if inference_engine is None:
            inference_engine = ContinuousKalmanUpdate()

        state = kwargs.get("state")
        if state is None:
            pass

        super().__init__(
            "user",
            state=state,
            policy=agent_policy,
            observation_engine=observation_engine,
            inference_engine=inference_engine,
        )

    def finit(self):
        """finit

        1. Create an :math:`\\hat{x}` state;
        2. attach the model dynamics to the inference engine
        3. compute K and L;
        4. set K and L in inference engine and policy
        """
        task = self.bundle.task
        # ---- init xhat state
        self.state["xhat"] = copy.deepcopy(self.bundle.task.state["x"])

        # ---- Attach the model dynamics to the inference engine.
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

    # untested
    def reset(self):
        pass

    # def reset(self, dic=None):
    #     if dic is None:
    #         super().reset()

    #     if dic is not None:
    #         super().reset(dic=dic)

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
