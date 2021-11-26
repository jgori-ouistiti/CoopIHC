import numpy
from coopihc.agents import BaseAgent
import coopihc.observation
from coopihc.space import State, StateElement
from coopihc.policy import LinearFeedback
from coopihc.inference import ContinuousKalmanUpdate
import copy
import gym.spaces


# Infinite Horizon Continuous Time LQG controller, based on Phillis 1985
class IHCT_LQGController(BaseAgent):
    """An Infinite Horizon (Steady-state) LQG controller, based on Phillis 1985, using notations from Qian 2013."""

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

        class RuleObswithLQreward(coopihc.observation.RuleObservationEngine):
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
                    coopihc.observation.observation_linear_combination,
                    (C,),
                )
            }
            extradeterministicrules = {}
            extradeterministicrules.update(obs_matrix)

            # extraprobabilisticrule
            agn_rule = {
                ("task_state", "x"): (
                    coopihc.observation.additive_gaussian_noise,
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

    def reset(self, dic=None):
        if dic is None:
            super().reset()

        if dic is not None:
            super().reset(dic=dic)

    class _MContainer:
        """The purpose of this container is to facilitate common manipulations of the matrices of the LQG problem, as well as potentially storing their evolution. (not implemented yet)"""

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
        """Returns norm of an equation of the form AX + XA.T + BXB.T + C = 0"""
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
        def wrapped(*args, **kwargs):
            wrapped.calls += 1
            return f(*args, **kwargs)

        wrapped.calls = 0
        return wrapped

    @counted_decorator
    def _check_KL(self, Knorm, Lnorm, K, L, matrices):
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
