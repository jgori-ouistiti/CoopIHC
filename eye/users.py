from core.agents import BaseAgent
from core.observation import RuleObservationEngine, base_user_engine_specification
from core.policy import LinearFeedback
from core.space import State, StateElement, Space
from core.inference import LinearGaussianContinuous
import eye.noise
from scipy.linalg import toeplitz

import gym
import numpy


class ChenEye(BaseAgent):
    """The Eye model of Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021., --> with a real 2D implementation.

    :param perceptualnoise: noise in the eye observation process

    :meta public:
    """

    def __init__(self, perceptualnoise, oculomotornoise, dimension=2, *args, **kwargs):
        self.dimension = dimension
        self.perceptualnoise = perceptualnoise
        self.oculomotornoise = oculomotornoise

        agent_policy = kwargs.get("agent_policy")
        if agent_policy is None:
            action_state = State()
            action_state["action"] = StateElement(
                values=None,
                spaces=Space(
                    [
                        -numpy.ones((dimension,), dtype=numpy.float32),
                        numpy.ones((dimension,), dtype=numpy.float32),
                    ]
                ),
                clipping_mode="warning",
            )

            def noise_function(self, action, observation, *args):
                oculomotornoise = args[0]
                noise_obs = State()
                noise_obs["task_state"] = State()
                noise_obs["task_state"]["targets"] = action
                noise_obs["task_state"]["fixation"] = observation["task_state"][
                    "fixation"
                ]
                noise = self.host.eccentric_noise_gen(noise_obs, oculomotornoise)[0]
                return noise

            agent_policy = LinearFeedback(
                ("user_state", "belief"),
                0,
                action_state,
                noise_function=noise_function,
                noise_function_args=(self.oculomotornoise,),
            )

        observation_engine = kwargs.get("observation_engine")

        if observation_engine is None:
            extraprobabilisticrules = {
                ("task_state", "targets"): (self._eccentric_noise_gen, ())
            }

            observation_engine = RuleObservationEngine(
                base_user_engine_specification,
                extraprobabilisticrules=extraprobabilisticrules,
            )

        inference_engine = kwargs.get("inference_engine")
        if inference_engine is None:

            def provide_likelihood_to_inference_engine(self):
                ## specify here what part of the internal observation will be used as observation sample
                observation = self.buffer[-1]
                mu = observation["task_state"]["targets"]["values"][0]
                sigma = self.host.Sigma + 1e-4 * toeplitz(
                    [1] + [0.1 for i in range(self.host.dimension - 1)]
                )  # Avoid null sigma
                return mu, sigma

            inference_engine = LinearGaussianContinuous(
                provide_likelihood_to_inference_engine
            )

        state = kwargs.get("state")
        if state is None:
            belief = StateElement(
                values=None,
                spaces=[
                    Space(
                        [
                            -numpy.ones((self.dimension,), dtype=numpy.float32),
                            numpy.ones((self.dimension,), dtype=numpy.float32),
                        ]
                    ),
                    Space(
                        [
                            -numpy.inf
                            * numpy.ones(
                                (self.dimension, self.dimension), dtype=numpy.float32
                            ),
                            numpy.inf
                            * numpy.ones(
                                (self.dimension, self.dimension), dtype=numpy.float32
                            ),
                        ]
                    ),
                ],
                clipping_mode="warning",
            )
            state = State()
            state["belief"] = belief

        super().__init__(
            "user",
            agent_state=state,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
        )

    def finit(self):
        pass

    def reset(self, dic=None):
        """Reset the fixation at the center (0;0), reset the prior belief

        :meta public:
        """
        # call reset before so that the code below gets executed when no arguments are provided to reset (no forced reset)
        if dic is None:
            super().reset()

        # Initialize here the start position of the eye as well as initial uncertainty
        observation = State()
        observation["task_state"] = State()
        observation["task_state"]["targets"] = self.bundle.task.state["targets"]
        observation["task_state"]["fixation"] = self.bundle.task.state["fixation"]
        # Initialize with a huge Gaussian noise so that the first observation massively outweights the prior. Put more weight on the pure variance components to ensure that it will behave well.
        Sigma = toeplitz([1000] + [100 for i in range(self.dimension - 1)])
        self.state["belief"]["values"] = [
            numpy.array([0 for i in range(self.dimension)]),
            Sigma,
        ]

        # call reset after, so that the code below gets overwritten by arguments provided in reset (forced reset)
        if dic is not None:
            super().reset(dic=dic)

    def _eccentric_noise_gen(self, _obs, observation, *args):
        # clip the noisy observation --> this makes sense e.g. on a screen, were we know that the observation has to be on screen at all times.
        noise = self.eccentric_noise_gen(observation, *args)[0]
        return (
            numpy.clip(
                _obs[0] + noise,
                observation["task_state"]["targets"]["spaces"][0].low,
                observation["task_state"]["targets"]["spaces"][0].high,
            ),
            noise,
        )

    def eccentric_noise_gen(self, observation, *args):
        """Define eccentric noise that will be used in the noisyrule

        :return: noise samples drawn according to the 2D centered Gaussian with covariance matrix specified by the eccentric noise.

        :meta public:
        """
        try:
            noise_std = args[0]
        except IndexError:
            noise_std = self.perceptualnoise

        target = observation["task_state"]["targets"]
        position = observation["task_state"]["fixation"]
        Sigma = eye.noise.eccentric_noise(target, position, noise_std)
        noise = numpy.random.multivariate_normal(
            numpy.zeros(shape=(self.dimension,)), Sigma
        )
        self.Sigma = Sigma
        return noise, Sigma

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"
        try:
            axtask, axuser, axassistant = args
            self.inference_engine.render(axtask, axuser, axassistant, mode=mode)
        except ValueError:
            self.inference_engine.render(mode=mode)
