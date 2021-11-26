import numpy
from coopihc.agents import BaseAgent
from coopihc.space import State, StateElement
from coopihc.policy import LinearFeedback
from coopihc.observation import (
    RuleObservationEngine,
    base_task_engine_specification,
)
import gym.spaces


class LQRController(BaseAgent):
    """
    .. math::

        action =  -K X + \Gamma  \mathcal{N}(\overline{\mu}, \Sigma)

    """

    def __init__(self, role, Q, R, *args, **kwargs):

        self.R = R
        self.Q = Q
        self.role = role

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

            agent_policy = LinearFeedback(
                ("task_state", "x"),
                0,
                action_state,
                noise_function=shaped_gaussian_noise,
                noise_function_args=(self.gamma, self.mu, self.sigma),
            )

        observation_engine = kwargs.get("observation_engine")
        if observation_engine is None:
            observation_engine = RuleObservationEngine(base_task_engine_specification)

        inference_engine = kwargs.get("inference_engine")
        if inference_engine is None:
            pass

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

    def reset(self, dic=None):
        if dic is None:
            super().reset()

        # Nothing to reset

        if dic is not None:
            super().reset(dic=dic)

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is None:
                self.ax = axuser
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Action")
            if self.action["values"][0]:
                self.ax.plot(
                    self.bundle.task.turn * self.bundle.task.timestep,
                    self.action["values"][0],
                    "bo",
                )
        if "text" in mode:
            print("Action")
            print(self.action)
