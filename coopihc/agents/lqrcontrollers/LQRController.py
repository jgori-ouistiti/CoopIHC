import numpy


from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.policy.LinearFeedback import LinearFeedback
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_task_engine_specification


class LQRController(BaseAgent):
    """A Linear Quadratic Regulator.

    Tested only on 1d output. This agent will read a state named 'x' from the task, and produce actions according to:

    .. math::

        action =  -K x + \Gamma  \mathcal{N}(\overline{\mu}, \Sigma)

    where K is the so-called feedback gain, which has to be specified externally. For an example, see the :py:class:`coopihc.agents.lqrcontrollers.FHDT_LQRController.FHDT_LQRController` source code.

    .. note::

        This class is meant to be subclassed

    :param role: "user" or "assistant"
    :type role: string
    :param Q: State cost
    :type Q: numpy.ndarray
    :param R: Control cost
    :type R: numpy.ndarray
    :param Gamma: Noise weight, defaults to None
    :type Gamma: float, optional
    :param Mu: Noise mean, defaults to None
    :type Mu: float, optional
    :param sigma: Noise variance, defaults to None
    :type sigma: float, optional
    """

    def __init__(self, role, Q, R, *args, Gamma=None, Mu=None, Sigma=None, **kwargs):

        self.R = R
        self.Q = Q
        self.role = role

        self.gamma = Gamma
        self.mu = Mu
        self.sigma = Sigma

        # ================== Policy ================
        action_state = State()
        action_state["action"] = StateElement(
            0,
            Space(
                [numpy.full((1, 1), -numpy.inf), numpy.full((1, 1), numpy.inf)],
                "continuous",
            ),
        )

        def shaped_gaussian_noise(action, observation, *args):
            gamma, mu, sigma = args[:3]
            if gamma is None:
                return action
            if sigma is None:
                sigma = numpy.sqrt(self.host.timestep)  # Wiener process
            if mu is None:
                mu = 0
            noise = gamma * numpy.random.normal(mu, sigma)
            return action + noise

        agent_policy = LinearFeedback(
            action_state,
            ("task_state", "x"),
            noise_function=shaped_gaussian_noise,
            noise_func_args=(self.gamma, self.mu, self.sigma),
        )

        # ================== Observation Engine

        observation_engine = RuleObservationEngine(base_task_engine_specification)

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
        )

    def render(self, *args, **kwargs):
        """render

        Displays actions selected by the LQR agent.
        """
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is None:
                self.ax = axuser
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Action")
            if self.action:
                self.ax.plot(
                    self.bundle.round_number * self.bundle.task.timestep,
                    self.action,
                    "bo",
                )
        if "text" in mode:
            print("Action")
            print(self.action)
