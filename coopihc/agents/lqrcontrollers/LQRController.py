import numpy
import gym.spaces


from coopihc.agents.BaseAgent import BaseAgent
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
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

    def __init__(self, role, Q, R, *args, Gamma=None, Mu=None, sigma=None, **kwargs):

        self.R = R
        self.Q = Q
        self.role = role

        self.gamma = Gamma
        self.mu = Mu
        self.sigma = Sigma

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
        # Below commented out without actually checking, but should be accounted for by the new base_reset mechanism.
        # if dic is None:
        #     super().reset()

        # # Nothing to reset

        # if dic is not None:
        #     super().reset(dic=dic)
        pass

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
            if self.action["values"][0]:
                self.ax.plot(
                    self.bundle.task.turn * self.bundle.task.timestep,
                    self.action["values"][0],
                    "bo",
                )
        if "text" in mode:
            print("Action")
            print(self.action)
