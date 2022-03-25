import numpy


from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.policy.LinearFeedback import LinearFeedback
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_task_engine_specification


class LQRController(BaseAgent):
    """A Linear Quadratic Regulator.

    This agent will read a state named 'x' from the task, and produce actions according to:

    .. math::

        \\text{action} =  -K X

    where K is the so-called feedback gain, which has to be specified externally. For an example, see the :py:class:`coopihc.agents.lqrcontrollers.FHDT_LQRController.FHDT_LQRController` source code.

    The controller will also output observation rewards J, for state X and action u


    .. math::

        J = -X^t Q X - u^t R u

    .. note::

        This class is meant to be subclassed

    .. warning::

        Tested only on 1d output.

    :param role: "user" or "assistant"
    :type role: string
    :param Q: State cost
    :type Q: numpy.ndarray
    :param R: Control cost
    :type R: numpy.ndarray

    """

    def __init__(self, role, Q, R, *args, **kwargs):

        self.R = R
        self.Q = Q
        self.role = role

        # ================== Policy ================
        action_state = State()
        action_state["action"] = array_element(
            low=numpy.full((1,), -numpy.inf), high=numpy.full((1,), numpy.inf)
        )

        agent_policy = LinearFeedback(
            action_state,
            ("task_state", "x"),
        )

        # ================== Observation Engine

        class RuleObsWithRewards(RuleObservationEngine):
            def __init__(
                self,
                Q,
                R,
                *args,
                deterministic_specification=base_task_engine_specification,
                extradeterministicrules={},
                extraprobabilisticrules={},
                mapping=None,
                **kwargs
            ):
                self.R = R
                self.Q = Q
                super().__init__(
                    *args,
                    deterministic_specification=base_task_engine_specification,
                    extradeterministicrules={},
                    extraprobabilisticrules={},
                    mapping=None,
                    **kwargs
                )

            def observe(self, game_state=None):
                obs, _ = super().observe(game_state=game_state)
                x = obs["task_state"]["x"].view(numpy.ndarray)
                u = obs["user_action"]["action"].view(numpy.ndarray)
                reward = -x.T @ self.R @ x - u.T @ self.Q @ u
                return obs, reward

        observation_engine = RuleObsWithRewards(
            self.R, self.Q, deterministic_specification=base_task_engine_specification
        )

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
