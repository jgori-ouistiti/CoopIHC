import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict

import coopihc.observation
from coopihc.space import State, StateElement, Space
from coopihc.policy import BasePolicy, LinearFeedback
from coopihc.observation import (
    RuleObservationEngine,
    base_user_engine_specification,
    base_assistant_engine_specification,
    base_task_engine_specification,
)
from coopihc.inference import (
    BaseInferenceEngine,
    GoalInferenceWithUserPolicyGiven,
    ContinuousKalmanUpdate,
)
from coopihc.helpers import flatten, sort_two_lists

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.linalg


import sys
import copy
import gym.spaces


class BaseAgent(ABC):
    """Instantiate or subclass this class to define an agent that can be used in a Bundle.

    By default, this class will be initialized with an empty internal state, a BasePolicy with a single 'None' action, a RuleObservationEngine with a BaseUser of BaseAssistant profile, and a BaseInference engine.

    You can override some components, e.g. to override the existing policy of an agent, do:
        changed_policy_user = MyNewUser(agent_policy = some_other_policy)
    In that case, MyNewUser class which had been defined with some specific policy, has had its policy overriden by 'some_other_policy'; all other components remain equal.


    The main API methods that users of this class should redefine are:

        + finit
        + reset
        + render


    :param str role: 'user' or 'assistant'
    :param type **kwargs: keyword values ( each agent_X key expects a valid X object, and X_kwargs expects a valid dictionnary of keyword arguments for X)

        + agent_policy
        + agent_inference_engine
        + agent_observation_engine
        + agent_state
        + policy_kwargs
        + inference_engine_kwargs
        + observation_engine_kwargs
        + state_kwargs


    :return: A Bundle-compatible agent
    :rtype: BaseAgent

    """

    def __init__(self, role, **kwargs):
        component_dic, remaining_kwargs = self._allow_override(
            **kwargs
        )  # This line can probably be removed

        # Bundles stuff
        self.bundle = None
        self.ax = None

        # Set role of agent
        if role not in ["user", "assistant"]:
            raise ValueError(
                "First argument 'role' should be either 'user' or 'assistant'"
            )
        else:
            self.role = role

        # Define policy
        self.attach_policy(component_dic["policy"], **kwargs.pop("policy_kwargs", {}))
        # self._attach_policy(policy, **policy_kwargs)

        # Init state
        if component_dic["state"] is None:
            self.state = State(**kwargs.pop("state_kwargs", {}))
        else:
            self.state = component_dic["state"]

        # Define observation engine
        self.attach_observation_engine(
            component_dic["observation_engine"],
            **kwargs.pop("observation_engine_kwargs", {})
        )

        # Define inference engine
        self.attach_inference_engine(
            component_dic["inference_engine"],
            **kwargs.pop("inference_engine_kwargs", {})
        )

    def _allow_override(self, **kwargs):
        ## Seems to me that the commented out part is useless. I forgot why I did it in the first place.

        # oap = kwargs.pop("override_agent_policy", None)
        # if oap is not None:
        #     agent_policy = oap
        # else:
        agent_policy = kwargs.pop("agent_policy", None)

        # oaoe = kwargs.pop("override_agent_observation_engine", None)
        # if oaoe is not None:
        #     agent_observation_engine = oaoe
        # else:
        agent_observation_engine = kwargs.pop("agent_observation_engine", None)

        # oaie = kwargs.pop("override_agent_inference_engine", None)
        # if oaie is not None:
        #     agent_inference_engine = oaie
        # else:
        agent_inference_engine = kwargs.pop("agent_inference_engine", None)

        # oas = kwargs.pop("override_agent_state", None)
        # if oas is not None:
        #     agent_state = oas
        # else:
        agent_state = kwargs.pop("agent_state", None)

        return {
            "state": agent_state,
            "policy": agent_policy,
            "observation_engine": agent_observation_engine,
            "inference_engine": agent_inference_engine,
        }, kwargs

    def _content__(self):
        return {
            "Name": self.__class__.__name__,
            "State": self.state._content__(),
            "Observation Engine": self.observation_engine._content__(),
            "Inference Engine": self.inference_engine._content__(),
            "Policy": self.policy._content__(),
        }

    @property
    def observation(self):
        """Returns the latest observation"""
        return (
            self.inference_engine.buffer[-1]
            if self.inference_engine.buffer is not None
            else None
        )

    @property
    def action(self):
        """Returns the latest selected action"""
        return self.policy.action_state["action"]

    def attach_policy(self, policy, **kwargs):
        if policy is None:
            self.policy = BasePolicy()
        else:
            self.policy = policy
        self.policy.host = self

    def attach_observation_engine(self, observation_engine, **kwargs):
        if observation_engine is None:
            if self.role == "user":
                self.observation_engine = RuleObservationEngine(
                    base_user_engine_specification
                )
            elif self.role == "assistant":
                self.observation_engine = RuleObservationEngine(
                    base_assistant_engine_specification
                )
            else:
                raise NotImplementedError
        else:
            self.observation_engine = observation_engine
        self.observation_engine.host = self

    def attach_inference_engine(self, inference_engine, **kwargs):
        if inference_engine is None:
            self.inference_engine = BaseInferenceEngine()
        else:
            self.inference_engine = inference_engine
        self.inference_engine.host = self

    def _base_reset(self, all=True, dic=None):

        if all:
            self.policy.reset()
            self.inference_engine.reset()
            self.observation_engine.reset()

        if not dic:
            self.state.reset()
            # self.reset(dic=dic)   # Check but this should not be needed
            self.reset()

            return

        # self.reset(dic=dic)   # Check but this should not be needed
        self.reset()
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value["values"]
            if value is not None:
                self.state[key]["values"] = value

    def reset(self):
        """What should happen when initializing the agent.

        Bundles will reset all engines, states and policies and handles forced resets (by the reset dictionnary mechanism), so you don't need to take care of that here.

        """
        pass

    def finit(self):
        """finit is called by bundle during its initialization. This gives the possibility to finish initializing (finit) the agent when information from another component is required e.g. an assistant which requires the list of possible targets from the task."""
        pass

    def _take_action(self):
        return self.policy.sample()

    def _agent_step(self, infer=True):
        """Play one agent's turn: Observe the game state via the observation engine, update the internal state via the inference engine, collect rewards for both processes and pass them to the caller (usually the bundle).

        :return: agent_obs_reward; agent_infer_reward: agent_obs_reward (float) reward associated with observing. agent_infer_reward (float) reward associated with inferring

        :meta private:
        """
        # agent observes the state
        agent_observation, agent_obs_reward = self._observe(self.bundle.game_state)

        # Pass observation to InferenceEngine Buffer
        self.inference_engine.add_observation(agent_observation)
        # Infer the new user state
        if infer:

            agent_state, agent_infer_reward = self.inference_engine.infer()
            # Broadcast new agent_state
            self.state.update(agent_state)

            # Update agent observation
            if self.role == "user":
                if self.inference_engine.buffer[-1].get("user_state") is not None:
                    self.inference_engine.buffer[-1]["user_state"].update(agent_state)
            elif self.role == "assistant":
                if self.inference_engine.buffer[-1].get("assistant_state") is not None:
                    self.inference_engine.buffer[-1]["assistant_state"].update(
                        agent_state
                    )
        else:
            agent_infer_reward = 0
        return agent_obs_reward, agent_infer_reward

    def _observe(self, game_state):
        observation, reward = self.observation_engine.observe(game_state)
        return observation, reward

    def render(self, *args, **kwargs):
        """Renders the agent part of the bundle. Render can be redefined but should keep the same signature. Currently supports text and plot modes.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axuser, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                if self.role == "user":
                    self.ax = axuser
                else:
                    self.ax = axassistant
                self.ax.axis("off")
                self.ax.set_title(type(self).__name__ + " State")
        if "text" in mode:
            print(type(self).__name__ + " State")


# Should not be needed anymore


# class DummyAssistant(BaseAgent):
#     """An Assistant that does nothing, used by Bundles that don't use an assistant.
#
#     :meta private:
#     """
#
#     def __init__(self, **kwargs):
#         """
#         :meta private:
#         """
#         super().__init__("assistant", **kwargs)
#
#     def finit(self):
#         """
#         :meta private:
#         """
#         pass
#
#     def reset(self, dic=None):
#         """
#         :meta private:
#         """
#         pass
#
#
# class DummyUser(BaseAgent):
#     """An User that does nothing, used by Bundles that don't use an user.
#
#     :meta private:
#     """
#
#     def __init__(self, **kwargs):
#         """
#         :meta private:
#         """
#         super().__init__("user", **kwargs)
#
#     def finit(self):
#         """
#         :meta private:
#         """
#         pass
#
#     def reset(self, dic=None):
#         """
#         :meta private:
#         """
#         pass


### Goal could be defined as a target state of the task, in a more general description.
class GoalDrivenDiscreteUser(BaseAgent):
    """A User that is driven by a Goal and uses Discrete actions. It expects to be used with a task that has a substate named targets. Its internal state includes a 'goal' substate, whose value is either one of the task's targets."""

    def finit(self):
        """Appends a Goal substate to the agent's internal state (with dummy values).

        :meta public:
        """
        target_space = self.bundle.task.state["targets"]["spaces"][0]
        self.state["goal"] = StateElement(values=None, spaces=copy.copy(target_space))

        return

    def render(self, *args, **kwargs):
        """Similar to BaseAgent's render, but prints the "Goal" state in addition.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axuser, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """

        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                self.ax = axuser
                self.ax.text(0, 0, "Goal: {}".format(self.state["goal"][0][0]))
                self.ax.set_xlim([-0.5, 0.5])
                self.ax.set_ylim([-0.5, 0.5])
                self.ax.axis("off")
                self.ax.set_title(type(self).__name__ + " Goal")
        if "text" in mode:
            print(type(self).__name__ + " Goal")
            print(self.state["Goal"][0][0])


# class BIGAssistant(BaseAgent):
#     """An Assistant that maintains a discrete belief, updated with Bayes' rule. It supposes that the task has targets, and that the user selects one of these as a goal.
#
#     :param action_space: (list(gym.spaces)) space in which the actions of the user take place, e.g.``[gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]``
#     :param action_set: (list) possible action set for each subspace (None for Box) e.g. ``[None, None]``
#     :param user_model: (coopihc.user_model) user_model used by the assistant to form the likelihood in Bayes rule. It can be the exact same model that is used by the user, or a different one (e.g. if the assistant has to learn the model)
#     :param observation_engine: (coopihc.observation_engine).
#
#     :meta public:
#     """
#
#     def __init__(
#         self, action_space, action_set, user_model, observation_engine=None
#     ):
#
#         agent_state = State()
#         agent_policy = BIGDiscretePolicy()
#         observation_engine = None
#         inference_engine = GoalInferenceWithUserPolicyGiven(user_model)
#
#         super().__init__(
#             "assistant",
#             state=agent_state,
#             policy=agent_policy,
#             observation_engine=observation_engine,
#             inference_engine=inference_engine,
#         )
#
#     def finit(self):
#         """Appends a Belief substate to the agent's internal state.
#
#         :meta public:
#         """
#         targets = self.bundle.task.state["_value_Targets"]
#         self.state["_value_Beliefs"] = [
#             None,
#             [
#                 gym.spaces.Box(
#                     low=numpy.zeros(
#                         len(targets),
#                     ),
#                     high=numpy.ones(
#                         len(targets),
#                     ),
#                 )
#             ],
#             None,
#         ]
#         self.targets = targets
#
#     def reset(self, dic=None):
#         """Resets the belief substate with Uniform prior.
#
#         :meta public:
#         """
#         self.state.reset()
#         super().reset(dic)
#
#     def render(self, mode, *args, **kwargs):
#         """Draws a boxplot of the belief substate in the assistant axes.
#
#         :param args: see other render methods
#         :param mode: see other render methods
#         """
#         ## Begin Helper functions
#         def set_box(
#             ax,
#             pos,
#             draw="k",
#             fill=None,
#             symbol=None,
#             symbol_color=None,
#             shortcut=None,
#             box_width=1,
#             boxheight=1,
#             boxbottom=0,
#         ):
#             if shortcut == "void":
#                 draw = "k"
#                 fill = "#aaaaaa"
#                 symbol = None
#             elif shortcut == "target":
#                 draw = "#96006c"
#                 fill = "#913979"
#                 symbol = "1"
#                 symbol_color = "k"
#             elif shortcut == "goal":
#                 draw = "#009c08"
#                 fill = "#349439"
#                 symbol = "X"
#                 symbol_color = "k"
#             elif shortcut == "position":
#                 draw = "#00189c"
#                 fill = "#6573bf"
#                 symbol = "X"
#                 symbol_color = "k"
#
#             BOX_HW = box_width / 2
#             _x = [pos - BOX_HW, pos + BOX_HW, pos + BOX_HW, pos - BOX_HW]
#             _y = [
#                 boxbottom,
#                 boxbottom,
#                 boxbottom + boxheight,
#                 boxbottom + boxheight,
#             ]
#             x_cycle = _x + [_x[0]]
#             y_cycle = _y + [_y[0]]
#             if fill is not None:
#                 fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color=fill)
#
#             (draw,) = ax.plot(x_cycle, y_cycle, "-", color=draw, lw=2)
#             symbol = None
#             if symbol is not None:
#                 symbol = ax.plot(
#                     pos, 0, color=symbol_color, marker=symbol, markersize=100
#                 )
#
#             return draw, fill, symbol
#
#         def draw_beliefs(ax):
#             targets = self.targets
#             beliefs = self.state["Beliefs"]
#             targets, beliefs = sort_two_lists(targets, beliefs)
#             ticks = []
#             ticklabels = []
#             for i, (t, b) in enumerate(zip(targets, beliefs)):
#                 draw, fill, symbol = set_box(
#                     ax, 2 * i, shortcut="target", boxheight=b
#                 )
#                 ticks.append(2 * i)
#                 try:
#                     _label = [int(_t) for _t in t]
#                 except TypeError:
#                     _label = int(t)
#                 ticklabels.append(_label)
#             self.ax.set_xticks(ticks)
#             self.ax.set_xticklabels(ticklabels, rotation=90)
#
#         ## End Helper functions
#
#         if "plot" in mode:
#             axtask, axuser, axassistant = args[:3]
#             ax = axassistant
#             if self.ax is not None:
#                 title = self.ax.get_title()
#                 self.ax.clear()
#                 draw_beliefs(ax)
#                 ax.set_title(title)
#
#             else:
#                 self.ax = ax
#                 draw_beliefs(ax)
#                 self.ax.set_title(type(self).__name__ + " Beliefs")
#
#         if "text" in mode:
#             targets = self.targets
#             beliefs = self.state["Beliefs"]
#             targets, beliefs = sort_two_lists(targets, beliefs)
#             print("Targets", targets)
#             print("Beliefs", beliefs)

# ===================== Classic Control ==================== #

# An LQR Controller (implements a linear feedback policy)


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


# Finite Horizon Discrete Time Controller
# Outdated
class FHDT_LQRController(LQRController):
    def __init__(self, N, role, Q, R, Gamma):
        self.N = N
        self.i = 0
        super().__init__(role, Q, R, gamma=Gamma)
        self.timespace = "discrete"

    def reset(self, dic=None):
        self.i = 0
        super().reset(dic)

    def finit(self):
        self.K = []
        task = self.bundle.task
        A, B = task.A, task.B
        # Compute P(k) matrix for k in (N:-1:1)
        self.P = [self.Q]
        for k in range(self.N - 1, 0, -1):
            Pcurrent = self.P[0]
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            Pnext = (
                self.Q
                + A.T @ Pcurrent @ A
                - A.T @ Pcurrent @ B @ invPart @ B.T @ Pcurrent @ A
            )
            self.P.insert(0, Pnext)

        # Compute Kalman Gain
        for Pcurrent in self.P:
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            K = -invPart @ B.T @ Pcurrent @ A
            self.K.append(K)


# Infinite Horizon Discrete Time Controller
# Uses Discrete Algebraic Ricatti Equation to get P


class IHDT_LQRController(LQRController):
    def __init__(self, role, Q, R, Gamma):
        super().__init__(role, Q, R, gamma=Gamma)
        self.timespace = "discrete"

    def finit(self):
        task = self.bundle.task
        A, B = task.A, task.B
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        invPart = scipy.linalg.inv((self.R + B.T @ P @ B))
        K = invPart @ B.T @ P @ A
        self.policy.set_feedback_gain(K)


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


# ================ Examples ================
from coopihc.policy import ExamplePolicy


class ExampleUser(BaseAgent):
    """An agent that handles the ExamplePolicy."""

    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = StateElement(
            values=numpy.array([4]),
            spaces=[
                Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])
            ],
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            values=None,
            spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
        )
        agent_policy = ExamplePolicy(action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    # Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4
    def reset(self, dic=None):
        self.state["goal"]["values"] = 4

    # [start-baseagent-init]

    # Define a state
    state = State()
    state["goalstate"] = StateElement(
        values=numpy.array([4]),
        spaces=[
            Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])
        ],
    )

    # Define a policy (random policy)
    action_state = State()
    action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )
    agent_policy = BasePolicy(action_state)

    # Explicitly use default observation and inference engines (default behavior is triggered when keyword argument is not provided or keyword value is None)
    observation_engine = RuleObservationEngine(
        deterministic_specification=base_user_engine_specification
    )
    inference_engine = BaseInferenceEngine(buffer_depth=0)

    BaseAgent(
        "user",
        agent_policy=agent_policy,
        agent_observation_engine=observation_engine,
        agent_inference_engine=inference_engine,
        agent_state=state,
    )
    # [end-baseagent-init]