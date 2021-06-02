import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict

import core.observation
from core.space import State
from core.policy import Policy, LinearFeedback
from core.observation import RuleObservationEngine, base_operator_engine_specification, base_assistant_engine_specification, base_task_engine_specification
from core.inference import InferenceEngine, GoalInferenceWithOperatorPolicyGiven, ContinuousGaussian, ContinuousKalmanUpdate
from core.helpers import flatten, sort_two_lists

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.linalg

class BaseAgent(ABC):
    """Subclass this to define an agent that can be used in a Bundle.
    A Baseagent class, with an internal state, methods to access this state, a policy (random sampling by default), and observation and inference engine.

    By default, a BaseAgent's internal state contains the other agent's last action (either 'AssistantAction' or 'OperatorAction'), but nothing else.

    The main API methods that users of this class need to know are:

        finit

        sample

        append_state

        render

    :param role: (int or string) role of the agent. 0 or 'operator' for Operator, 1 or 'assistant' for Assistant
    :param action_space: (list(gym.spaces)) a list of gym.spaces that specifies in which space actions take place e.g. ``[gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))]``
    :param action_set: (list(list)) corresponding list of values for actions, encapsulated in a list. If the action subspace is a Box, then pass None. If the action subspace is Discrete, pass the action list. e.g. ``[[-1,1], None]``
    :param observation_engine: (object) an observation_engine, see core.observation_engine. If None, a RuleObservationEngine using BaseOperatorObservationRule/BaseAssistantObservationRule will be used. e.g. ``observation_engine = RuleObservationEngine(BaseOperatorObservationRule)``
    :param inference_engine: (object) an inference_engine, see core.inference_engine. If None, the basic InferenceEngine is used, which does nothing with the agent's internal state e.g. ``inference_engine = InferenceEngine()``
    """

    # def __init__(self, role, action_space, action_set = None, observation_engine = None, inference_engine = None):

    def __init__(self, role, policy = None, state = None, observation_engine = None, inference_engine = None):

        # Bundles stuff
        self.encapsulation = False
        self.bundle = None
        self.ax = None


        # Set role of agent
        if role not in ["operator", "assistant"]:
            raise ValueError("First argument role should be either 'operator' or 'assistant'")
        else:
            self.role = role

        # Define policy
        self.attach_policy(policy)


        # Init state
        if state is None:
            self.state = State()
        else:
            self.state = state


        # Define observation engine
        self.attach_observation_engine(observation_engine)


        # Define inference engine
        self.attach_inference_engine(inference_engine)


    def attach_policy(self, policy):
        self.policy = policy
        try:
            self.policy.host = self
        except AttributeError as err:
            pass

    def attach_observation_engine(self, observation_engine):
        if observation_engine is None:
            if self.role == "operator":
                self.observation_engine = RuleObservationEngine(base_operator_engine_specification)
            elif self.role == "assistant":
                self.observation_engine = RuleObservationEngine(base_assistant_engine_specification)
            else:
                raise NotImplementedError
        else:
            self.observation_engine = observation_engine
        self.observation_engine.host = self

    def attach_inference_engine(self, inference_engine):
        if inference_engine is None:
            self.inference_engine = InferenceEngine()
        else:
            self.inference_engine = inference_engine
        self.inference_engine.host = self


    def reset(self, dic = {}):
        """ Resets the agent to an initial state. Does not reset the other agents or task. Usually, it is enough to reset the interbal state of the agent. This method has to be redefined when subclassing BaseAgent.

        :param args: (collections.OrderedDict) internal state to which the agent should be reset e.g. ``OrderedDict([('AssistantAction', [1]), ('Goal', [2])])``

        :meta public:
        """
        if self.policy is None:
            return RuntimeError('A policy needs to be attached to this agent.')
        if not dic:
            self.state.reset()
            return

        if self.role == "operator":
            dictionnary = dic.get("operator_state")
        else:
            dictionnary = dic.get("assistant_state")
        if dictionnary:
            for key in list(self.state.keys()):
                value = dictionnary.get(key)
                if value is not None:
                    self.state[key]['value'] = value

    def finit(self):
        """ finit is called by bundle after the two agents and task have been linked together. This gives the possibility to finish initializing (finit) when information from another component is required e.g. an assistant which requires the list of possible targets from the task. This method has to be redefined when subclassing BaseAgent.

        :meta public:
        """
        # Finish initializing agent
        pass

    def take_action(self):
        return self.policy.sample()

    def agent_step(self, infer = True):
        """ Play one agent's turn: Observe the game state via the observation engine, update the internal state via the inference engine, collect rewards for both processes and pass them to the caller (usually the bundle).

        :return: agent_obs_reward; agent_infer_reward: agent_obs_reward (float) reward associated with observing. agent_infer_reward (float) reward associated with inferring

        :meta private:
        """
        # agent observes the state
        agent_observation, agent_obs_reward = self.observe(self.bundle.game_state)
        # Pass observation to InferenceEngine Buffer
        self.inference_engine.add_observation(agent_observation)
        # Infer the new operator state
        if infer:
            agent_state, agent_infer_reward = self.inference_engine.infer()
            # Broadcast new agent_state
            self.state.update(agent_state)
            # Update agent observation if
            if self.role == "operator":
                if self.inference_engine.buffer[-1].get('operator_state') is not None:
                    self.inference_engine.buffer[-1]['operator_state'].update(agent_state)
            elif self.role == "assistant":
                if self.inference_engine.buffer[-1].get('assistant_state') is not None:
                    self.inference_engine.buffer[-1]['assistant_state'].update(agent_state)
        else:
            agent_infer_reward = 0
        return agent_obs_reward, agent_infer_reward


    def observe(self, game_state):
        """ This method is called by agent_step to produce an observation from a given game_state.

        :param game_state: (collections.OrderedDict) state of the game (usually obtained from ``bundle.game_state``)

        :returns: observation, reward. Observation: (collections.OrderedDict) observation of the state of the game; Reward: (float) reward associated with the observation

        :meta public:
        """
        observation, reward = self.observation_engine.observe( game_state)
        return observation, reward



    def render(self, *args, **kwargs):
        """ Renders the agent part of the bundle. Render can be redefined but should keep the same signature. Currently supports text and plot modes.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axoperator, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'


        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                if self.role == "operator":
                    self.ax = axoperator
                else:
                    self.ax = axassistant
                self.ax.axis('off')
                self.ax.set_title(type(self).__name__ + " State")
        if 'text' in mode:
            print(type(self).__name__ + " State")





class DummyAssistant(BaseAgent):
    """ An Assistant that does nothing, used by Bundles that don't use an assistant.

    :meta private:
    """
    def __init__(self, **kwargs):
        """
        :meta private:
        """
        super().__init__("assistant", kwargs)

    def finit(self):
        """
        :meta private:
        """
        pass

    def reset(self, *args):
        """
        :meta private:
        """
        pass


class DummyOperator(BaseAgent):
    """ An Operator that does nothing, used by Bundles that don't use an operator.

    :meta private:
    """
    def __init__(self, **kwargs):
        """
        :meta private:
        """
        super().__init__("operator", kwargs)

    def finit(self):
        """
        :meta private:
        """
        pass

    def reset(self, *args):
        """
        :meta private:
        """
        pass


### Goal could be defined as a target state of the task, in a more general description.
class GoalDrivenDiscreteOperator(BaseAgent):
    """ An Operator that is driven by a Goal and uses Discrete actions. It has to be used with a task that has a substate named Targets. Its internal state includes a goal substate, whose value is either one of the task's Targets.


    :param operator_model: (core.models), an operator model contained in core.models or that subclasses a class in core.models
    :param observation_engine: (core.ObservationEngine), see BaseAgent for definition

    :meta public:
    """

    def finit(self):
        """ Appends a Goal substate to the agent's internal state (with dummy values).

        :meta public:
        """
        target_space = self.bundle.task.state['Targets'][1]
        self.state["Goal"] = [None, [gym.spaces.Discrete(len(target_space))], None]

        return
    #
    # def reset(self, dic = None):
    #     """ Picks a random target as goal.
    #
    #     :meta public:
    #     """
    #     targets = self.bundle.game_state['task_state']['Targets'][0]
    #     goal = numpy.random.choice([i for i in range(len(targets))])
    #     self.state['Goal'] = [goal, None, None]
    #     super().reset(dic)

    # def sample(self):
    #     """ Use policy provided by the operator model to select actions.
    #
    #     :meta public:
    #     """
    #
    #
    #     actions = self.operator_model.sample(self.inference_engine.buffer[-1])
    #
    #     if isinstance(actions, (int, float)):
    #         actions = [actions]
    #     return actions

    def render(self, *args, **kwargs):
        """ Similar to BaseAgent's render, but prints the "Goal" state in addition.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axoperator, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """

        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'

        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                self.ax = axoperator
                self.ax.text(0,0, "Goal: {}".format(self.state['Goal'][0][0]))
                self.ax.set_xlim([-0.5,0.5])
                self.ax.set_ylim([-0.5,0.5])
                self.ax.axis('off')
                self.ax.set_title(type(self).__name__ + " Goal")
        if 'text' in mode:
            print(type(self).__name__ + " Goal")
            print(self.state['Goal'][0][0])





# An agent that has handles Gaussian Continuous Beliefs
class GaussianContinuousBeliefOperator(BaseAgent):
    """ An Operator that maintains a continuous Gaussian belief. It can be used in cases where the goal of the operator is not directly observable to it.

    :param action_space: (list(gym.spaces)) space in which the actions of the operator take place, e.g.``[gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]``
    :param action_set: (list) possible action set for each subspace (None for Box) e.g. ``[None, None]``
    :param observation_engine: (core.observation_engine).
    :param dim: dimension of the multivariate Gaussian used to model the belief.

    :meta public:
    """
    def __init__(self, action_space, action_set, observation_engine, dim = 1):
        self.dim = dim
        inference_engine = ContinuousGaussian()
        super().__init__("operator", action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)
        self.append_state("MuBelief", [gym.spaces.Box(low=-1, high=1, shape=(dim, ), dtype=numpy.float64)])
        self.append_state("SigmaBelief", [gym.spaces.Box(low=-1, high=1, shape=(dim**2, ), dtype=numpy.float64)])


    def reset(self, dic = None):
        super().reset(dic)

    def render(self, mode, *args, **kwargs):
        """ Similar to BaseAgent's render. In text mode, prints the parameters of the belief model. In plot, prints a 95% confidence ellipsis for the belief on the task axis.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axoperator, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """

        if 'text' in mode:
            print(self.state['MuBelief'].tolist())
            print(self.state['SigmaBelief'].tolist())
        elif 'plot' in mode:
            if self.dim == 2:
                # try:
                #     self.patch.remove()
                # except AttributeError:
                #     pass
                axtask, axoperator, axassistant = args[:3]
                covariance = self.state['SigmaBelief']
                mu = self.state['MuBelief']
                self.patch = self.confidence_ellipse(mu, covariance, axtask)
            else:
                raise NotImplementedError


    def confidence_ellipse(self, mu, covariance, ax, n_std=2.0, facecolor='#d1dcf0', edgecolor = 'b', **kwargs):
        """
        :meta private:
        """
        ## See https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html for source. Computing eigenvalues directly should lead to code that is more readily understandable
        rho = numpy.sqrt(covariance[0,1]**2/covariance[0,0]/covariance[1,1])
        ellipse_radius_x = numpy.sqrt(1 + rho)
        ellipse_radius_y = numpy.sqrt(1 - rho)

        ellipse = Ellipse((0, 0), width=ellipse_radius_x * 2, height=ellipse_radius_y * 2, facecolor=facecolor, edgecolor = edgecolor, **kwargs)
        n_std = 2
        scale_x = numpy.sqrt(covariance[0, 0]) * n_std
        mean_x = mu[0]

        scale_y = numpy.sqrt(covariance[1, 1]) * n_std
        mean_y = mu[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)



# An agent that has a substate called Beliefs, which are updated in a Bayesian fashion. Requires a model of the operator as well as the potential target states that can serve as goals. Subclass this to implement various policies w/r beliefs.

class BIGAssistant(BaseAgent):
    """ An Assistant that maintains a discrete belief, updated with Bayes' rule. It supposes that the task has targets, and that the operator selects one of these as a goal.

    :param action_space: (list(gym.spaces)) space in which the actions of the operator take place, e.g.``[gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]``
    :param action_set: (list) possible action set for each subspace (None for Box) e.g. ``[None, None]``
    :param operator_model: (core.operator_model) operator_model used by the assistant to form the likelihood in Bayes rule. It can be the exact same model that is used by the operator, or a different one (e.g. if the assistant has to learn the model)
    :param observation_engine: (core.observation_engine).

    :meta public:
    """
    def __init__(self, action_space, action_set, operator_model, observation_engine = None):

        agent_state = State()
        agent_policy = BIGDiscretePolicy()
        observation_engine = None
        inference_engine = GoalInferenceWithOperatorPolicyGiven(operator_model)

        super().__init__(   "assistant",
                            state = agent_state,
                            policy = agent_policy,
                            observation_engine = observation_engine, inference_engine = inference_engine)






    def finit(self):
        """ Appends a Belief substate to the agent's internal state.

        :meta public:
        """
        targets = self.bundle.task.state['_value_Targets']
        self.state['_value_Beliefs'] = [None, [gym.spaces.Box( low = numpy.zeros( len( targets ),), high = numpy.ones( len( targets), ) )], None]
        self.targets = targets



    def reset(self, dic = None):
        """ Resets the belief substate with Uniform prior.

        :meta public:
        """
        self.state.reset()
        # targets = self.bundle.task.state['Targets']
        # self.modify_state("Beliefs", value = [1/len(targets) for t in targets])
        # self.inference_engine.reset()
        # self.targets = targets
        super().reset(dic)

    def render(self, mode, *args, **kwargs):
        """ Draws a boxplot of the belief substate in the assistant axes.

        :param args: see other render methods
        :param mode: see other render methods
        """
    ## Begin Helper functions
        def set_box(ax, pos, draw = "k", fill = None, symbol = None, symbol_color = None, shortcut = None, box_width = 1, boxheight = 1, boxbottom = 0):
            if shortcut == 'void':
                draw = 'k'
                fill = '#aaaaaa'
                symbol = None
            elif shortcut == 'target':
                draw = '#96006c'
                fill = "#913979"
                symbol = "1"
                symbol_color = 'k'
            elif shortcut == 'goal':
                draw = '#009c08'
                fill = '#349439'
                symbol = "X"
                symbol_color = 'k'
            elif shortcut == 'position':
                draw = '#00189c'
                fill = "#6573bf"
                symbol = "X"
                symbol_color = 'k'

            BOX_HW = box_width/2
            _x = [pos-BOX_HW, pos+BOX_HW, pos + BOX_HW, pos - BOX_HW]
            _y = [boxbottom, boxbottom, boxbottom + boxheight, boxbottom + boxheight]
            x_cycle = _x + [_x[0]]
            y_cycle = _y + [_y[0]]
            if fill is not None:
                fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color = fill)

            draw, = ax.plot(x_cycle,y_cycle, '-', color = draw, lw = 2)
            symbol = None
            if symbol is not None:
                symbol = ax.plot(pos, 0, color = symbol_color, marker = symbol, markersize = 100)

            return draw, fill, symbol

        def draw_beliefs(ax):
            targets = self.targets
            beliefs = self.state['Beliefs']
            targets, beliefs = sort_two_lists(targets, beliefs)
            ticks = []
            ticklabels = []
            for i, (t,b) in enumerate(zip(targets, beliefs)):
                draw, fill, symbol = set_box(ax, 2*i, shortcut = 'target', boxheight = b)
                ticks.append(2*i)
                try:
                    _label = [int(_t) for _t in t]
                except TypeError:
                    _label = int(t)
                ticklabels.append(_label)
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(ticklabels, rotation = 90)

    ## End Helper functions


        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            ax = axassistant
            if self.ax is not None:
                title = self.ax.get_title()
                self.ax.clear()
                draw_beliefs(ax)
                ax.set_title(title)

            else:
                self.ax = ax
                draw_beliefs(ax)
                self.ax.set_title(type(self).__name__ + " Beliefs")

        if 'text' in mode:
            targets = self.targets
            beliefs = self.state['Beliefs']
            targets, beliefs = sort_two_lists(targets, beliefs)
            print('Targets', targets)
            print("Beliefs", beliefs)



#
# class DiscreteBayesianBeliefAssistant(BaseAgent):
#     """ An Assistant that maintains a discrete belief, updated with Bayes' rule. It supposes that the task has targets, and that the operator selects one of these as a goal.
#
#     :param action_space: (list(gym.spaces)) space in which the actions of the operator take place, e.g.``[gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]``
#     :param action_set: (list) possible action set for each subspace (None for Box) e.g. ``[None, None]``
#     :param operator_model: (core.operator_model) operator_model used by the assistant to form the likelihood in Bayes rule. It can be the exact same model that is used by the operator, or a different one (e.g. if the assistant has to learn the model)
#     :param observation_engine: (core.observation_engine).
#
#     :meta public:
#     """
#     def __init__(self, action_space, action_set, operator_model, observation_engine = None):
#         inference_engine = GoalInferenceWithOperatorModelGiven(operator_model)
#         super().__init__("assistant", action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)
#
#     def finit(self):
#         """ Appends a Belief substate to the agent's internal state.
#
#         :meta public:
#         """
#         targets = self.bundle.task.state['Targets']
#         self.append_state( "Beliefs", [gym.spaces.Box( low = numpy.zeros( len( targets ),), high = numpy.ones( len( targets), ) )] )
#         self.targets = targets
#
#     def reset(self, dic = None):
#         """ Resets the belief substate with Uniform prior.
#
#         :meta public:
#         """
#         targets = self.bundle.task.state['Targets']
#         self.modify_state("Beliefs", value = [1/len(targets) for t in targets])
#         self.inference_engine.reset()
#         self.targets = targets
#         super().reset(dic)
#
#     def render(self, mode, *args, **kwargs):
#         """ Draws a boxplot of the belief substate in the assistant axes.
#
#         :param args: see other render methods
#         :param mode: see other render methods
#         """
#     ## Begin Helper functions
#         def set_box(ax, pos, draw = "k", fill = None, symbol = None, symbol_color = None, shortcut = None, box_width = 1, boxheight = 1, boxbottom = 0):
#             if shortcut == 'void':
#                 draw = 'k'
#                 fill = '#aaaaaa'
#                 symbol = None
#             elif shortcut == 'target':
#                 draw = '#96006c'
#                 fill = "#913979"
#                 symbol = "1"
#                 symbol_color = 'k'
#             elif shortcut == 'goal':
#                 draw = '#009c08'
#                 fill = '#349439'
#                 symbol = "X"
#                 symbol_color = 'k'
#             elif shortcut == 'position':
#                 draw = '#00189c'
#                 fill = "#6573bf"
#                 symbol = "X"
#                 symbol_color = 'k'
#
#             BOX_HW = box_width/2
#             _x = [pos-BOX_HW, pos+BOX_HW, pos + BOX_HW, pos - BOX_HW]
#             _y = [boxbottom, boxbottom, boxbottom + boxheight, boxbottom + boxheight]
#             x_cycle = _x + [_x[0]]
#             y_cycle = _y + [_y[0]]
#             if fill is not None:
#                 fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color = fill)
#
#             draw, = ax.plot(x_cycle,y_cycle, '-', color = draw, lw = 2)
#             symbol = None
#             if symbol is not None:
#                 symbol = ax.plot(pos, 0, color = symbol_color, marker = symbol, markersize = 100)
#
#             return draw, fill, symbol
#
#         def draw_beliefs(ax):
#             targets = self.targets
#             beliefs = self.state['Beliefs']
#             targets, beliefs = sort_two_lists(targets, beliefs)
#             ticks = []
#             ticklabels = []
#             for i, (t,b) in enumerate(zip(targets, beliefs)):
#                 draw, fill, symbol = set_box(ax, 2*i, shortcut = 'target', boxheight = b)
#                 ticks.append(2*i)
#                 try:
#                     _label = [int(_t) for _t in t]
#                 except TypeError:
#                     _label = int(t)
#                 ticklabels.append(_label)
#             self.ax.set_xticks(ticks)
#             self.ax.set_xticklabels(ticklabels, rotation = 90)
#
#     ## End Helper functions
#
#
#         if 'plot' in mode:
#             axtask, axoperator, axassistant = args[:3]
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
#         if 'text' in mode:
#             targets = self.targets
#             beliefs = self.state['Beliefs']
#             targets, beliefs = sort_two_lists(targets, beliefs)
#             print('Targets', targets)
#             print("Beliefs", beliefs)
#


# ===================== Classic Control ==================== #

# An LQR Controller (implements a linear feedback policy)

class LQRController(BaseAgent):
    def __init__(self, role, Q, R):
        agent_state = State()
        action_space = [gym.spaces.Box(-numpy.inf, numpy.inf, shape = (1,))]
        action_set = [None]
        agent_policy = LinearFeedback(('task_state','x'), action_space, action_set = action_set)
        observation_engine = RuleObservationEngine(base_task_engine_specification)

        super().__init__(   'operator',
                            state = agent_state,
                            policy = agent_policy,
                            observation_engine = observation_engine
                            )
        self.R = R
        self.Q = Q
        self.action = None

    def reset(self, dic = None):
        super().reset(dic)

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'

        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.ax is None:
                self.ax = axoperator
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Action")
            if self.action:
                self.ax.plot(self.bundle.task.turn*self.bundle.task.timestep, self.action, 'bo')
        if 'text' in mode:
            print('Action')
            print(self.action)


# Finite Horizon Discrete Time Controller
# Outdated
class FHDT_LQRController(LQRController):
    def __init__(self, N, role, Q, R):
        self.N = N
        self.i = 0
        super().__init__(role, Q, R)
        self.timespace = 'discrete'


    def reset(self, dic = None):
        self.i = 0
        super().reset(dic)

    def finit(self):
        self.K = []
        task = self.bundle.task
        A, B = task.A, task.B
        # Compute P(k) matrix for k in (N:-1:1)
        self.P = [self.Q]
        for k in range(self.N-1,0,-1):
            Pcurrent = self.P[0]
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            Pnext = self.Q + A.T @ Pcurrent @ A - A.T @ Pcurrent @ B @ invPart @ B.T @ Pcurrent @ A
            self.P.insert(0, Pnext)

        # Compute Kalman Gain
        for Pcurrent in self.P:
            invPart = scipy.linalg.inv((self.R + B.T @ Pcurrent @ B))
            K = - invPart @ B.T @ Pcurrent @ A
            self.K.append(K)

    def sample(self):
        x = self.inference_engine.buffer[-1]['task_state']['x']
        self.action = self.K[self.i] @ x
        self.i += 1
        return [self.action]

# Infinite Horizon Discrete Time Controller
# Uses Discrete Algebraic Ricatti Equation to get P

class IHDT_LQRController(LQRController):
    def __init__(self, role, Q, R):
        super().__init__(role, Q, R)
        self.timespace = 'discrete'

    def finit(self):
        task = self.bundle.task
        A, B = task.A, task.B
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        invPart = scipy.linalg.inv((self.R + B.T @ P @ B))
        K = invPart @ B.T @ P @ A
        self.policy.feedback_gain = K



# Infinite Horizon Continuous Time LQG controller, based on Phillis 1985
class IHCT_LQGController(BaseAgent):
    """ An Infinite Horizon (Steady-state) LQG controller, based on Phillis 1985, using notations from Qian 2013.

    """
    def __init__(self, role, timestep, Q, R, U, C, Gamma, D, noise = 'on'):
        self.C = C
        self.Gamma = numpy.array(Gamma)
        self.Q = Q
        self.R = R
        self.U = U
        self.D = D
        self.timestep = timestep
        self.role = role
        self.timespace = 'continuous'

        ### Initialize Random Kalmain gains
        self.K = numpy.random.rand(*C.T.shape)
        self.L = numpy.random.rand(1, Q.shape[1])
        self.noise = noise

        # ====== Initialize agent ====== #
        agent_state = State()
        action_space = [gym.spaces.Box(-numpy.inf, numpy.inf, shape = (1,))]
        action_set = [None]
        agent_policy = LinearFeedback(("operator_state","xhat"), action_space, action_set = action_set, noise = noise, Gamma = self.Gamma)

        # ========== Observation Engine=========================
        # Specify the selection of substates
        operator_engine_specification  =    [ ('bundle_state', 'all'),
                                    ('task_state', None),
                                    ('operator_state', 'all'),
                                    ('assistant_state', None),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

        # Specify additional rules, including noise

        # extradeterministicrules
        obs_matrix = {('task_state', 'x'): (core.observation.f_obs_matrix, (C,))}
        extradeterministicrules = {}
        extradeterministicrules.update(obs_matrix)

        # extraprobabilisticrule
        agn_rule = {('task_state', 'x'): (core.observation.agn, (D, numpy.zeros((C.shape[0],1)).reshape(-1,), numpy.sqrt(timestep)*numpy.eye(C.shape[0])))}

        extraprobabilisticrules = {}
        extraprobabilisticrules.update(agn_rule)

        # Instantiate the observation engine
        observation_engine = RuleObservationEngine(deterministic_specification = operator_engine_specification, extradeterministicrules = extradeterministicrules, extraprobabilisticrules = extraprobabilisticrules)

        # =========== Inference Engine ============================
        inference_engine = ContinuousKalmanUpdate()

        super().__init__(role,
                        state = agent_state,
                        policy = agent_policy,
                        observation_engine = observation_engine,
                        inference_engine = inference_engine
                            )

    def finit(self):
        task = self.bundle.task
        self.A_c, self.B_c,  self.G = task.A_c, task.B_c, task.G
        # Attach the model dynamics to the inference engine.
        # self.state['_value_xhat'] = [None]
        self.state['xhat'] =  [ [None], [core.space.Array(task.state['_value_x'])] , [None] ]
        self.inference_engine.set_forward_model_dynamics(self.A_c, self.B_c, self.C)
        # Set K and L up
        mc = self._MContainer(self.A_c, self.B_c, self.C, self.D, self.G, self.Gamma, self.Q, self.R, self.U)
        self.K, self.L = self._compute_Kalman_matrices(mc.pass_args())
        self.inference_engine.set_K(self.K)
        self.policy.feedback_gain = self.L

    def reset(self, dic = None):
        self.state['xhat'] = self.bundle.task.state['x']
        super().reset(dic)

    # def sample(self):
    #     xhat = self.inference_engine.buffer[-1]['operator_state']['xhat']
    #     if self.noise == 'off':
    #         gamma = numpy.random.normal(0, 0)
    #     else:
    #         gamma = numpy.random.normal(0, numpy.sqrt(self.timestep))
    #
    #     self.action = -self.L @ xhat + self.Gamma * gamma
    #
    #     return [self.action]




    class _MContainer:
        """ The purpose of this container is to facilitate common manipulations of the matrices of the LQG problem, as well as potentially storing their evolution. (not implemented yet)
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
            return (self.A, self.B, self.C, self.D, self.G, self.Gamma, self.Q, self.R, self.U)

    def _compute_Kalman_matrices(self, matrices, N=20):
        A, B, C, D, G, Gamma, Q, R, U = matrices
        Y = B @ numpy.array(Gamma).reshape(1,-1)
        Lnorm = []
        Knorm = []
        K = numpy.random.rand(*C.T.shape)
        L = numpy.random.rand(1, A.shape[1])
        for i in range(N):
            Lnorm.append(numpy.linalg.norm(L))
            Knorm.append(numpy.linalg.norm(K))

            n,m = A.shape
            Abar = numpy.block([
                [A - B@L, B@L],
                [numpy.zeros((n,m)), A - K@C]
            ])

            Ybar = numpy.block([
                [-Y@L, Y@L],
                [-Y@L, Y@L]
            ])



            Gbar = numpy.block([
                [G, numpy.zeros((G.shape[0], D.shape[1]))],
                [G, -K@D]
            ])

            V = numpy.block([
                [Q + L.T@R@L, -L.T@R@L],
                [-L.T@R@L, L.T@R@L + U]
            ])

            P, p_res = self._LinRicatti(Abar, Ybar, Gbar@Gbar.T)
            S, s_res = self._LinRicatti(Abar.T, Ybar.T, V)

            P22 = P[n:,n:]
            S11 = S[:n,:n]
            S22 = S[n:,n:]

            K = P22@C.T@numpy.linalg.pinv(D@D.T)
            L = numpy.linalg.pinv(R + Y.T@(S11 + S22)@Y)@B.T@S11

        K, L = self._check_KL(Knorm, Lnorm, K, L, matrices)
        return K,L

    def _LinRicatti(self, A, B, C):
        """ Returns norm of an equation of the form AX + XA.T + BXB.T + C = 0
        """
        #
        n,m = A.shape
        nc,mc = C.shape
        if n !=m:
            print('Matrix A has to be square')
            return -1
        M = numpy.kron( numpy.identity(n), A ) + numpy.kron( A, numpy.identity(n) ) + numpy.kron(B,B)
        C = C.reshape(-1, 1)
        X = -numpy.linalg.pinv(M)@C
        X = X.reshape(n,n)
        C = C.reshape(nc,mc)
        res = numpy.linalg.norm(A@X + X@A.T + B@X@B.T + C)
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
        average_delta = numpy.convolve(numpy.diff(Lnorm) + numpy.diff(Knorm), numpy.ones(5)/5, mode='full')[-5]
        if average_delta > 0.01: # Arbitrary threshold
            print('Warning: the K and L matrices computations did not converge. Retrying with different starting point and a N={:d} search'.format(int(20*1.3**self._check_KL.calls)))
            K, L = self._compute_Kalman_matrices(matrices, N=int(20*1.3**self.check_KL.calls))
        else:
            return K, L



class PackageToOperator(BaseAgent):
    def __init__(self, bundle):
        # Maybe don't check at all ?
        if 'core.bundle.Bundle' not in bundle.__class__.__mro__.__str__():
            return TypeError('The first argument bundle should be of type Bundle')

        self.encapsulated_bundle = bundle
        agent_state = self.encapsulated_bundle.game_state
        agent_policy = BundlePolicy(self.encapsulated_bundle)
        from core.observer import base_operator_engine_specification
        observation_engine = RuleObservationEngine(deterministic_specification = base_operator_engine_specification)

        super().__init__(    'operator',
                            state = agent_state,
                            policy = agent_policy,
                            observation_engine = observation_engine
                                    )


    def step(self, action, convert = False):
        pass
