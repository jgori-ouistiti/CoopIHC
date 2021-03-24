import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict

from core.observation import BaseOperatorObservationRule, BaseAssistantObservationRule, RuleObservationEngine
from core.inference import InferenceEngine, GoalInferenceWithOperatorModelGiven
from core.models import BinaryOperatorModel
from core.helpers import flatten, sort_two_lists

import matplotlib.pyplot as plt

class BaseAgent(ABC):
    def __init__(self, role, action_space, action_set, observation_engine = None, inference_engine = None):

        if role == 0 or role == 'operator':
            self.role = [0, 'operator']
        elif role == 1 or role == 'assistant':
            self.role = [1, 'assistant']
        else:
            raise NotImplementedError

        self.action_space = gym.spaces.Tuple(action_space)
        self.action_set = tuple(action_set)
        self._state = OrderedDict()
        self.state_space = gym.spaces.Tuple([])
        self.state_dict = OrderedDict()
        self.bundle = None

        if observation_engine is None:
            if self.role[0] == 0:
                self.observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
            elif self.role[0] == 1:
                self.observation_engine = RuleObservationEngine(BaseAssistantObservationRule)
            else:
                raise NotImplementedError
        else:
            self.observation_engine = observation_engine
        self.observation_engine.host = self

        if inference_engine is None:
            self.inference_engine = InferenceEngine()
        else:
            self.inference_engine = inference_engine
        self.inference_engine.host = self

        # Rendering stuff
        self.ax = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        if self.bundle:
            self.bundle.game_state[self.role[1]+'_state'] = state

    @abstractmethod
    def finit(self):
        # Finish initializing agent
        pass

    @abstractmethod
    def reset(self):
        pass

    def append_state(self, substate, substate_space, possible_values = None):
        """Args:
            substate (str): Name of the substate.
            substate_space (list(gym.spaces)): List of the substate spaces.
            values (list(float, int)): Values on which to map Discrete actions (None for Boxes)
        """
        self.state_space = gym.spaces.Tuple(self.state_space.spaces + substate_space)
        self.modify_state(substate, possible_values = possible_values, substate_space = substate_space)

    def modify_state(self, substate, possible_values = None, value = None, substate_space = None):
        # You have to supply substate_space if value is None
        self.state_dict[substate] = possible_values
        state = self.state.copy()
        if value is not None:
            state[substate] = value
        else:
            state[substate] = [space.sample() for space in substate_space]
        self.state = state

    def agent_step(self):
        # agent observes the state
        agent_observation, agent_obs_reward = self.observe(self.bundle.game_state)
        # Pass observation to InferenceEngine Buffer
        self.inference_engine.add_observation(agent_observation)
        # Infer the new operator state
        agent_state, agent_infer_reward = self.inference_engine.infer()
        # Broadcast new agent_state
        self.state = agent_state
        # Update agent observation
        if self.role[0] == 0:
            self.observation['operator_state'] = agent_state
        elif self.role[0] == 1:
            self.observation['assistant_state'] = agent_state
        return agent_obs_reward, agent_infer_reward


    def observe(self, game_state):
        self.observation, reward = self.observation_engine.observe(self, game_state)
        return self.observation, reward

    def sample(self):
        actions = self.action_space.sample()
        if isinstance(actions, (int, float)):
            actions = [actions]
        return actions


    def render(self, *args, mode="mode"):
        if 'plot' in mode:
            axtask, axoperator, axassistant = args
            if self.ax is not None:
                pass
            else:
                if self.role[0] == 0:
                    self.ax = axoperator
                else:
                    self.ax = axassistant
                self.ax.axis('off')
                self.ax.set_title(type(self).__name__ + " State")
        if 'text' in mode:
            print(type(self).__name__ + " State")



### Goal could be defined as a target state of the task, in a more general description.
class GoalDrivenDiscreteOperator(BaseAgent):
    def __init__(self, operator_model, observation_engine = None):
        action_set = [operator_model.actions]
        action_space = [gym.spaces.Discrete(len(action_set))]
        self.operator_model = operator_model
        super().__init__(0, action_space, action_set, observation_engine = observation_engine, inference_engine = None)
        # Define goal_state here, which is the state of the task that the operator is trying to achieve.

    def finit(self):
        targets = self.bundle.task.state['Targets']
        self.append_state('Goal', [gym.spaces.Discrete(len(targets))], possible_values = targets)
        return

    def reset(self):
        targets = self.bundle.task.state['Targets']
        goal = numpy.random.choice(targets)
        self.modify_state('Goal', possible_values = targets, value = [goal])


    def sample(self, *args):
        if args:
            # maybe put args = observation for testing purposes
            raise NotImplementedError
        else:
            actions = self.operator_model.sample(self.observation)
            if isinstance(actions, (int, float)):
                actions = [actions]
            return actions

    def render(self, *args, mode="mode"):
        if 'plot' in mode:
            axtask, axoperator, axassistant = args
            if self.ax is not None:
                pass
            else:
                self.ax = axoperator
                self.ax.text(0,0, "Goal: {}".format(self.state['Goal'][0]))
                self.ax.set_xlim([-0.5,0.5])
                self.ax.set_ylim([-0.5,0.5])
                self.ax.axis('off')
                self.ax.set_title(type(self).__name__ + " Goal")
        if 'text' in mode:
            print(type(self).__name__ + " Goal")
            print(self.state['Goal'][0])




# An agent that has a substate called Beliefs, which are updated in a Bayesian fashion. Requires a model of the operator as well as the potential target states that can serve as goals. Subclass this to implement various policies w/r beliefs.

class BayesianBeliefAssistant(BaseAgent):
    def __init__(self, action_space, action_set, operator_model, observation_engine = None):
        inference_engine = GoalInferenceWithOperatorModelGiven(operator_model)
        super().__init__(1, action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

    def finit(self):
        targets = self.bundle.task.state['Targets']
        self.append_state( "Beliefs", [gym.spaces.Box( low = numpy.zeros( len( targets ),), high = numpy.ones( len( targets), ) )] )
        self.targets = targets

    def reset(self):
        targets = self.bundle.task.state['Targets']
        self.modify_state("Beliefs", value = [1/len(targets) for t in targets])
        self.inference_engine.reset()
        self.targets = targets

    def render(self, *args, mode = 'plotext'):
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
            axtask, axoperator, axassistant = args
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
