import operator
import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt

from core.space import State, StateElement
from core.helpers import flatten, hard_flatten
from core.agents import DummyAssistant, DummyOperator
from core.observation import BaseObservationEngine
from core.core import Handbook
from core.interactiontask import PipeTaskWrapper

import copy

import sys
from loguru import logger
import yaml
import time
import json

from tabulate import tabulate
from tqdm import tqdm
import pandas as pd
import scipy.optimize
import scipy.stats
import inspect
import seaborn as sns
from copy import copy
import statsmodels.stats.proportion


# List of kwargs for bundles init()
#    - reset_skip_first_half_step (if True, skips the first_half_step of the bundle on reset. The idea being that the internal state of the agent provided during initialization should not be updated during reset). To generate a consistent observation, what we do is run the observation engine, but without potential noisefactors.


class Bundle:
    """A bundle combines a task with an operator and an assistant. All bundles are obtained by subclassing this main Bundle class.

    A bundle will create the ``game_state`` by combining three states of the task, the operator and the assistant as well as the turn index. It also takes care of adding the assistant action substate to the operator state and vice-versa.
    It also takes care of rendering each of the three component in a single place.

    Bundle subclasses should only have to redefine the step() and reset() methods.


    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, assistant, **kwargs):
        logger.info("Starting to Bundle {}, {} and {}".format(
            task.__class__.__name__, operator.__class__.__name__, assistant.__class__.__name__))

        self.kwargs = kwargs
        self.task = task
        self.task.bundle = self
        self.operator = operator
        self.operator.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self

        # Form complete game state
        self.game_state = State()

        turn_index = StateElement(
            values=[0], spaces=[gym.spaces.Discrete(2)], possible_values=[None])

        self.game_state['turn_index'] = turn_index
        self.game_state["task_state"] = task.state
        self.game_state["operator_state"] = operator.state
        self.game_state["assistant_state"] = assistant.state

        if operator.policy is not None:
            self.game_state["operator_action"] = operator.policy.action_state
        else:
            self.game_state["operator_action"] = State()
            self.game_state["operator_action"]['action'] = StateElement()
        if assistant.policy is not None:
            self.game_state["assistant_action"] = assistant.policy.action_state
        else:
            self.game_state["assistant_action"] = State()
            self.game_state["assistant_action"]['action'] = StateElement()

        logger.info('Finish Initializing task, operator and assistant')
        self.task.finit()
        self.operator.finit()
        self.assistant.finit()

        logger.info('Task state:\n{}'.format(str(self.task.state)))
        logger.info('Operator state:\n{}'.format(str(self.operator.state)))
        logger.info('Assistant state:\n{}'.format(str(self.assistant.state)))

        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211, 223, 224]
        self.rendered_mode = None
        self.render_perm = False
        self.playspeed = 0.1

        logger.info('\n========================\nCreated bundle {} with task {}, operator {}, and assistant {}\n========================\n'.format(
            self.__class__.__name__, task.__class__.__name__, operator.__class__.__name__, assistant.__class__.__name__))
        logger.info("The game state is \n{}".format(str(self.game_state)))
        logger.info('\n\n========================\n')

        self.handbook = Handbook(
            {'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})

    def __repr__(self):
        return "{}\n".format(self.__class__.__name__) + yaml.safe_dump(self.__content__())

    def __content__(self):
        return {"Task": self.task.__content__(), "Operator": self.operator.__content__(), "Assistant": self.assistant.__content__()}

    def reset(self, task=True, operator=True, assistant=True, dic={}):
        """ Reset the bundle. When subclassing Bundle, make sure to call super().reset() in the new reset method

        :param dic: (dictionnary) Reset the bundle with a game_state

        :return: (list) Flattened game_state

        :meta private:
        """
        if task:
            task_dic = dic.get('task_state')
            task_state = self.task.reset(dic=task_dic)

        if operator:
            operator_dic = dic.get('operator_state')
            operator_state = self.operator.reset(dic=operator_dic)

        if assistant:
            assistant_dic = dic.get("assistant_state")
            assistant_state = self.assistant.reset(dic=assistant_dic)

        logger.info('Resetting {}'.format(self.__class__.__name__))
        logger.info('Reset dic used:\n{}'.format(str(dic)))
        logger.info('Reset to state:\n{}'.format(str(self.game_state)))

        return self.game_state

    def step(self, action):
        """ Define what happens with the bundle when applying a joint action. Should be redefined when subclassing bundle.

        :param action: (list) joint action.

        :return: observation, sum_rewards, is_done, rewards

        :meta public:
        """
        logger.info('Stepping into {} with action {}'.format(
            self.__class__.__name__, str(action)))

    def render(self, mode, *args, **kwargs):
        """ Combines all render methods.

        :param mode: (str) text or plot

        :meta public:
        """
        self.rendered_mode = mode
        if 'text' in mode:
            print('Task Render')
            self.task.render(mode='text', *args, **kwargs)
            print("Operator Render")
            self.operator.render(mode='text', *args, **kwargs)
            print('Assistant Render')
            self.assistant.render(mode='text', *args, **kwargs)
        if 'log' in mode:
            logger.info('Task Render')
            self.task.render(mode='log', *args, **kwargs)
            logger.info("Operator Render")
            self.operator.render(mode='log', *args, **kwargs)
            logger.info('Assistant Render')
            self.assistant.render(mode='log', *args, **kwargs)
        if 'plot' in mode:
            if self.active_render_figure:
                plt.pause(self.playspeed)
                self.task.render(self.axtask, self.axoperator,
                                 self.axassistant, mode=mode, *args, **kwargs)
                self.operator.render(
                    self.axtask, self.axoperator, self.axassistant, mode='plot', *args, **kwargs)
                self.assistant.render(
                    self.axtask, self.axoperator, self.axassistant, mode='plot', *args, **kwargs)
                self.fig.canvas.draw()
            else:
                self.active_render_figure = True
                self.fig = plt.figure()
                self.axtask = self.fig.add_subplot(self.figure_layout[0])
                self.axtask.set_title('Task State')
                self.axoperator = self.fig.add_subplot(self.figure_layout[1])
                self.axoperator.set_title('Operator State')
                self.axassistant = self.fig.add_subplot(self.figure_layout[2])
                self.axassistant.set_title('Assistant State')
                self.task.render(self.axtask, self.axoperator,
                                 self.axassistant, mode='plot', *args, **kwargs)
                self.operator.render(
                    self.axtask, self.axoperator, self.axassistant, *args,  mode='plot', **kwargs)
                self.assistant.render(
                    self.axtask, self.axoperator, self.axassistant, *args,  mode='plot', **kwargs)
                self.fig.show()

            plt.tight_layout()

        if not ('plot' in mode or 'text' in mode):
            self.task.render(None, mode=mode, *args, **kwargs)
            self.operator.render(None, mode=mode, *args, **kwargs)
            self.assistant.render(None, mode=mode, *args, **kwargs)

    def close(self):
        """ Close bundle. Call this after the bundle returns is_done True.

        :meta public:
        """
        if self.active_render_figure:
            plt.close(self.fig)
            self.active_render_figure = None

        logger.info("Closing bundle {}".format(self.__class__.__name__))

    def _operator_first_half_step(self):
        """ This is the first half of the operator step, where the operaror observes the game state and updates its state via inference.

        :return: operator_obs_reward, operator_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """

        if not self.kwargs.get('onreset_deterministic_first_half_step'):
            operator_obs_reward, operator_infer_reward = self.operator.agent_step()

        else:
            logger.info('Observing with probabilistic rules excluded')
            # Store the probabilistic rules
            store = self.operator.observation_engine.extraprobabilisticrules
            # Remove the probabilistic rules
            self.operator.observation_engine.extraprobabilisticrules = {}
            # Generate an observation without generating an inference
            operator_obs_reward, operator_infer_reward = self.operator.agent_step(
                infer=False)
            # Reposition the probabilistic rules, and reset mapping
            self.operator.observation_engine.extraprobabilisticrules = store
            self.operator.observation_engine.mapping = None

        self.kwargs['onreset_deterministic_first_half_step'] = False
        logger.info('Observation rewards: {} / Inference rewards: {}'.format(
            operator_obs_reward, operator_infer_reward))

        return operator_obs_reward, operator_infer_reward

    def _operator_second_half_step(self, operator_action):
        """ This is the second half of the operator step. The operaror takes an action, which is applied to the task leading to a new game state.

        :param operator_action: (list) operator action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """

        logger.info('Applying action to task ---')
        # Play operator's turn in the task
        task_state, task_reward, is_done, _ = self.task.operator_step(
            operator_action)

        # update task state (likely not needed, remove ?)
        self.broadcast_state('operator', 'task_state', task_state)

        logger.info('Resulting task state:')
        logger.info(str(task_state))
        logger.info('Associated rewards: {}'.format(task_reward))

        return task_reward, is_done

    def _assistant_first_half_step(self):
        """ This is the first half of the assistant step, where the assistant observes the game state and updates its state via inference.

        :return: assistant_obs_reward, assistant_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """
        logger.info('Assistant {} first half step'.format(
            self.operator.__class__.__name__))

        assistant_obs_reward, assistant_infer_reward = self.assistant.agent_step()

        logger.info('Observation rewards: {} / Inference rewards: {}'.format(
            assistant_obs_reward, assistant_infer_reward))

        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        """ This is the second half of the assistant step. The assistant takes an action, which is applied to the task leading to a new game state.

        :param assistant_action: (list) assistant action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """
        # update action_state

        # Play assistant's turn in the task
        logger.info('Assistant second half step')

        task_state, task_reward, is_done, _ = self.task.assistant_step(
            assistant_action)
        # update task state
        self.broadcast_state('assistant', 'task_state', task_state)

        logger.info('Applied {} to task, which resulted in the new state'.format(
            str(assistant_action)))
        logger.info(str(task_state))
        logger.info('Associated rewards: {}'.format(task_reward))
        logger.info('task is done: {}'.format(is_done))

        return task_reward, is_done

    def _operator_step(self, *args, infer=True):
        """ Combines the first and second half step of the operator.

        :param args: (None or list) either provide the operator action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: operator_obs_reward, operator_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
        """
        logger.info(
            ' ---------- >>>> operator {} step'.format(self.operator.__class__.__name__))
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        try:
            # If human input is provided
            operator_action = args[0]
        except IndexError:
            # else sample from policy
            operator_action, operator_policy_reward = self.operator.take_action()

        self.broadcast_action('operator', operator_action)

        task_reward, is_done = self._operator_second_half_step(operator_action)
        logger.info(' <<<<<<<< operator {} step'.format(
            self.operator.__class__.__name__))
        return operator_obs_reward, operator_infer_reward, operator_policy_reward, task_reward, is_done

    def _assistant_step(self, *args):
        """ Combines the first and second half step of the assistant.

        :param args: (None or list) either provide the assistant action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: assistant_obs_reward, assistant_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
        """
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()

        try:
            # If human input is provided
            assistant_action = args[0]
            manual = True
        except IndexError:
            # else sample from policy
            assistant_action, assistant_policy_reward = self.assistant.take_action()
            manual = False

        self.broadcast_action('assistant', assistant_action)

        task_reward, is_done = self._assistant_second_half_step(
            assistant_action)
        return assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, task_reward, is_done

    def broadcast_state(self, role, state_key, state):
        self.game_state[state_key] = state
        getattr(self, role).observation[state_key] = state

    def broadcast_action(self, role, action, key=None):
        # update game state and observations
        if key is None:
            getattr(self, role).policy.action_state['action'] = action
            getattr(self, role).observation['{}_action'.format(
                role)]["action"] = action
        else:
            getattr(self, role).policy.action_state['action'][key] = action
            getattr(self, role).observation['{}_action'.format(
                role)]["action"][key] = action


class PlayNone(Bundle):
    """ A bundle which samples actions directly from operators and assistants. It is used to evaluate an operator and an assistant where the policies are already implemented.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_state = None

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)

    def step(self, infer=True):
        """ Play a step, actions are obtained by sampling the agent's policies.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(None)

        operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, is_done = self._operator_step(
            infer=infer)
        if is_done:
            return self.game_state, sum([operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward]
        assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward,  is_done = self._assistant_step()
        return self.game_state, sum([operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]


class PlayOperator(Bundle):
    """ A bundle which samples assistant actions directly from the assistant but uses operator actions provided externally in the step() method.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_space = copy.copy(
            self.operator.policy.action_state['action']['spaces'])

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)
        self._operator_first_half_step()
        return self.operator.observation
        # return self.operator.inference_engine.buffer[-1]

    def step(self, operator_action):
        """ Play a step, assistant actions are obtained by sampling the agent's policy and operator actions are given externally in the step() method.

        :param operator_action: (list) operator action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(operator_action)

        self.broadcast_action('operator', operator_action, key='values')
        first_task_reward, is_done = self._operator_second_half_step(
            operator_action)
        if is_done:
            return self.operator.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward,  assistant_policy_reward, second_task_reward, is_done = self._assistant_step()
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]


class PlayAssistant(Bundle):
    """ A bundle which samples oeprator actions directly from the operator but uses assistant actions provided externally in the step() method.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_space = self.assistant.policy.action_state['action']['spaces']

        # assistant.policy.action_state['action'] = StateElement(
        #     values = None,
        #     spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(assistant.policy.action_state['action']))],
        #     possible_values = None
        #      )

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle. A first  operator step and assistant observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)
        self._operator_step()
        self._assistant_first_half_step()
        return self.assistant.inference_engine.buffer[-1]

    def step(self, assistant_action):
        """ Play a step, operator actions are obtained by sampling the agent's policy and assistant actions are given externally in the step() method.

        :param assistant_action: (list) assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(assistant_action)

        self.broadcast_action('assistant', assistant_action, key='values')
        second_task_reward, is_done = self._assistant_second_half_step(
            assistant_action)
        if is_done:
            return self.assistant.inference_engine.buffer[-1], second_task_reward, is_done, [second_task_reward]
        operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, is_done = self._operator_step()
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()
        return self.assistant.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, operator_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]


class PlayBoth(Bundle):
    """ A bundle which samples both actions directly from the operator and assistant.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_space = self._tuple_to_flat_space(gym.spaces.Tuple(
            [self.operator.action_space, self.assistant.action_space]))

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle. Operator observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)
        self._operator_first_half_step()
        return self.task.state

    def step(self, joint_action):
        """ Play a step, operator and assistant actions are given externally in the step() method.

        :param joint_action: (list) joint operator assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(joint_action)

        operator_action, assistant_action = joint_action
        first_task_reward, is_done = self._operator_second_half_step(
            operator_action)
        if is_done:
            return self.task.state, first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward,  is_done = self._assistant_step()
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.task.state, sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]


class SinglePlayOperator(Bundle):
    """ A bundle without assistant. This is used e.g. to model psychophysical tasks such as perception, where there is no real interaction loop with a computing device.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, operator, **kwargs):
        super().__init__(task, operator, DummyAssistant(), **kwargs)

    @property
    def observation(self):
        return self.operator.observation

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)
        self._operator_first_half_step()
        return self.observation

    def step(self, operator_action):
        """ Play a step, operator actions are given externally in the step() method.

        :param operator_action: (list) operator action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(operator_action)

        self.broadcast_action('operator', operator_action)
        first_task_reward, is_done = self._operator_second_half_step(
            operator_action)
        if is_done:
            return self.operator.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        self.task.assistant_step([None])
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward]


class SinglePlayOperatorAuto(Bundle):
    """ Same as SinglePlayOperator, but this time the operator action is obtained by sampling the operator policy.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param kwargs: additional controls to account for some specific subcases. See Doc for a full list.

    :meta public:
    """

    def __init__(self, task, operator, **kwargs):
        super().__init__(task, operator, DummyAssistant(), **kwargs)
        self.action_space = None
        self.kwargs = kwargs

        _start_at_action = {'value': kwargs.get(
            'start_at_action'), 'meaning': 'If Start at action is True, then the reset will first perform an observation before returning.'}
        self.handbook['kwargs'] = [_start_at_action]

    @property
    def observation(self):
        return self.operator.observation

    def reset(self, dic={}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        super().reset(dic=dic, **kwargs)

        if self.kwargs.get('start_at_action'):
            self._operator_first_half_step()
            return self.observation

        return self.game_state
        # Return observation

    def step(self):
        """ Play a step, operator actions are obtained by sampling the agent's policy.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        if not self.kwargs.get('start_at_action'):
            operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        operator_action, operator_policy_reward = self.operator.take_action()
        self.broadcast_action('operator', operator_action)

        first_task_reward, is_done = self._operator_second_half_step(
            operator_action)
        if is_done:
            return self.observation, first_task_reward, is_done, [first_task_reward]
        _, _, _, _ = self.task.assistant_step([0])
        if self.kwargs.get('start_at_action'):
            operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.observation, sum([operator_obs_reward, operator_infer_reward, first_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward]


# Wrappers
# ====================


class BundleWrapper(Bundle):
    def __init__(self, bundle):
        self.__class__ = type(bundle.__class__.__name__,
                              (self.__class__, bundle.__class__), {})
        self.__dict__ = bundle.__dict__


class PipedTaskBundleWrapper(Bundle):
    # Wrap it by taking over bundles attribute via the instance __dict__. Methods can not be taken like that since they belong to the class __dict__ and have to be called via self.bundle.method()
    def __init__(self, bundle, taskwrapper, pipe):
        self.__dict__ = bundle.__dict__  # take over bundles attributes
        self.bundle = bundle
        self.pipe = pipe
        pipedtask = taskwrapper(bundle.task, pipe)
        self.bundle.task = pipedtask  # replace the task with the piped task
        bundle_kwargs = bundle.kwargs
        bundle_class = self.bundle.__class__
        self.bundle = bundle_class(
            pipedtask, bundle.operator, bundle.assistant, **bundle_kwargs)

        self.framerate = 1000
        self.iter = 0

        self.run()

    def run(self, reset_dic={}, **kwargs):
        reset_kwargs = kwargs.get('reset_kwargs')
        if reset_kwargs is None:
            reset_kwargs = {}
        self.bundle.reset(dic=reset_dic, **reset_kwargs)
        time.sleep(1/self.framerate)
        while True:
            obs, sum_reward, is_done, rewards = self.bundle.step()
            time.sleep(1/self.framerate)
            if is_done:
                break
        self.end()

    def end(self):
        self.pipe.send("done")


#  =====================
# Train

# https://stackoverflow.com/questions/1012185/in-python-how-do-i-index-a-list-with-another-list/1012197
#
# class Flexlist(list):
#     def __getitem__(self, keys):
#         if isinstance(keys, (int, numpy.int, slice)): return list.__getitem__(self, keys)
#         return [self[k] for k in keys]
#
# class Flextuple(tuple):
#     def __getitem__(self, keys):
#         if isinstance(keys, (int, numpy.int, slice)): return tuple.__getitem__(self, keys)
#         return [self[k] for k in keys]


class Train(gym.Env):
    """ Use this class to wrap a Bundle up, so that it is compatible with the gym API and can be trained with off-the-shelf RL algorithms.


    The observation size can be reduced by using the squeeze_output function, removing irrelevant substates of the game state.

    :param bundle: (core.bundle.Bundle) A bundle.

    :meta public:
    """

    def __init__(self, bundle, *args, **kwargs):
        self.bundle = bundle
        self.action_space = gym.spaces.Tuple(bundle.action_space)

        obs = bundle.reset()

        self.observation_mode = kwargs.get('observation_mode')
        self.observation_dict = kwargs.get('observation_dict')

        if self.observation_mode is None:
            self.observation_space = obs.filter('spaces', obs)
        elif self.observation_mode == 'tuple':
            self.observation_space = gym.spaces.Tuple(
                hard_flatten(obs.filter('spaces', self.observation_dict)))
        elif self.observation_mode == 'multidiscrete':
            self.observation_space = gym.spaces.MultiDiscrete(
                [i.n for i in hard_flatten(obs.filter('spaces', self.observation_dict))])
        elif self.observation_mode == 'dict':
            self.observation_space = obs.filter(
                'spaces', self.observation_dict)
        else:
            raise NotImplementedError

    def convert_observation(self, observation):
        if self.observation_mode is None:
            return observation
        elif self.observation_mode == 'tuple':
            return self.convert_observation_tuple(observation)
        elif self.observation_mode == 'multidiscrete':
            return self.convert_observation_multidiscrete(observation)
        elif self.observation_mode == 'dict':
            return self.convert_observation_dict(observation)
        else:
            raise NotImplementedError

    def convert_observation_tuple(self, observation):
        return tuple(hard_flatten(observation.filter('values', self.observation_dict)))

    def convert_observation_multidiscrete(self, observation):
        return numpy.array(hard_flatten(observation.filter('values', self.observation_dict)))

    def convert_observation_dict(self, observation):
        return observation.filter('values', self.observation_dict)

    def reset(self, dic={}, **kwargs):
        """ Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(dic=dic, **kwargs)
        return self.convert_observation(obs)

    def step(self, action):
        """ Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """

        obs, sum_reward, is_done, rewards = self.bundle.step(action)

        return self.convert_observation(obs), sum_reward, is_done, {'rewards': rewards}

    def render(self, mode):
        """ See Bundle

        :meta public:
        """
        self.bundle.render(mode)

    def close(self):
        """ See Bundle

        :meta public:
        """
        self.bundle.close()


class _DevelopTask(Bundle):
    """ A bundle without operator or assistant. It can be used when developping tasks to facilitate some things like rendering
    """

    def __init__(self, task, **kwargs):
        operator = kwargs.get("operator")
        if operator is None:
            operator = DummyOperator()
        else:
            kwargs.pop('operator')

        assistant = kwargs.get('assistant')
        if assistant is None:
            assistant = DummyAssistant()
        else:
            kwargs.pop("assistant")

        super().__init__(task, operator, assistant, **kwargs)

    def reset(self, dic={}, **kwargs):
        super().reset(dic=dic, **kwargs)

    def step(self, joint_action):
        operator_action, assistant_action = joint_action
        self.game_state["assistant_action"]['action']['values'] = assistant_action
        self.game_state['operator_action']['action']['values'] = operator_action
        self.task.operator_step(operator_action)
        self.task.assistant_step(assistant_action)


class _DevelopOperator(SinglePlayOperator):
    """A bundle without an assistant. It can be used when developing operators and
    includes methods for modeling checks (e.g. parameter or model recovery).

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) An operator, which is a subclass of BaseAgent
    :param kwargs: Additional controls to account for some specific subcases, see Doc for a full list
    """

    def _operator_can_compute_likelihood(operator):
        """Returns whether the specified operator's policy has a method called "compute_likelihood".

        :param operator: An operator, which is a subclass of BaseAgent
        :type operator: core.agents.BaseAgent
        """
        # Method name
        COMPUTE_LIKELIHOOD = "compute_likelihood"

        # Method exists
        policy_has_attribute_compute_likelihood = hasattr(
            operator.policy, COMPUTE_LIKELIHOOD)

        # Method is callable
        compute_likelihood_is_a_function = callable(
            getattr(operator.policy, COMPUTE_LIKELIHOOD))

        # Return that both exists and is callable
        operator_can_compute_likelihood = policy_has_attribute_compute_likelihood and compute_likelihood_is_a_function
        return operator_can_compute_likelihood

    def test_parameter_recovery(self, parameter_fit_bounds, correlation_threshold=0.7, significance_level=0.05, n_simulations=20, plot=True, save_plot_to=None):
        """Returns whether the recovered operator parameters correlate to the used parameters for a simulation given the supplied thresholds
        (only available for operators with a policy that has a compute_likelihood method).

        It simulates n_simulations agents of the operator's class using random parameters within the supplied parameter_fit_bounds,
        executes the provided task and tries to recover the operator's parameters from the simulated data. These recovered parameters are then
        correlated to the originally used parameters for the simulation using Pearson's r and checks for the given correlation and significance
        thresholds.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :param plot: Flag whether to plot the correlation between the used and recovered parameters and the correlation statistics, defaults to True
        :type plot: bool, optional
        :param save_plot_to: String of path where to save the resulting correlation scatter plot (None if not saved), defaults to None
        :type save_plot_to: str, optional
        :return: `True` if the correlation between used and recovered parameter values meets the supplied thresholds, `False` otherwise
        :rtype: bool
        """
        # Make sure operator has a policy that can compute likelihood of an action given an observation
        operator_can_compute_likelihood = _DevelopOperator._operator_can_compute_likelihood(
            self.operator)

        # If it cannot compute likelihood...
        if not operator_can_compute_likelihood:
            # Raise an exception
            raise ValueError(
                "Sorry, the given checks are only implemented for operator's with a policy that has a compute_likelihood method so far.")

        # Calculate the correlations between used and recovered parameters for and from simulation
        correlations = self._correlations(
            parameter_fit_bounds=parameter_fit_bounds, correlation_threshold=correlation_threshold, significance_level=significance_level, n_simulations=n_simulations, plot=plot, save_plot_to=save_plot_to)

        # If wanted, ...
        if plot:
            # Print the correlations between the used and recovered parameters as a table
            correlations_table = tabulate(
                correlations, headers="keys", tablefmt="psql", showindex=False)
            print(correlations_table)

        # Check that all correlations meet the threshold and are significant and return the result
        all_correlations_meet_threshold = correlations[f"r>{correlation_threshold}"].all(
        )
        all_correlations_significant = correlations[f"p<{significance_level}"].all(
        )

        return all_correlations_meet_threshold and all_correlations_significant

    def _correlations(self, parameter_fit_bounds, correlation_threshold=0.7, significance_level=0.05, n_simulations=20, plot=True, save_plot_to=None):
        """Returns a DataFrame containing the correlation information for the recovery of each specified parameter.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :param plot: Flag whether to plot the correlation between the used and recovered parameters, defaults to True
        :type plot: bool, optional
        :param save_plot_to: String of path where to save the resulting correlation scatter plot (None if not saved), defaults to None
        :type save_plot_to: str, optional
        :return: A DataFrame containing the correlation information for the recovery of each specified parameter.
        :rtype: pandas.DataFrame
        """
        # Transform the specified dict of parameter fit bounds into an OrderedDict
        # based on the order of parameters in the operator class constructor
        ordered_parameter_fit_bounds = self._ordered_parameter_fit_bounds(
            unordered_parameter_fit_bounds=parameter_fit_bounds)

        # Compute the likelihood data (i.e. the used and recovered parameter pairs)
        likelihood = self._likelihood(
            parameter_fit_bounds=ordered_parameter_fit_bounds, n_simulations=n_simulations)

        # If wanted, ...
        if plot:
            # Plot the correlations between the used and recovered parameters as a graph
            path_for_file_output = save_plot_to
            self._plot_correlations(
                parameter_fit_bounds=ordered_parameter_fit_bounds, data=likelihood, save_to=path_for_file_output)

        # Compute the correlation metric Pearson's r and its significance for each parameter pair and return it
        pearsons_r = self._pearsons_r(parameter_fit_bounds=ordered_parameter_fit_bounds, data=likelihood,
                                      correlation_threshold=correlation_threshold, significance_level=significance_level)

        return pearsons_r

    def _ordered_parameter_fit_bounds(self, unordered_parameter_fit_bounds):
        """Returns an OrderedDict representing the parameter fit bounds based on the order of the
        operator class parameters and the unordered parameter fit bounds.

        :param unordered_parameter_fit_bounds: A dictionary of the parameter names and their associated fit bounds.
        :type unordered_parameter_fit_bounds: dict
        :return: An OrderedDict representing the parameter fit bounds based on the order of the
            operator class parameters and the unordered parameter fit bounds
        :rtype: collections.OrderedDict
        """
        # Create an OrderedDict to store the parameter fit bounds
        ordered_parameter_fit_bounds = OrderedDict()

        # Get an OrderedDict of the parameter names and types of the operator class constructor
        operator_parameters = inspect.signature(
            self.operator.__class__).parameters

        # For each parameter in the operator class instructor...
        for param_name, _ in operator_parameters.items():

            # If the parameter name is included in the provided unordered parameter fit bounds...
            if param_name in unordered_parameter_fit_bounds.keys():

                # Add the parameter fit bounds to the OrderedDict
                ordered_parameter_fit_bounds[param_name] = unordered_parameter_fit_bounds[param_name]

        # And return the ordered parameter fit bounds
        return ordered_parameter_fit_bounds

    def _likelihood(self, parameter_fit_bounds, n_simulations=20):
        """Returns a DataFrame containing the likelihood of each recovered parameter.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :return: A DataFrame containing the likelihood of each recovered parameter.
        :rtype: pandas.DataFrame
        """
        # Data container
        likelihood_data = []

        # For each agent...
        for _ in tqdm(range(n_simulations), file=sys.stdout):

            # Generate a random agent
            random_agent = None
            if not len(parameter_fit_bounds) > 0:
                random_agent = self.operator.__class__()
            else:
                random_parameters = _DevelopOperator._random_parameters(
                    parameter_fit_bounds=parameter_fit_bounds)
                random_agent = self.operator.__class__(**random_parameters)

            # Simulate the task
            simulated_data = self._simulate(operator=random_agent)

            # Determine best-fit parameter values
            best_fit_parameters, _ = self._best_fit_parameters(
                operator_class=self.operator.__class__, parameter_fit_bounds=parameter_fit_bounds, data=simulated_data)

            # Backup parameter values
            for parameter_index, (parameter_name, parameter_value) in enumerate(random_parameters.items()):
                _, best_fit_parameter_value = best_fit_parameters[parameter_index]
                likelihood_data.append({
                    "Parameter": parameter_name,
                    "Used to simulate": parameter_value,
                    "Recovered": best_fit_parameter_value
                })

        # Create dataframe and return it
        likelihood = pd.DataFrame(likelihood_data)

        return likelihood

    def _random_parameters(parameter_fit_bounds):
        """Returns a dictionary of parameter-value pairs where the value is random within the specified fit bounds.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :return: A dictionary of parameter-value pairs where the value is random within the specified fit bounds
        :rtype: dict
        """
        # Data container
        random_parameters = {}

        # If parameters and their fit bounds were specified
        if len(parameter_fit_bounds) > 0:

            # For each parameter and their fit bounds
            for (current_parameter_name, current_parameter_fit_bounds) in parameter_fit_bounds.items():

                # Compute a random parameter value within the fit bounds
                random_parameters[current_parameter_name] = numpy.random.uniform(
                    *current_parameter_fit_bounds)

        # Return the parameter-value pairs
        return random_parameters

    def _simulate(self, operator=None):
        """Returns a DataFrame containing the behavioral data from simulating the given task with
        the specified operator.

        :param operator: The operator to use for the simulation (if None is specified, will use the operator
            of the bundle (i.e. self.operator)), defaults to None
        :type operator: core.agents.BaseAgent, optional
        :return: A DataFrame containing the behavioral data from simulating the given task with
            the specified operator
        :rtype: pandas.DataFrame
        """
        # Bundle definition
        operator_to_use_for_simulation = operator if operator is not None else self.operator
        bundle = SinglePlayOperatorAuto(
            self.task, operator_to_use_for_simulation)

        # Reset the bundle to default values
        bundle.reset()

        # Data container
        data = []

        # Flag whether the task has been completed
        done = False

        # While the task is not finished...
        while not done:

            # Save the current round number
            round = bundle.task.round

            # Simulate a round of the operator executing the task
            game_state, reward, done, _ = bundle.step()

            # Save the choice that the artificial agent made
            choice = copy(game_state["operator_action"]["action"])

            # Store this round's data
            data.append({
                "time": round,
                "choice": choice,
                "reward": reward
            })

        # When the task is done, create and return DataFrame
        simulated_data = pd.DataFrame(data)
        return simulated_data

    def _best_fit_parameters(self, operator_class, parameter_fit_bounds, data):
        """Returns a list of the parameters with their best-fit values based on the supplied data.

        :param operator_class: The operator class to find best-fit parameters for
        :type operator_class: core.agents.BaseAgent
        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param data: The behavioral data to infer the best-fit parameters from
        :type data: pandas.DataFrame
        :return: A list of the parameters with their best-fit values based on the supplied data
        :rtype: list
        """
        # If no parameters are specified...
        if not len(parameter_fit_bounds) > 0:
            # Calculate the negative log likelihood for the data without parameters...
            best_parameters = []
            ll = self._log_likelihood(
                operator_class=operator_class, parameter_values=best_parameters, data=data)
            best_objective_value = -ll

            # ...and return an empty list and the negative log-likelihood
            return best_parameters, best_objective_value

        # Define an initital guess
        random_initial_guess = _DevelopOperator._random_parameters(
            parameter_fit_bounds=parameter_fit_bounds)
        initial_guess = [parameter_value for _,
                         parameter_value in random_initial_guess.items()]

        # Run the optimizer
        res = scipy.optimize.minimize(
            fun=self._objective,
            x0=initial_guess,
            bounds=[fit_bounds for _,
                    fit_bounds in parameter_fit_bounds.items()],
            args=(operator_class, data))

        # Make sure that the optimizer ended up with success
        assert res.success

        # Get the best parameter value from the result
        best_parameter_values = res.x
        best_objective_value = res.fun

        # Construct a list for the best parameters and return it
        best_parameters = [(current_parameter_name, best_parameter_values[current_parameter_index])
                           for current_parameter_index, (current_parameter_name, _) in enumerate(parameter_fit_bounds.items())]

        return best_parameters, best_objective_value

    def _log_likelihood(self, operator_class, parameter_values, data):
        """Returns the log-likelihood of the specified parameter values given the provided data.

        :param operator_class: The operator class to compute the log-likelihood for
        :type operator_class: core.agents.BaseAgent
        :param parameter_values: A list of the parameter values to compute the log-likelihood for
        :type parameter_values: list
        :param data: The behavioral data to compute the log-likelihood for
        :type data: pandas.DataFrame
        :return: The log-likelihood of the specified parameter values given the provided data
        :rtype: float
        """
        # Data container
        ll = []

        # Create a new agent with the current parameters
        agent = None
        if not len(parameter_values) > 0:
            agent = operator_class()
        else:
            agent = operator_class(*parameter_values)

        # Bundle definition
        bundle = SinglePlayOperator(task=self.task, operator=agent)

        bundle.reset()

        # Simulate the task
        for _, row in data.iterrows():

            # Get choice and success for t
            choice, reward = row["choice"], row["reward"]

            # Get probability of this choice
            p = agent.policy.compute_likelihood(choice, agent.observation)

            # Compute log
            log = numpy.log(p + numpy.finfo(float).eps)
            ll.append(log)

            # Make operator take specified action
            _, resulting_reward, _, _ = bundle.step(operator_action=choice)

            # Compare simulated and resulting reward
            failure_message = """The provided operator action did not yield the same reward from the task.
            Maybe there is some randomness involved that could be solved by seeding."""
            assert reward == resulting_reward, failure_message

        return numpy.sum(ll)

    def _objective(self, parameter_values, operator_class, data):
        """Returns the negative log-likelihood of the specified parameter values given the provided data.

        :param parameter_values: A list of the parameter values to compute the log-likelihood for
        :type parameter_values: list
        :param operator_class: The operator class to calculate the negative log-likelihood for
        :type operator_class: core.agents.BaseAgent
        :param data: The behavioral data to compute the log-likelihood for
        :type data: pandas.DataFrame
        :return: The negative log-likelihood of the specified parameter values given the provided data
        :rtype: float
        """
        # Since we will look for the minimum,
        # let's return -LLS instead of LLS
        return - self._log_likelihood(operator_class=operator_class, parameter_values=parameter_values, data=data)

    def _plot_correlations(self, parameter_fit_bounds, data, save_to=None):
        """Plot the correlation between the true and recovered parameters.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param save_to: String of path where to save the resulting correlation scatter plot (None if not saved), defaults to None
        :type save_to: str, optional
        :param data: The correlation data including each parameter value used and recovered
        :type data: pandas.DataFrame
        """
        # Containers
        param_names = []
        param_bounds = []

        # Store parameter names and fit bounds in separate lists
        for (parameter_name, fit_bounds) in parameter_fit_bounds.items():
            param_names.append(parameter_name)
            param_bounds.append(fit_bounds)

        # Calculate number of parameters
        n_param = len(parameter_fit_bounds)

        # Define colors
        colors = [f'C{i}' for i in range(n_param)]

        # Create fig and axes
        _, axes = plt.subplots(ncols=n_param,
                               figsize=(10, 9))

        # For each parameter...
        for i in range(n_param):

            # Select ax
            ax = axes
            if n_param > 1:
                ax = axes[i]

            # Get param name
            p_name = param_names[i]

            # Set title
            ax.set_title(p_name)

            # Create scatter
            scatterplot = sns.scatterplot(data=data[data["Parameter"] == p_name],
                                          x="Used to simulate", y="Recovered",
                                          alpha=0.5, color=colors[i],
                                          ax=ax)

            # Plot identity function
            ax.plot(param_bounds[i], param_bounds[i],
                    linestyle="--", alpha=0.5, color="black", zorder=-10)

            # Set axes limits
            ax.set_xlim(*param_bounds[i])
            ax.set_ylim(*param_bounds[i])

            # Square aspect
            ax.set_aspect(1)

        # Display plot
        plt.tight_layout()

        # If wanted...
        if save_to:
            # Save plot in file
            path_for_file_output = save_to
            scatterplot.figure.savefig(path_for_file_output)

        plt.show()

    def _pearsons_r(self, parameter_fit_bounds, data, correlation_threshold=0.7, significance_level=0.05):
        """Returns a DataFrame containing the correlation value (Pearson's r) and significance for each parameter.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param data: The correlation data including each parameter value used and recovered
        :type data: pandas.DataFrame
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        """
        def pearson_r_data(parameter_name):
            """Returns a dictionary containing the parameter name and its correlation value and significance.

            :param parameter_name: The name of the parameter
            :type parameter_name: str
            :return: A dictionary containing the parameter name and its correlation value and significance
            :rtype: dict
            """
            # Get the elements to compare
            x = data.loc[data["Parameter"] ==
                         parameter_name, "Used to simulate"]
            y = data.loc[data["Parameter"] == parameter_name, "Recovered"]

            # Compute a Pearson correlation
            r, p = scipy.stats.pearsonr(x, y)

            # Return
            pearson_r_data = {
                "parameter": parameter_name,
                "r": r,
                f"r>{correlation_threshold}": r > correlation_threshold,
                "p": p,
                f"p<{significance_level}": p < significance_level
            }

            return pearson_r_data

        # Compute correlation data
        correlation_data = [pearson_r_data(parameter_name)
                            for parameter_name, _ in parameter_fit_bounds.items()]

        # Create dataframe
        pearsons_r = pd.DataFrame(correlation_data)

        return pearsons_r

    def test_model_recovery(self, other_competing_models, this_parameter_fit_bounds, f1_threshold=0.7, n_simulations_per_model=20, plot=True, save_plot_to=None):
        """Returns whether the bundle's operator model can be recovered from simulated data using the specified competing models
        meeting the specified F1-score threshold (only available for operators with a policy that has a compute_likelihood method).

        It simulates n_simulations agents for each of the operator's class and the competing models using random parameters within the supplied
        parameter_fit_bounds, executes the provided task and tries to recover the operator's best-fit parameters from the simulated data. Each of
        the best-fit models is then evaluated for fit using the BIC-score. The model recovery is then evaluated using recall, precision and
        the F1-score which is finally evaluated against the specified threshold.

        :param other_competing_models: A list of dictionaries for the other competing models including their parameter fit bounds (i.e. their names,
            their minimum and maximum values) that will be used for simulation (example: `[{"model": OperatorClass, "parameter_fit_bounds": {"alpha": (0., 1.), ...}}, ...]`)
        :type other_competing_models: list(dict)
        :param this_parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation of the bundle operator class (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type this_parameter_fit_bounds: dict
        :param f1_threshold: The threshold for F1-score, defaults to 0.7
        :type f1_threshold: float, optional
        :param n_simulations_per_model: The number of agents to simulate (i.e. the population size) for each model, defaults to 20
        :type n_simulations_per_model: int, optional
        :param plot: Flag whether to plot the confusion matrix and robustness statistics, defaults to True
        :type plot: bool, optional
        :param save_plot_to: String of path where to save the resulting confusion matrix plot (None if not saved), defaults to None
        :type save_plot_to: str, optional
        :return: `True` if the F1-score for the model recovery meets the supplied threshold, `False` otherwise
        :rtype: bool
        """
        # Transform this_parameter_fit_bounds to empty dict if falsy (e.g. [], {}, None, False)
        if not this_parameter_fit_bounds:
            this_parameter_fit_bounds = {}

        # All operator models that are competing
        all_operator_classes = other_competing_models + \
            [{"model": self.operator.__class__,
                "parameter_fit_bounds": this_parameter_fit_bounds}]

        # Calculate the confusion matrix between used and recovered models for and from simulation
        confusion_matrix = self._confusion_matrix(
            all_operator_classes=all_operator_classes, n_simulations=n_simulations_per_model)

        # If wanted, ...
        if plot:
            # Plot the confusion matrix
            path_for_output_file = save_plot_to
            _DevelopOperator._plot_confusion_matrix(
                data=confusion_matrix, save_to=path_for_output_file)

        # Get the model names
        model_names = [m["model"].__name__ for m in all_operator_classes]

        # Compute the model recovery statistics (recall, precision, f1)
        robustness = self._robustness_statistics(
            model_names=model_names, f1_threshold=f1_threshold, data=confusion_matrix)

        # If wanted, ...
        if plot:
            # Print the model recovery statistics (recall, precision, f1) as a table
            robustness_table = tabulate(
                robustness, headers="keys", tablefmt="psql", showindex=False)
            print(robustness_table)

        # Check that all correlations meet the threshold and are significant and return the result
        all_f1_meet_threshold = robustness[f"f1>{f1_threshold}"].all()
        return all_f1_meet_threshold

    def _bic(log_likelihood, k, n):
        """Returns the score for the Bayesian information criterion (BIC-score) for the given log-likelihood, number of parameters k and and number of observations n

        :param log_likelihood: The maximized value of the likelihood function for the model
        :type log_likelihood: float
        :param k: The number of parameters
        :type k: int
        :param n: The number of observations or rounds of the task that were played
        :type n: int
        :return: The score for the Bayesian information criterion (BIC-score) for the given log-likelihood, number of parameters k and and number of observations n
        :rtype: float
        """
        return -2 * log_likelihood + k * numpy.log(n)

    def _confusion_matrix(self, all_operator_classes, n_simulations=20):
        """Returns a DataFrame with the model recovery data (used to simulate vs recovered model) based on the BIC-score.

        :param all_operator_classes: The user models that are competing and can be recovered (example: `[{"model": OperatorClass, "parameter_fit_bounds": {"alpha": (0., 1.), ...}}, ...]`)
        :type all_operator_classes: list(dict)
        :param n_simulations: The number of agents to simulate (i.e. the population size) for each model, defaults to 20
        :type n_simulations: int, optional
        :return: A DataFrame with the model recovery data (used to simulate vs recovered model) based on the BIC-score
        :rtype: pandas.DataFrame
        """
        # Number of models
        n_models = len(all_operator_classes)

        # Data container
        confusion_matrix = numpy.zeros((n_models, n_models))

        # Set up progress bar
        with tqdm(total=n_simulations*n_models**2, file=sys.stdout) as pbar:

            # Loop over each model
            for i, operator_class_to_sim in enumerate(all_operator_classes):

                m_to_sim = operator_class_to_sim["model"]
                parameters_for_sim = operator_class_to_sim["parameter_fit_bounds"]

                for _ in range(n_simulations):

                    # Generate a random agent
                    random_agent = None
                    if not len(parameters_for_sim) > 0:
                        random_agent = m_to_sim()
                    else:
                        random_parameters = _DevelopOperator._random_parameters(
                            parameter_fit_bounds=parameters_for_sim)
                        random_agent = m_to_sim(**random_parameters)

                    # Simulate the task
                    simulated_data = self._simulate(
                        operator=random_agent)

                    # Container for BIC scores
                    bs_scores = numpy.zeros(n_models)

                    # For each model
                    for k, operator_class_to_fit in enumerate(all_operator_classes):

                        m_to_fit = operator_class_to_fit["model"]
                        parameters_for_fit = operator_class_to_fit["parameter_fit_bounds"]

                        # Determine best-fit parameter values
                        _, best_fit_objective_value = self._best_fit_parameters(
                            operator_class=m_to_fit, parameter_fit_bounds=parameters_for_fit, data=simulated_data)

                        # Get log-likelihood for best param
                        ll = -best_fit_objective_value

                        # Compute the BIC score
                        n_param_m_to_fit = len(parameters_for_fit)
                        bs_scores[k] = _DevelopOperator._bic(
                            log_likelihood=ll, k=n_param_m_to_fit, n=len(simulated_data))

                        # Update progress bar
                        pbar.update(1)

                    # Get minimum value for bic (min => best)
                    min_score = numpy.min(bs_scores)

                    # Get index(es) of models that get best bic
                    idx_min = numpy.flatnonzero(bs_scores == min_score)

                    # Add result in matrix
                    confusion_matrix[i, idx_min] += 1/len(idx_min)

        # Get the model names
        model_names = [m["model"].__name__ for m in all_operator_classes]

        # Create dataframe
        confusion = pd.DataFrame(confusion_matrix,
                                 index=model_names,
                                 columns=model_names)

        return confusion

    def _plot_confusion_matrix(data, save_to=None):
        """Plots the confusion matrix for the model recovery comparison.

        :param data: The confusion matrix (model used to simulate vs recovered model) as a DataFrame
        :type data: pandas.DataFrame
        :param save_to: String of path where to save the resulting confusion matrix plot (None if not saved), defaults to None
        :type save_to: str, optional
        """
        # Create figure and axes
        _, ax = plt.subplots(figsize=(12, 10))

        # Display the results using a heatmap
        heatmap = sns.heatmap(data=data, cmap='viridis', annot=True, ax=ax)

        # Set x-axis and y-axis labels
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.show()

        # If wanted...
        if save_to:
            # Save figure
            path_for_file_output = save_to
            heatmap.figure.savefig(path_for_file_output)

    def _recall(model_name, data):
        """Returns the recall value and its confidence interval for the given model and confusion matrix.

        :param model_name: The name of the model to compute the recall for
        :type model_name: str
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: The recall value and its confidence interval for the given model and confusion matrix
        :rtype: tuple[float, tuple[float, float]]
        """
        # Get the number of true positive
        k = data.at[model_name, model_name]

        # Get the number of true positive + false NEGATIVE
        n = numpy.sum(data.loc[model_name])

        # Compute the recall and return it
        recall = k/n

        # Compute the confidence interval
        ci_recall = statsmodels.stats.proportion.proportion_confint(
            count=k, nobs=n)

        return recall, ci_recall

    def _precision(model_name, data):
        """Returns the precision value and its confidence interval for the given model and confusion matrix.

        :param model_name: The name of the model to compute the precision for
        :type model_name: str
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: The precision value and its confidence interval for the given model and confusion matrix
        :rtype: tuple[float, tuple[float, float]]
        """
        # Get the number of true positive
        k = data.at[model_name, model_name]

        # Get the number of true positive + false POSITIVE
        n = numpy.sum(data[model_name])

        # Compute the precision
        precision = k/n

        # Compute the confidence intervals
        ci_pres = statsmodels.stats.proportion.proportion_confint(k, n)

        return precision, ci_pres

    def _f1(precision, recall):
        """Returns the F1-score for the given precision and recall.

        :param precision: The precision value.
        :type precision: float
        :param recall: The recall value.
        :type recall: float
        :return: The F1-score for the given precision and recall
        :rtype: float
        """
        # Compute the f score
        f_score = 2*(precision * recall)/(precision+recall)
        return f_score

    def _robustness_statistics(self, model_names, f1_threshold, data):
        """Returns a DataFrame with the robustness statistics (precision, recall, F1-score) based on the supplied confusion data and user models.

        :param model_names: The names of the user models that are competing and could be recovered
        :type all_operator_classes: list(str)
        :param f1_threshold: The threshold for F1-score, defaults to 0.7
        :type f1_threshold: float, optional
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: A DataFrame with the robustness statistics (precision, recall, F1-score) based on the supplied confusion data and user models
        :rtype: pandas.DataFrame
        """
        # Results container
        row_list = []

        # For each model...
        for m in model_names:

            # Compute the recall
            recall, ci_recall = _DevelopOperator._recall(
                model_name=m, data=data)

            # Compute the precision and confidence intervals
            precision, ci_pres = _DevelopOperator._precision(
                model_name=m, data=data)

            # Compute the f score
            f_score = _DevelopOperator._f1(precision=precision, recall=recall)

            # Backup
            row_list.append({
                "model": m,
                "Recall": recall,
                "Recall [CI]": ci_recall,
                "Precision": precision,
                "Precision [CI]": ci_pres,
                "F1 score": f_score,
                f"f1>{f1_threshold}": f_score > f1_threshold
            })

        # Create dataframe and display it
        stats = pd.DataFrame(row_list, index=model_names)

        return stats
