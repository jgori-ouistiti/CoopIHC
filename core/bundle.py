import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt

from core.space import State, StateElement
from core.helpers import flatten, hard_flatten
from core.agents import DummyAssistant, DummyOperator
from core.observation import ObservationEngine
from core.core import Handbook

import copy

import sys
from loguru import logger



######### List of kwargs for bundles init()
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
    def __init__(self, task, operator, assistant,**kwargs):
        logger.info("Starting to Bundle {}, {} and {}".format(task.__class__.__name__, operator.__class__.__name__, assistant.__class__.__name__))

        self.kwargs = kwargs
        self.task = task
        self.task.bundle = self
        self.operator = operator
        self.operator.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self

        # Form complete game state
        self.game_state = State()

        turn_index = StateElement(values = [0], spaces = [gym.spaces.Discrete(2)], possible_values = [None])
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
        self.figure_layout = [211,223,224]
        self.rendered_mode = None
        self.render_perm  = False
        self.playspeed = 0.1


        logger.info('\n========================\nCreated bundle {} with task {}, operator {}, and assistant {}\n========================\n'.format(self.__class__.__name__, task.__class__.__name__, operator.__class__.__name__, assistant.__class__.__name__))
        logger.info("The game state is \n{}".format(str(self.game_state)))
        logger.info('\n\n========================\n')

        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})

    def reset(self, dic = {}):
        """ Reset the bundle. When subclassing Bundle, make sure to call super().reset() in the new reset method

        :param dic: (dictionnary) Reset the bundle with a game_state

        :return: (list) Flattened game_state

        :meta private:
        """
        task_dic = dic.get('task_state')
        operator_dic = dic.get('operator_state')
        assistant_dic = dic.get("assistant_state")
        task_state = self.task.reset(dic = task_dic)
        operator_state = self.operator.reset(dic = operator_dic)
        assistant_state = self.assistant.reset(dic = assistant_dic)

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
        logger.info('Stepping into {} with action {}'.format(self.__class__.__name__, str(action)))

    def render(self, mode, *args, **kwargs):
        """ Combines all render methods.

        :param mode: (str) text or plot

        :meta public:
        """
        self.rendered_mode = mode
        if 'text' in mode:
            print('Task Render')
            self.task.render(mode='text', *args , **kwargs)
            print("Operator Render")
            self.operator.render(mode='text', *args , **kwargs)
            print('Assistant Render')
            self.assistant.render(mode = 'text', *args , **kwargs)
        if 'log' in mode:
            logger.info('Task Render')
            self.task.render(mode='log', *args , **kwargs)
            logger.info("Operator Render")
            self.operator.render(mode='log', *args , **kwargs)
            logger.info('Assistant Render')
            self.assistant.render(mode = 'log', *args , **kwargs)
        if 'plot' in mode:
            if self.active_render_figure:
                plt.pause(self.playspeed)
                self.task.render(self.axtask, self.axoperator, self.axassistant, mode = mode, *args , **kwargs)
                self.operator.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot', *args , **kwargs)
                self.assistant.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot', *args , **kwargs)
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
                self.task.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot', *args , **kwargs)
                self.operator.render(self.axtask, self.axoperator, self.axassistant, *args ,  mode = 'plot', **kwargs)
                self.assistant.render(self.axtask, self.axoperator, self.axassistant, *args ,  mode = 'plot', **kwargs)
                self.fig.show()

            plt.tight_layout()

        if not ('plot' in mode or 'text' in mode):
            self.task.render(None, mode = mode, *args, **kwargs)
            self.operator.render(None, mode = mode, *args, **kwargs)
            self.assistant.render(None, mode = mode, *args, **kwargs)

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
            operator_obs_reward, operator_infer_reward = self.operator.agent_step(infer = False)
            # Reposition the probabilistic rules, and reset mapping
            self.operator.observation_engine.extraprobabilisticrules = store
            self.operator.observation_engine.mapping = None

        logger.info('Observation rewards: {} / Inference rewards: {}'.format( operator_obs_reward, operator_infer_reward))

        return operator_obs_reward, operator_infer_reward




    def _operator_second_half_step(self, operator_action):
        """ This is the second half of the operator step. The operaror takes an action, which is applied to the task leading to a new game state.

        :param operator_action: (list) operator action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """

        logger.info('Applying action to task ---')
        # Play operator's turn in the task
        task_state, task_reward, is_done, _ = self.task.operator_step(operator_action)

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
        logger.info('Assistant {} first half step'.format(self.operator.__class__.__name__))

        assistant_obs_reward, assistant_infer_reward = self.assistant.agent_step()

        logger.info('Observation rewards: {} / Inference rewards: {}'.format( assistant_obs_reward, assistant_infer_reward))

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

        task_state, task_reward, is_done, _ = self.task.assistant_step(assistant_action)
        # update task state
        self.broadcast_state('operator', 'task_state', task_state)

        logger.info('Applied {} to task, which resulted in the new state'.format(str(assistant_action)))
        logger.info(str(task_state))
        logger.info('Associated rewards: {}'.format(task_reward))
        logger.info('task is done: {}'.format(is_done))

        return task_reward, is_done

    def _operator_step(self, *args):
        """ Combines the first and second half step of the operator.

        :param args: (None or list) either provide the operator action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: operator_obs_reward, operator_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
        """
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        try:
            # If human input is provided
            operator_action = args[0]
        except IndexError:
            # else sample from policy
            operator_action = self.operator.take_action()

        self.broadcast_action('operator', operator_action)

        task_reward, is_done = self._operator_second_half_step(operator_action)
        return operator_obs_reward, operator_infer_reward, task_reward, is_done



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
            assistant_action = self.assistant.take_action()
            manual = False


        print(assistant_action)

        self.broadcast_action('assistant', assistant_action)

        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return assistant_obs_reward, assistant_infer_reward, task_reward, is_done


    def broadcast_state(self, role, state_key, state):
        self.game_state[state_key] = state
        getattr(self, role).inference_engine.buffer[-1][state_key] = state

    def broadcast_action(self, role, action, key = None):
        # update game state and observations
        if key is None:
            getattr(self, role).policy.action_state['action'] = action
            getattr(self, role).inference_engine.buffer[-1]['{}_action'.format(role)]["action"] = action
        else:
            getattr(self, role).policy.action_state['action'][key] = action
            getattr(self, role).inference_engine.buffer[-1]['{}_action'.format(role)]["action"][key] = action



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

    def reset(self, dic = {}):
        """ Reset the bundle.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic)


    def step(self):
        """ Play a step, actions are obtained by sampling the agent's policies.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(None)

        operator_obs_reward, operator_infer_reward, first_task_reward, is_done = self._operator_step()
        if is_done:
            return operator_obs_reward, operator_infer_reward, task_reward, is_done
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step()
        return sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]

class PlayOperator(Bundle):
    """ A bundle which samples assistant actions directly from the assistant but uses operator actions provided externally in the step() method.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_space = copy.copy(self.operator.policy.action_state['action']['spaces'])

        # operator.policy.action_state['action'] = StateElement(
        #     values = None,
        #     spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(operator.policy.action_state['action']))],
        #     possible_values = None
        #      )



    def reset(self, dic = {}):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic)
        self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1]

    def step(self, operator_action):
        """ Play a step, assistant actions are obtained by sampling the agent's policy and operator actions are given externally in the step() method.

        :param operator_action: (list) operator action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(operator_action)

        self.broadcast_action('operator', operator_action, key = 'values')
        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.operator.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step()
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]


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

    def reset(self, dic = {}):
        """ Reset the bundle. A first  operator step and assistant observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic)
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

        self.broadcast_action('assistant', assistant_action, key = 'values')
        second_task_reward, is_done = self._assistant_second_half_step(assistant_action)
        if is_done:
            return self.assistant.inference_engine.buffer[-1], second_task_reward, is_done, [second_task_reward]
        operator_obs_reward, operator_infer_reward, first_task_reward, is_done = self._operator_step()
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()
        return self.assistant.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]

class PlayBoth(Bundle):
    """ A bundle which samples both actions directly from the operator and assistant.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param operator: (core.agents.BaseAgent) an operator, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, operator, assistant, **kwargs):
        super().__init__(task, operator, assistant, **kwargs)
        self.action_space = self._tuple_to_flat_space(gym.spaces.Tuple([self.operator.action_space, self.assistant.action_space]))

    def reset(self, dic = None):
        """ Reset the bundle. Operator observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic)
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
        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.task.state, first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step(assistant_action)
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


    def reset(self, dic = {}):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic)
        self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1]

    def step(self, operator_action):
        """ Play a step, operator actions are given externally in the step() method.

        :param operator_action: (list) operator action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(operator_action)

        self.broadcast_action('operator', operator_action)
        first_task_reward, is_done = self._operator_second_half_step(operator_action)
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



        _start_at_action = {'value': kwargs.get('start_at_action'), 'meaning': 'If Start at action is True, then the reset will first perform an observation before returning.'}
        self.handbook['kwargs'] = [_start_at_action]


    def reset(self, dic = {}):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        super().reset(dic = dic)

        if self.kwargs.get('start_at_action'):
            self._operator_first_half_step()
            return self.operator.inference_engine.buffer[-1]

        return self.game_state
        # Return observation


    def step(self):
        """ Play a step, operator actions are obtained by sampling the agent's policy.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        if self.kwargs.get('start_at_action'):
            operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        operator_action = self.operator.take_action()
        self.broadcast_action('operator', operator_action)

        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.operator.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        self.task.assistant_step([0])
        if not self.kwargs.get('start_at_action'):
            operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward]


## Wrappers
# ====================


class BundleWrapper(Bundle):
    def __init__(self, bundle):
        self.__class__ = type(bundle.__class__.__name__, (self.__class__, bundle.__class__), {})
        self.__dict__ = bundle.__dict__



## =====================
## Train

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
            self.observation_space = gym.spaces.Tuple(hard_flatten(obs.filter('spaces', self.observation_dict)))
        elif self.observation_mode == 'multidiscrete':
            self.observation_space = gym.spaces.MultiDiscrete([i.n for i in hard_flatten(obs.filter('spaces', self.observation_dict))])
        elif self.observation_mode == 'dict':
            self.observation_space = obs.filter('spaces', self.observation_dict)
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


    # def squeeze_output(self, extract_object):
    #     """ Call this on the environment to remove some of the outputs to lower the dimension of the observation space for the RL algorithms. To know which substates to keep, use get_state_mapping(). Call this right after initializing the environment.
    #
    #     :param extract_object: (slice, array) a set of indices which will be extracted from the flattened observation. extract_object can be any type used for indexing i.e. a slice or a numpy.ndarray e.g. ``env.squeeze_output(slice(3,5,1))`` to keep the substates at index 3 and 4 in the flattened game_state.
    #     """
    #     self.extract_object = extract_object
    #     self.observation_space = gym.spaces.Tuple(Flextuple(self.observation_space)[extract_object])


    def reset(self, dic = None):
        """ Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(dic)
        return self.convert_observation(obs)

    def step(self, action):
        """ Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """

        obs, sum_reward, is_done, rewards = self.bundle.step(action)

        return self.convert_observation(obs), sum_reward, is_done, {'rewards':rewards}

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
        super().__init__(task, DummyOperator(), DummyAssistant(), **kwargs)

    def reset(self, dic = {}):
        super().reset(dic = dic)

    def step(self, joint_action):
        operator_action, assistant_action = joint_action
        self.game_state["assistant_action"]['action']['values'] = assistant_action
        self.game_state['operator_action']['action']['values'] = operator_action
        self.task.operator_step(operator_action)
        self.task.assistant_step(assistant_action)
