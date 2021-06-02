import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt

from core.space import State, StateElement
from core.helpers import flatten
from core.agents import DummyAssistant, DummyOperator
from core.observation import ObservationEngine

import copy


######### List of kwargs for bundles init()
#    - reset_skip_first_half_step (if True, skips the first_half_step of the bundle on reset. The idea being that the internal state of the agent provided during initialization should not be updated during reset). To generate a consistent observation, what we do is run the observation engine, but without potential noisefactors.







class Bundle(ABC):
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

        #
        # bundle_state = State(name = 'turn_index', spacetype = [gym.spaces.Discrete(2)], possible_values = None, value = 0)
        #
        # game_state.add_substates([bundle_state, self.task.state, self.operator.state, self.assistant.state, self.operator.policy.action_state, self.assistant.policy.action_state])


        # self.game_state = OrderedDict({'b_state': OrderedDict({'next_agent': [0]}), 'task_state': self.task.state, 'operator_state': self.operator.state.Value.value, 'assistant_state': self.assistant.state.Value.value})
        # self.game_state['operator_action'] = self.operator.policy.action_state.Value.value
        # self.game_state['assistant_action'] = self.assistant.policy.action_state.Value.value


        # Finish Initializing task, operator and assistant
        self.task.finit()
        self.operator.finit()
        self.assistant.finit()

        # # Form joint action state
        # self.action_state = core.space.merge([self.operator.policy.action_state, self.assistant.policy.action_state])



        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211,223,224]
        self.rendered_mode = None
        self.render_perm  = False
        self.playspeed = 0.1

    # def _tuple_to_flat_space(self, tupled_spaces):
    #     """ Turn a ``gym.spaces.Tuple`` of Discrete and Box spaces to a single box space.
    #
    #     :param tupled_spaces: (gym.spaces.Tuple) a tuple of Box or Discrete spaces.
    #
    #     :return: (gym.spaces.Box) a gym Box
    #
    #     :meta private:
    #     """
    #     low = []
    #     high = []
    #     for space in tupled_spaces:
    #         if isinstance(space, gym.spaces.Box):
    #             low += space.low.tolist()
    #             high += space.high.tolist()
    #         elif isinstance(space, gym.spaces.Discrete): #Should we use onehot encoding at all ?
    #             low += [0]
    #             high += [space.n]
    #             # if space.n <= 2:
    #             #     low += [0]
    #             #     high += [1]
    #             #     self.spacedim += [1]
    #             #     self.onehot += [False]
    #             # else:
    #             #     low = [0 for i in range(space.n)]
    #             #     high = [1 for i in range(space.n)]
    #             #     self.spacedim += [space.n]
    #             #     self.onehot += [True]
    #         else:
    #             raise NotImplementedError
    #     return gym.spaces.Box(numpy.array(low), numpy.array(high))






    def reset(self, dic = None):
        """ Reset the bundle. When subclassing Bundle, make sure to call super().reset() in the new reset method

        :param dic: (dictionnary) Reset the bundle with a game_state

        :return: (list) Flattened game_state

        :meta private:
        """
        self.task.reset(dic)
        self.operator.reset(dic)
        self.assistant.reset(dic)


        # self.full_observation, self.full_observation_labels, self.indices, self.game_state_indices = self._flatten_game_state()


        return self.game_state



    # def _flatten_game_state(self):
    #     """ Used for __repr__
    #
    #     :meta private:
    #     """
    #     obs = flatten([list(d.values()) for d in list(self.game_state.Value.value.values())])
    #     labels = []
    #     indices = []
    #     upper_indices = []
    #     for pkey, pvalue in self.game_state.items():
    #         index = 0
    #         for skey, svalue in pvalue.items():
    #             indices.append((pkey, skey, len(flatten(svalue))))
    #             for n,i in enumerate(flatten(svalue)):
    #                 labels.append("/".join([pkey, skey, str(n)]))
    #                 index += 1
    #         upper_indices.append((pkey, index))
    #     return obs, labels, indices, upper_indices

    @abstractmethod
    def step(self, action, convert = False):
        """ Define what happens with the bundle when applying a joint action. Should be redefined when subclassing bundle.

        :param action: (list) joint action.

        :return: observation, sum_rewards, is_done, rewards

        :meta public:
        """
        pass

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

    def _operator_first_half_step(self):
        """ This is the first half of the operator step, where the operaror observes the game state and updates its state via inference.

        :return: operator_obs_reward, operator_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """
        if not self.kwargs.get('onreset_deterministic_first_half_step'):
            operator_obs_reward, operator_infer_reward = self.operator.agent_step()

        else:
            # Store the probabilistic rules
            store = self.operator.observation_engine.extraprobabilisticrules
            # Remove the probabilistic rules
            self.operator.observation_engine.extraprobabilisticrules = {}
            # Generate an observation without generating an inference
            operator_obs_reward, operator_infer_reward = self.operator.agent_step(infer = False)
            # Reposition the probabilistic rules, and reset mapping
            self.operator.observation_engine.extraprobabilisticrules = store
            self.operator.observation_engine.mapping = None

        return operator_obs_reward, operator_infer_reward




    def _operator_second_half_step(self, operator_action):
        """ This is the second half of the operator step. The operaror takes an action, which is applied to the task leading to a new game state.

        :param operator_action: (list) operator action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """

        # convert operator_action from human_readable


        # Play operator's turn in the task
        task_state, task_reward, is_done, _ = self.task.operator_step(operator_action)
        # update task state
        self.task.state.update(task_state)
        return task_reward, is_done

    def _assistant_first_half_step(self):
        """ This is the first half of the assistant step, where the assistant observes the game state and updates its state via inference.

        :return: assistant_obs_reward, assistant_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """
        assistant_obs_reward, assistant_infer_reward = self.assistant.agent_step()
        # assistant takes action

        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        """ This is the second half of the assistant step. The assistant takes an action, which is applied to the task leading to a new game state.

        :param assistant_action: (list) assistant action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """
        # update action_state

        # Play assistant's turn in the task
        task_state, task_reward, is_done, _ = self.task.assistant_step(assistant_action)
        # update task state
        self.task.state.update(task_state)
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

        self.broadcast_action('assistant', assistant_action)

        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return assistant_obs_reward, assistant_infer_reward, task_reward, is_done



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

    def reset(self, dic = None):
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
        self.action_state = copy.copy(self.operator.policy.action_state)

        operator.policy.action_state['action'] = StateElement(
            values = None,
            spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(operator.policy.action_state['action']))],
            possible_values = None
             )



    def reset(self, dic = None):
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
        self.broadcast_action('operator', operator_action)
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
        self.action_state = copy.copy(self.assistant.policy.action_state)

        assistant.policy.action_state['action'] = StateElement(
            values = None,
            spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(assistant.policy.action_state['action']))],
            possible_values = None
             )

    def reset(self, dic = None):
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

    def step(self, joint_action, convert = False):
        """ Play a step, operator and assistant actions are given externally in the step() method.

        :param joint_action: (list) joint operator assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
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


    def reset(self, dic = None):
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

    def reset(self, dic = None):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        super().reset(dic)
        self._operator_first_half_step()


        # Return observation
        return self.operator.inference_engine.buffer[-1]

    def step(self):
        """ Play a step, operator actions are obtained by sampling the agent's policy.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        operator_action = self.operator.take_action()
        self.broadcast_action('operator', operator_action)

        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.operator.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        self.task.assistant_step([0])
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.inference_engine.buffer[-1], sum([operator_obs_reward, operator_infer_reward, first_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward]


## Wrappers
# ====================

## =====================
## Train

class Train(gym.Env):
    """ Use this class to wrap a Bundle up, so that it is compatible with the gym API and can be trained with off-the-shelf RL algorithms.


    The observation size can be reduced by using the squeeze_output function, removing irrelevant substates of the game state.

    :param bundle: (core.bundle.Bundle) A bundle.

    :meta public:
    """
    def __init__(self, bundle):
        self.bundle = bundle
        self.action_space = bundle.action_space
        observation = flatten(bundle.reset())
        # For now let's assume that the observation space takes value in here. This should be normalized anyway.
        self.observation_space = gym.spaces.Box(low = -100, high = 100, shape = (len(observation),))
        self.extract_object = slice(0, self.observation_space.shape[0], 1)

    def get_state_mapping(self):
        """ Print out the game_state and the name of each substate with according indices.
        """
        obs = self.bundle.reset()
        labels = []
        for pkey, pvalue in obs.items():
            index = 0
            for skey, svalue in pvalue.items():
                for n,i in enumerate(flatten(svalue)):
                    labels.append("/".join([pkey, skey, str(n)]))
        for n,label in enumerate(labels):
            print("{}: \t {}".format(str(n), label))

    def squeeze_output(self, extract_object):
        """ Call this on the environment to remove some of the outputs to lower the dimension of the observation space for the RL algorithms. To know which substates to keep, use get_state_mapping(). Call this right after initializing the environment.

        :param extract_object: (slice, array) a set of indices which will be extracted from the flattened observation. extract_object can be any type used for indexing i.e. a slice or a numpy.ndarray e.g. ``env.squeeze_output(slice(3,5,1))`` to keep the substates at index 3 and 4 in the flattened game_state.
        """
        self.extract_object = extract_object
        self.observation_space = gym.spaces.Box(low = -100, high = 100, shape = (len(numpy.array(flatten(self.bundle.reset()))[extract_object]),))


    def reset(self, dic = None):
        """ Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(dic)
        return numpy.array(flatten(obs))[self.extract_object]

    def step(self, action):
        """ Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """
        if isinstance(action, numpy.ndarray):
            action = action.tolist()
        if isinstance(self.bundle, PlayBoth):
            action_operator = action[:self.bundle.operator.action_space.shape[0]]
            action_assistant = action[self.bundle.operator.action_space.shape[0]:]
            action = [action_operator, action_assistant]
        obs, sum_reward, is_done, rewards = self.bundle.step(action)
        return numpy.array(flatten(obs))[self.extract_object], sum_reward, is_done, {'rewards':rewards}

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

    def reset(self, dic = None):
        super().reset(dic)

    def step(self, joint_action):
        operator_action, assistant_action = joint_action
        self.game_state["assistant_action"]['action']['values'] = assistant_action
        self.game_state['operator_action']['action']['values'] = operator_action
        self.task.operator_step(operator_action)
        self.task.assistant_step(assistant_action)
