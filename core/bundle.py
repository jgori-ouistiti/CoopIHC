import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt


from core.helpers import flatten

class Bundle(ABC):
    def __init__(self, task, operator, assistant):
        self.task = task
        self.task.bundle = self
        self.operator = operator
        self.operator.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self
        self.game_state = OrderedDict({'b_state': OrderedDict({'next_agent': [0]}), 'task_state': self.task.state, 'operator_state': self.operator.state, 'assistant_state': self.assistant.state})


        self.operator.append_state('AssistantAction', self.assistant.action_space.spaces, possible_values = self.assistant.action_set)
        self.assistant.append_state('OperatorAction', self.operator.action_space.spaces, possible_values = self.operator.action_set)





        # Finish Initializing operator and assistant
        self.operator.finit()
        self.assistant.finit()

        # Make the action space Boxes
        self.operator._original_action_space = self.operator.action_space
        self.operator.action_space = self._tuple_to_flat_space(self.operator.action_space)
        self.assistant._original_action_space = self.assistant.action_space
        self.assistant.action_space = self._tuple_to_flat_space(self.assistant.action_space)

        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211,223,224]
        self.rendered_mode = None
        self.render_perm  = False

    def _tuple_to_flat_space(self, tupled_spaces):
        "Should only be used to wrap Tuple envs."
        low = []
        high = []
        for space in tupled_spaces:
            if isinstance(space, gym.spaces.Box):
                low += space.low.tolist()
                high += space.high.tolist()
            elif isinstance(space, gym.spaces.Discrete): # One hot encoding does not seem to work properly as currently implemented. But should we use onehot encoding at all ?
                low += [0]
                high += [space.n]
                # if space.n <= 2:
                #     low += [0]
                #     high += [1]
                #     self.spacedim += [1]
                #     self.onehot += [False]
                # else:
                #     low = [0 for i in range(space.n)]
                #     high = [1 for i in range(space.n)]
                #     self.spacedim += [space.n]
                #     self.onehot += [True]
            else:
                raise NotImplementedError
        return gym.spaces.Box(numpy.array(low), numpy.array(high))

    def __repr__(self):
        _str = ""
        for o,l in zip(flatten([list(d.values()) for d in list(self.game_state.values())]), self.full_observation_labels):
            if isinstance(o, float):
                _str += "{:>40}  {:<10.3f}\n".format(l, o)
            else:
                _str += "{:>40}  {:<10}\n".format(l, o)
        return _str

    def reset(self, *args):
        if args:
            raise NotImplementedError # Force a state on start
        else:
            self.task.reset()
            self.operator.reset()
            self.assistant.reset()


            self.full_observation, self.full_observation_labels, self.indices, self.game_state_indices = self._flatten_game_state()
            # Broadcast game_state_indices
            self.operator.game_state_indices = self.game_state_indices
            self.assistant.game_state_indices = self.game_state_indices


            return flatten([list(d.values()) for d in list(self.game_state.values())])



    def _flatten_game_state(self):
        obs = flatten([list(d.values()) for d in list(self.game_state.values())])
        labels = []
        indices = []
        upper_indices = []
        for pkey, pvalue in self.game_state.items():
            index = 0
            for skey, svalue in pvalue.items():
                indices.append((pkey, skey, len(flatten(svalue))))
                for n,i in enumerate(flatten(svalue)):
                    labels.append("/".join([pkey, skey, str(n)]))
                    index += 1
            upper_indices.append((pkey, index))
        return obs, labels, indices, upper_indices

    def step(self, action):
        pass

    def render(self, mode, *args, **kwargs):
        self.rendered_mode = mode
        if 'text' in mode:
            print('Task Render')
            self.task.render(None, mode = 'text')
            print("Operator Render")
            self.operator.render(None, mode = 'text')
            print('Assistant Render')
            self.assistant.render(None, mode = 'text')
        if 'plot' in mode:
            if self.active_render_figure:
                plt.pause(.5)
                self.task.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
                self.operator.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
                self.assistant.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
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
                # self.task.render(ax, mode)
                self.task.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
                self.operator.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
                self.assistant.render(self.axtask, self.axoperator, self.axassistant, mode = 'plot')
        if 'permanent' in mode:
            self.render_perm  = True
        if not ('plot' in mode or 'text' in mode):
            self.task.render(*args, **kwargs)
            self.operator.render(*args, **kwargs)
            self.assistant.render(*args, **kwargs)

    def close(self):
        if self.active_render_figure:
            plt.close(self.fig)
            self.active_render_figure = None

    def _operator_first_half_step(self):
        operator_obs_reward, operator_infer_reward = self.operator.agent_step()
        if self.render_perm == True:
            self.render(self.rendered_mode)
        return operator_obs_reward, operator_infer_reward

    def _operator_second_half_step(self, operator_action):
        self.assistant.state['OperatorAction'] = operator_action
        # Play operator's turn in the task
        task_state, task_reward, is_done, _ = self.task.operator_step(operator_action)
        # Broadcast new task state
        self.task.state = task_state
        return task_reward, is_done

    def _assistant_first_half_step(self):
        assistant_obs_reward, assistant_infer_reward = self.assistant.agent_step()
        # assistant takes action
        if self.render_perm == True:
            self.render(self.rendered_mode)
        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        # Broadcast new operator_state_state
        self.operator.state['AssistantAction'] = assistant_action
        # Play assistant's turn in the task
        task_state, task_reward, is_done, _ = self.task.assistant_step(assistant_action)
        # Broadcast new task state
        self.task.state = task_state
        return task_reward, is_done

    def _operator_step(self, *args):
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        try:
            operator_action = args[0]
        except IndexError:
            operator_action = self.operator.sample()
        task_reward, is_done = self._operator_second_half_step(operator_action)
        return operator_obs_reward, operator_infer_reward, task_reward, is_done

    def _assistant_step(self, *args):
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()
        try:
            assistant_action = args[0]
        except IndexError:
            assistant_action = self.assistant.sample()
        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return assistant_obs_reward, assistant_infer_reward, task_reward, is_done

class PlayNone(Bundle):

    def __init__(self, task, operator, assistant):
        super().__init__(task, operator, assistant)
        self.action_space = None


    def reset(self, *args):
        full_obs = super().reset(*args)


    def step(self):
        operator_obs_reward, operator_infer_reward, first_task_reward, is_done = self._operator_step()
        if is_done:
            return operator_obs_reward, operator_infer_reward, task_reward, is_done
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step()
        return sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]

class PlayOperator(Bundle):
    def __init__(self, task, operator, assistant):
        super().__init__(task, operator, assistant)
        self.action_space = self.operator.action_space

    def reset(self, *args):
        full_obs = super().reset(*args)
        self._operator_first_half_step()
        return self.operator.observation

    def step(self, operator_action):
        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.operator.observation, first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step()
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.operator.observation, sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]


class PlayAssistant(Bundle):
    def __init__(self, task, operator, assistant):
        super().__init__(task, operator, assistant)
        self.action_space = self.assistant.action_space

    def reset(self, *args):
        full_obs = super().reset(*args)
        self._operator_step()
        self._assistant_first_half_step()
        return self.assistant.observation

    def step(self, assistant_action):
        second_task_reward, is_done = self._assistant_second_half_step(assistant_action)
        if is_done:
            return self.assistant.observation, second_task_reward, is_done, [second_task_reward]
        operator_obs_reward, operator_infer_reward, first_task_reward, is_done = self._operator_step()
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()
        return self.assistant.observation, sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]

class PlayBoth(Bundle):
    def __init__(self, task, operator, assistant):
        super().__init__(task, operator, assistant)
        self.action_space = self._tuple_to_flat_space(gym.spaces.Tuple([self.operator.action_space, self.assistant.action_space]))

    def reset(self, *args):
        full_obs = super().reset(*args)
        self._operator_first_half_step()
        return self.task.state

    def step(self, joint_action):
        operator_action, assistant_action = joint_action
        first_task_reward, is_done = self._operator_second_half_step(operator_action)
        if is_done:
            return self.task.state, first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, second_task_reward, is_done = self._assistant_step(assistant_action)
        operator_obs_reward, operator_infer_reward = self._operator_first_half_step()
        return self.task.state, sum([operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [operator_obs_reward, operator_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]



class Train(gym.Env):
    def __init__(self, bundle):
        self.bundle = bundle
        self.action_space = bundle.action_space
        observation = flatten(bundle.reset())
        # For now let's assume that the observation space take value in here. This should be normalized anyway.
        self.observation_space = gym.spaces.Box(low = -100, high = 100, shape = (len(observation),))


    def reset(self):
        obs = self.bundle.reset()
        return numpy.array(flatten(obs))

    def step(self, action):
        if isinstance(action, numpy.ndarray):
            action = action.tolist()
        if isinstance(self.bundle, PlayBoth):
            action_operator = action[:self.bundle.operator.action_space.shape[0]]
            action_assistant = action[self.bundle.operator.action_space.shape[0]:]
            action = [action_operator, action_assistant]
        obs, sum_reward, is_done, rewards = self.bundle.step(action)
        return numpy.array(flatten(obs)), sum_reward, is_done, {'rewards':rewards}

    def render(self, mode):
        self.bundle.render(mode)

    def close(self):
        self.bundle.close()
