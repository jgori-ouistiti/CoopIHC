import gym
import numpy
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt

import core
from core.space import State, StateElement, Space
from core.helpers import flatten, hard_flatten
from core.agents import DummyAssistant, DummyUser
from core.observation import BaseObservationEngine
from core.core import Handbook
from core.interactiontask import PipeTaskWrapper

import copy

import sys
import yaml
import time
import json


######### List of kwargs for bundles init()
#    - reset_skip_first_half_step (if True, skips the first_half_step of the bundle on reset. The idea being that the internal state of the agent provided during initialization should not be updated during reset). To generate a consistent observation, what we do is run the observation engine, but without potential noisefactors.







class Bundle:
    """A bundle combines a task with an user and an assistant. All bundles are obtained by subclassing this main Bundle class.

    A bundle will create the ``game_state`` by combining three states of the task, the user and the assistant as well as the turn index. It also takes care of adding the assistant action substate to the user state and vice-versa.
    It also takes care of rendering each of the three component in a single place.

    Bundle subclasses should only have to redefine the step() and reset() methods.


    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, assistant,**kwargs):
        self.kwargs = kwargs
        self.task = task
        self.task.bundle = self
        self.user = user
        self.user.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self

        # Form complete game state
        self.game_state = State()

        turn_index = StateElement(
            values = [0],
            spaces = Space([numpy.array([0,1])], dtype = numpy.int8)
            )


        self.game_state['turn_index'] = turn_index
        self.game_state["task_state"] = task.state
        self.game_state["user_state"] = user.state
        self.game_state["assistant_state"] = assistant.state

        if user.policy is not None:
            self.game_state["user_action"] = user.policy.action_state
        else:
            self.game_state["user_action"] = State()
            self.game_state["user_action"]['action'] = StateElement()
        if assistant.policy is not None:
            self.game_state["assistant_action"] =  assistant.policy.action_state
        else:
            self.game_state["assistant_action"] = State()
            self.game_state["assistant_action"]['action'] = StateElement()


        self.task.finit()
        self.user.finit()
        self.assistant.finit()



        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211,223,224]
        self.rendered_mode = None
        self.render_perm  = False
        self.playspeed = 0.1




    def __repr__(self):
        return "{}\n".format(self.__class__.__name__) + yaml.safe_dump(self.__content__())


    def __content__(self):
        return {"Task": self.task.__content__(), "User": self.user.__content__(), "Assistant": self.assistant.__content__()}




    def reset(self, task = True, user = True, assistant = True, dic = {}):
        """ Reset the bundle. When subclassing Bundle, make sure to call super().reset() in the new reset method

        :param dic: (dictionnary) Reset the bundle with a game_state

        :return: (list) Flattened game_state

        :meta private:
        """
        if task:
            task_dic = dic.get('task_state')
            task_state = self.task.base_reset(dic = task_dic)

        if user:
            user_dic = dic.get('user_state')
            user_state = self.user.reset(dic = user_dic)

        if assistant:
            assistant_dic = dic.get("assistant_state")
            assistant_state = self.assistant.reset(dic = assistant_dic)


        return self.game_state


    def step(self, action):
        """ Define what happens with the bundle when applying a joint action. Should be redefined when subclassing bundle.

        :param action: (list) joint action.

        :return: observation, sum_rewards, is_done, rewards

        :meta public:
        """
        return self.game_state, 0, False, [0]

    def render(self, mode, *args, **kwargs):
        """ Combines all render methods.

        :param mode: (str) text or plot

        :meta public:
        """
        self.rendered_mode = mode
        if 'text' in mode:
            print('Task Render')
            self.task.render(mode='text', *args , **kwargs)
            print("User Render")
            self.user.render(mode='text', *args , **kwargs)
            print('Assistant Render')
            self.assistant.render(mode = 'text', *args , **kwargs)
        if 'log' in mode:
            self.task.render(mode='log', *args , **kwargs)
            self.user.render(mode='log', *args , **kwargs)
            self.assistant.render(mode = 'log', *args , **kwargs)
        if 'plot' in mode:
            if self.active_render_figure:
                plt.pause(self.playspeed)
                self.task.render(self.axtask, self.axuser, self.axassistant, mode = mode, *args , **kwargs)
                self.user.render(self.axtask, self.axuser, self.axassistant, mode = 'plot', *args , **kwargs)
                self.assistant.render(self.axtask, self.axuser, self.axassistant, mode = 'plot', *args , **kwargs)
                self.fig.canvas.draw()
            else:
                self.active_render_figure = True
                self.fig = plt.figure()
                self.axtask = self.fig.add_subplot(self.figure_layout[0])
                self.axtask.set_title('Task State')
                self.axuser = self.fig.add_subplot(self.figure_layout[1])
                self.axuser.set_title('User State')
                self.axassistant = self.fig.add_subplot(self.figure_layout[2])
                self.axassistant.set_title('Assistant State')
                self.task.render(self.axtask, self.axuser, self.axassistant, mode = 'plot', *args , **kwargs)
                self.user.render(self.axtask, self.axuser, self.axassistant, *args ,  mode = 'plot', **kwargs)
                self.assistant.render(self.axtask, self.axuser, self.axassistant, *args ,  mode = 'plot', **kwargs)
                self.fig.show()

            plt.tight_layout()

        if not ('plot' in mode or 'text' in mode):
            self.task.render(None, mode = mode, *args, **kwargs)
            self.user.render(None, mode = mode, *args, **kwargs)
            self.assistant.render(None, mode = mode, *args, **kwargs)

    def close(self):
        """ Close bundle. Call this after the bundle returns is_done True.

        :meta public:
        """
        if self.active_render_figure:
            plt.close(self.fig)
            self.active_render_figure = None


    def _user_first_half_step(self):
        """ This is the first half of the user step, where the operaror observes the game state and updates its state via inference.

        :return: user_obs_reward, user_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """

        if not self.kwargs.get('onreset_deterministic_first_half_step'):
            user_obs_reward, user_infer_reward = self.user.agent_step()

        else:
            # Store the probabilistic rules
            store = self.user.observation_engine.extraprobabilisticrules
            # Remove the probabilistic rules
            self.user.observation_engine.extraprobabilisticrules = {}
            # Generate an observation without generating an inference
            user_obs_reward, user_infer_reward = self.user.agent_step(infer = False)
            # Reposition the probabilistic rules, and reset mapping
            self.user.observation_engine.extraprobabilisticrules = store
            self.user.observation_engine.mapping = None


        self.kwargs['onreset_deterministic_first_half_step'] = False

        return user_obs_reward, user_infer_reward


    def _user_second_half_step(self, user_action):
        """ This is the second half of the user step. The operaror takes an action, which is applied to the task leading to a new game state.

        :param user_action: (list) user action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """

        # Play user's turn in the task
        task_state, task_reward, is_done, _ = self.task.base_user_step(user_action)

        # update task state (likely not needed, remove ?)
        self.broadcast_state('user', 'task_state', task_state)


        return task_reward, is_done

    def _assistant_first_half_step(self):
        """ This is the first half of the assistant step, where the assistant observes the game state and updates its state via inference.

        :return: assistant_obs_reward, assistant_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """

        assistant_obs_reward, assistant_infer_reward = self.assistant.agent_step()


        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        """ This is the second half of the assistant step. The assistant takes an action, which is applied to the task leading to a new game state.

        :param assistant_action: (list) assistant action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """
        # update action_state

        # Play assistant's turn in the task

        task_state, task_reward, is_done, _ = self.task.base_assistant_step(assistant_action)
        # update task state
        self.broadcast_state('assistant', 'task_state', task_state)

        return task_reward, is_done

    def _user_step(self, *args):
        """ Combines the first and second half step of the user.

        :param args: (None or list) either provide the user action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: user_obs_reward, user_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
        """
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        try:
            # If human input is provided
            user_action = args[0]
        except IndexError:
            # else sample from policy
            user_action, user_policy_reward = self.user.take_action()

        self.broadcast_action('user', user_action)

        task_reward, is_done = self._user_second_half_step(user_action)

        return user_obs_reward, user_infer_reward, user_policy_reward, task_reward, is_done



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
        except IndexError:
            # else sample from policy
            assistant_action, assistant_policy_reward = self.assistant.take_action()

        self.broadcast_action('assistant', assistant_action)

        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, task_reward, is_done


    def broadcast_state(self, role, state_key, state):
        self.game_state[state_key] = state
        getattr(self, role).observation[state_key] = state



    def broadcast_action(self, role, action, key = None):
        # update game state and observations
        if key is None:
            getattr(self, role).policy.action_state['action'] = action
            getattr(self, role).observation['{}_action'.format(role)]["action"] = action
        else:
            getattr(self, role).policy.action_state['action'][key] = action
            getattr(self, role).observation['{}_action'.format(role)]["action"][key] = action





class PlayNone(Bundle):
    """ A bundle which samples actions directly from users and assistants. It is used to evaluate an user and an assistant where the policies are already implemented.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_state = None

    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic = dic, **kwargs)


    def step(self):
        """ Play a step, actions are obtained by sampling the agent's policies.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(None)

        user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, is_done = self._user_step()
        if is_done:
            return self.game_state, sum ([user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward]), is_done, [user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward]
        assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward,  is_done = self._assistant_step()
        return self.game_state, sum([user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]), is_done, [user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]

class PlayUser(Bundle):
    """ A bundle which samples assistant actions directly from the assistant but uses user actions provided externally in the step() method.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_space = copy.copy(self.user.policy.action_state['action']['spaces'])


    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic = dic, **kwargs)
        self._user_first_half_step()
        return self.user.observation
        # return self.user.inference_engine.buffer[-1]

    def step(self, user_action):
        """ Play a step, assistant actions are obtained by sampling the agent's policy and user actions are given externally in the step() method.

        :param user_action: (list) user action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(user_action)

        self.broadcast_action('user', user_action, key = 'values')

        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return self.user.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward,  assistant_policy_reward, second_task_reward, is_done = self._assistant_step()
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return self.user.inference_engine.buffer[-1], sum([user_obs_reward, user_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]), is_done, [user_obs_reward, user_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward]





class PlayAssistant(Bundle):
    """ A bundle which samples oeprator actions directly from the user but uses assistant actions provided externally in the step() method.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_space = self.assistant.policy.action_state['action']['spaces']

        # assistant.policy.action_state['action'] = StateElement(
        #     values = None,
        #     spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(assistant.policy.action_state['action']))],
        #     possible_values = None
        #      )

    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle. A first  user step and assistant observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic = dic, **kwargs)
        self._user_step()
        self._assistant_first_half_step()
        return self.assistant.inference_engine.buffer[-1]

    def step(self, assistant_action):
        """ Play a step, user actions are obtained by sampling the agent's policy and assistant actions are given externally in the step() method.

        :param assistant_action: (list) assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(assistant_action)

        self.broadcast_action('assistant', assistant_action, key = 'values')
        second_task_reward, is_done = self._assistant_second_half_step(assistant_action)
        if is_done:
            return self.assistant.inference_engine.buffer[-1], second_task_reward, is_done, [second_task_reward]
        user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, is_done = self._user_step()
        assistant_obs_reward, assistant_infer_reward = self._assistant_first_half_step()
        return self.assistant.inference_engine.buffer[-1], sum([user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [user_obs_reward, user_infer_reward, user_policy_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]

class PlayBoth(Bundle):
    """ A bundle which samples both actions directly from the user and assistant.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_space = self._tuple_to_flat_space(gym.spaces.Tuple([self.user.action_space, self.assistant.action_space]))

    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle. User observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic = dic, **kwargs)
        self._user_first_half_step()
        return self.task.state

    def step(self, joint_action):
        """ Play a step, user and assistant actions are given externally in the step() method.

        :param joint_action: (list) joint user assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(joint_action)

        user_action, assistant_action = joint_action
        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return self.task.state, first_task_reward, is_done, [first_task_reward]
        assistant_obs_reward, assistant_infer_reward, assistant_policy_reward, second_task_reward,  is_done = self._assistant_step()
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return self.task.state, sum([user_obs_reward, user_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]), is_done, [user_obs_reward, user_infer_reward, first_task_reward, assistant_obs_reward, assistant_infer_reward, second_task_reward]


class SinglePlayUser(Bundle):
    """ A bundle without assistant. This is used e.g. to model psychophysical tasks such as perception, where there is no real interaction loop with a computing device.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent

    :meta public:
    """
    def __init__(self, task, user, **kwargs):
        super().__init__(task, user, DummyAssistant(), **kwargs)


    @property
    def observation(self):
        return self.user.observation

    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic = dic, **kwargs)
        self._user_first_half_step()
        return self.observation

    def step(self, user_action):
        """ Play a step, user actions are given externally in the step() method.

        :param user_action: (list) user action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(user_action)

        self.broadcast_action('user', user_action)
        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return self.user.inference_engine.buffer[-1], first_task_reward, is_done, [first_task_reward]
        self.task.base_assistant_step([None])
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return self.user.inference_engine.buffer[-1], sum([user_obs_reward, user_infer_reward, first_task_reward]), is_done, [user_obs_reward, user_infer_reward, first_task_reward]


class SinglePlayUserAuto(Bundle):
    """ Same as SinglePlayUser, but this time the user action is obtained by sampling the user policy.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param kwargs: additional controls to account for some specific subcases. See Doc for a full list.

    :meta public:
    """
    def __init__(self, task, user, **kwargs):
        super().__init__(task, user, DummyAssistant(), **kwargs)
        self.action_space = None
        self.kwargs = kwargs


    @property
    def observation(self):
        return self.user.observation

    def reset(self, dic = {}, **kwargs):
        """ Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        super().reset(dic = dic, **kwargs)

        if self.kwargs.get('start_at_action'):
            self._user_first_half_step()
            return self.observation

        return self.game_state
        # Return observation


    def step(self):
        """ Play a step, user actions are obtained by sampling the agent's policy.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        if not self.kwargs.get('start_at_action'):
            user_obs_reward, user_infer_reward = self._user_first_half_step()
        user_action, user_policy_reward = self.user.take_action()
        self.broadcast_action('user', user_action)

        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return self.observation, first_task_reward, is_done, [first_task_reward]
        _,_,_,_ = self.task.base_assistant_step([0])
        if self.kwargs.get('start_at_action'):
            user_obs_reward, user_infer_reward = self._user_first_half_step()
        return self.observation, sum([user_obs_reward, user_infer_reward, first_task_reward]), is_done, [user_obs_reward, user_infer_reward, first_task_reward]


## Wrappers
# ====================


class BundleWrapper(Bundle):
    def __init__(self, bundle):
        self.__class__ = type(bundle.__class__.__name__, (self.__class__, bundle.__class__), {})
        self.__dict__ = bundle.__dict__


class PipedTaskBundleWrapper(Bundle):
    # Wrap it by taking over bundles attribute via the instance __dict__. Methods can not be taken like that since they belong to the class __dict__ and have to be called via self.bundle.method()
    def __init__(self, bundle, taskwrapper, pipe):
        self.__dict__ = bundle.__dict__ # take over bundles attributes
        self.bundle = bundle
        self.pipe = pipe
        pipedtask = taskwrapper(bundle.task, pipe)
        self.bundle.task = pipedtask # replace the task with the piped task
        bundle_kwargs = bundle.kwargs
        bundle_class = self.bundle.__class__
        self.bundle = bundle_class(pipedtask, bundle.user, bundle.assistant, **bundle_kwargs)

        self.framerate = 1000
        self.iter = 0

        self.run()

    def run(self, reset_dic = {}, **kwargs):
        reset_kwargs = kwargs.get('reset_kwargs')
        if reset_kwargs is None:
            reset_kwargs = {}
        self.bundle.reset(dic = reset_dic, **reset_kwargs)
        time.sleep(1/self.framerate)
        while True:
            obs, sum_reward, is_done, rewards = self.bundle.step()
            time.sleep(1/self.framerate)
            if is_done:
                break
        self.end()

    def end(self):
        self.pipe.send("done")





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


    def reset(self, dic = {}, **kwargs):
        """ Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state

        :meta public:
        """
        obs = self.bundle.reset(dic = dic, **kwargs)
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
    """ A bundle without user or assistant. It can be used when developping tasks to facilitate some things like rendering
    """
    def __init__(self, task, **kwargs):

        agents = []
        for role in ['user', 'assistant']:
            agent = kwargs.get(role)
            if agent is None:
                agent_kwargs = {}
                agent_policy = kwargs.get("{}_policy".format(role))
                if agent_policy is not None:
                    agent_kwargs['policy'] = agent_policy
                agent = getattr(core.agents, "Dummy"+role.capitalize())(**agent_kwargs)
            else:
                kwargs.pop(agent)

            agents.append(agent)


        super().__init__(task, *agents, **kwargs)

    def reset(self, dic = {}, **kwargs):
        super().reset(dic = dic, **kwargs)

    def step(self, joint_action):
        user_action, assistant_action = joint_action
        if isinstance(user_action, StateElement):
            user_action = user_action['values']
        if isinstance(assistant_action, StateElement):
            assistant_action = assistant_action['values']
        self.game_state["assistant_action"]['action']['values'] = assistant_action
        self.game_state['user_action']['action']['values'] = user_action
        ret_user = self.task.base_user_step(user_action)
        ret_assistant = self.task.base_assistant_step(assistant_action)
        return ret_user, ret_assistant
