from abc import ABC, abstractmethod
from collections import OrderedDict
import gym

class InteractionTask(gym.Env):
    def __init__(self):
        self._state = OrderedDict()
        self.bundle = None
        self.round = 0
        self.turn = 0

        # Render stuff
        self.ax = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        if self.bundle:
            self.bundle.game_state['task_state'] = state

    # def step(self, active_agent, action):
    #     self.frames += 1/2
    #     is_done = False
    #     operator_reward, assistant_reward = 0,0
    #     if active_agent == "operator":
    #         if action is None:
    #             action = self.bundle.operator.sample()
    #         self.bundle._modify_bundle_value('operator.action_space', action)
    #         obs, operator_reward, is_done, _ = self.operator_step(action)
    #         self.bundle._modify_bundle_value('game.observation_space', obs)
    #         state = self.bundle.assistant.update_state()
    #         self.bundle._modify_bundle_value('assistant.state_space', state)
    #         return self.observation, operator_reward, is_done, {}
    #     elif active_agent == 'assistant':
    #         if action is None:
    #             action = self.bundle.assistant.sample()
    #         self.bundle._modify_bundle_value('assistant.action_space', action)
    #         obs, assistant_reward, is_done, _ = self.assistant_step(action)
    #         self.bundle._modify_bundle_value('game.observation_space', obs)
    #         state = self.bundle.operator.update_state()
    #         self.bundle._modify_bundle_value('operator.state_space', state)
    #         return obs, assistant_reward, is_done, {}



    @abstractmethod
    def operator_step(self, operator_action):
        # return state, reward, is_done, {}
        pass

    @abstractmethod
    def assistant_step(self, operator_action, assistant_action):
        # return state, reward, is_done, {}
        pass

    def render(self, mode, *args, **kwargs):
        if 'text' in mode:
            print(self.state)
        else:
            pass
