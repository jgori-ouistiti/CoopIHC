import gym
from core.space import StateElement
from core.interactiontask import InteractionTask
import numpy as np


class MultiBanditTask(InteractionTask):

    def __init__(self, N=2, P=[0.5, 0.75], T=100):
        super().__init__()

        self.N = N
        self.P = P
        self.T = T

        self.state["last_reward"] = StateElement(
            values=None, spaces=[gym.spaces.Discrete(2)], possible_values=[[0, 1]])
        self.state["last_action"] = StateElement(
            values=None, spaces=[gym.spaces.Discrete(self.N)], possible_values=[list(range(self.N))])

    def _is_done(self):
        return self.round >= self.T

    def operator_step(self, operator_action):
        self.turn += 1/2
        choice = operator_action['values'][0]

        success = self.P[choice] > np.random.random()
        reward = int(success)

        self.state["last_action"]["values"] = [choice]
        self.state["last_reward"]["values"] = [reward]

        return self.state, reward, self._is_done(), {}

    def assistant_step(self, _assistant_action):
        self.turn += 1/2
        self.round += 1
        return self.state, 0, self._is_done(), {}

    def render(self, *_args, mode="text"):
        if "text" in mode:
            print()
            print(f"Round: {self.round}")
            print(f"Probabilities: {self.P}")
            print()
        else:
            raise NotImplementedError

    def reset(self, dic=None):
        self.state["last_reward"]["values"] = None
        self.state["last_action"]["values"] = None
        super().reset(dic=dic)
