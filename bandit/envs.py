from coopihc.space import StateElement, Space
from coopihc.interactiontask import InteractionTask
import numpy as np


class MultiBanditTask(InteractionTask):
    def __init__(self, N=2, P=[0.5, 0.75], T=100, seed=12345):
        super().__init__()

        self.N = N
        self.P = P
        self.T = T

        self.state["last_reward"] = StateElement(
            values=None, spaces=[Space([np.array([0, 1], dtype=int)])]
        )
        self.state["last_action"] = StateElement(
            values=None, spaces=[Space([np.arange(self.N, dtype=int)])]
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def _is_done(self):
        return self.round >= self.T

    def user_step(self, user_action):
        super().user_step(user_action)

        choice = user_action["values"][0][0][0]

        success = self.P[choice] > self.rng.random()
        reward = int(success)

        self.state["last_action"]["values"] = choice
        self.state["last_reward"]["values"] = reward

        return self.state, reward, self._is_done(), {}

    def assistant_step(self, _assistant_action):
        super().assistant_step()

        return self.state, 0, self._is_done(), {}

    def render(self, *_args, mode="text"):
        # super().render()

        if "text" in mode:
            print()
            print(f"Round: {self.round}")
            print(f"Probabilities: {self.P}")
            print()
        else:
            raise NotImplementedError

    def reset(self, dic=None):
        super().reset(dic=dic)

        self.state["last_reward"]["values"] = None
        self.state["last_action"]["values"] = None
        self.rng = np.random.default_rng(seed=self.seed)
