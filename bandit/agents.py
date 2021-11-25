from bandit.inference import RWInference

import numpy as np

from coopihc.space import State, StateElement, Space
from coopihc.agents import BaseAgent
from coopihc.policy import ELLDiscretePolicy


class GenericPlayer(BaseAgent):
    def __init__(self, user_model, seed=None):
        self.user_model = user_model
        self.seed = seed

        super().__init__("user")

    def render(self, *args, **kwargs):
        # super().render(*args, **kwargs)
        print()
        print(f"Action: {self.action.values[0]}")
        print()

    def finit(self):
        self.N = self.bundle.task.N

        action_state = State()
        action_state["action"] = StateElement(
            values=None, spaces=[Space([np.arange(self.N, dtype=np.int16)])]
        )
        agent_policy = ELLDiscretePolicy(action_state=action_state, seed=self.seed)

        agent_policy.attach_likelihood_function(self.user_model)

        self.attach_policy(agent_policy)

    def reset(self, dic=None):
        self.finit()


class RandomPlayer(GenericPlayer):
    def __init__(self, seed=None):
        def user_model(_self, _choice, _observation):
            return 1 / self.N

        super().__init__(user_model=user_model, seed=seed)


class WSLS(GenericPlayer):
    def __init__(self, epsilon, seed=None):
        self.epsilon = epsilon

        def user_model(_self, action, observation):
            choice = action.values[0][0][0]
            last_choice = self.action["values"][0]

            if last_choice is None:
                return 1 / self.N

            last_reward = observation["task_state"]["last_reward"]["values"][0]

            p_apply_rule = 1 - self.epsilon
            p_random = self.epsilon / self.N

            p_stay = p_apply_rule + p_random
            p_switch = p_apply_rule / (self.N - 1) + p_random

            # If last choice resulted in a success...
            if last_reward:
                # The probability of switching...
                if choice != last_choice:
                    # ...is determined by epsilon (randomness)
                    return p_random
                # The probability of staying is determined by 1 - epsilon
                return p_stay
            # If the last choice resulted in failure...
            # The probability of switching...
            if choice != last_choice:
                # ...is determined by 1 - epsilon
                return p_switch
            # The probability of staying is determined by epsilon (randomess)
            return p_random

        super().__init__(user_model=user_model, seed=seed)


class RW(GenericPlayer):
    def __init__(self, q_alpha, q_beta, initial_value=0.5, seed=None):
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.initial_value = initial_value

        def user_model(_self, action, observation):
            choice = action.values[0][0][0]
            q_values = self.state["q_values"]["values"][0][0]

            num = np.exp(self.q_beta * q_values[choice])
            denom = np.sum(np.exp(self.q_beta * q_values))

            return num / denom

        super().__init__(user_model=user_model, seed=seed)

    def finit(self):
        super().finit()

        self.state["q_values"] = StateElement(
            values=np.full(self.N, self.initial_value),
            spaces=[Space([np.zeros(self.N), np.ones(self.N)])],
        )

        self.attach_inference_engine(RWInference())

    def reset(self, dic=None):
        super().reset()
        self.finit()
        self.state["q_values"] = StateElement(
            values=np.full(self.N, self.initial_value),
            spaces=[Space([np.zeros(self.N), np.ones(self.N)])],
        )
