import gym
from core.policy import BasePolicy
from core.agents import BaseAgent


class DummyAssistant(BaseAgent):

    def __init__(self):
        action_space = [gym.spaces.Discrete(1)]
        action_set = [[0]]
        agent_policy = BasePolicy(
            action_space=action_space, action_set=action_set)

        super().__init__('assistant', policy=agent_policy,
                         observation_engine=None, inference_engine=None)
