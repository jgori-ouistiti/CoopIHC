import gym
from core.agents import BaseAgent
from core.observation import RuleObservationEngine, base_operator_engine_specification
from core.policy import ELLDiscretePolicy


class RandomOperator(BaseAgent):

    def __init__(self, N):
        self.N = N

        agent_policy = ELLDiscretePolicy(
            action_space=[gym.spaces.Discrete(N)], action_set=[list(range(N))])

        def user_model(self, _action, _observation):
            return 1/N

        agent_policy.attach_likelihood_function(user_model)

        observation_engine = RuleObservationEngine(
            deterministic_specification=base_operator_engine_specification, extradeterministicrules={}, extraprobabilisticrules={})

        super().__init__('operator', policy=agent_policy,
                         observation_engine=observation_engine)

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)
        print()
        print(f"Action: {self.action.values[0]}")
        print()
