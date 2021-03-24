from abc import ABC, abstractmethod
import numpy

from core.helpers import sort_two_lists

class DiscreteOperatorModel(ABC):
    @abstractmethod
    def __init__(self, actions):
        self.actions = actions
        return

    def forward_summary(self, observation):
        llh = []
        for i in self.actions:
            llh.append(self.compute_likelihood(i, observation))
        return self.actions, llh

    @abstractmethod
    def compute_likelihood(self, action, observation):
        return

    def sample(self, observation):
        actions, llh = self.forward_summary(observation)
        llh, actions = sort_two_lists(llh, actions)
        return actions[-1]


class BinaryOperatorModel(DiscreteOperatorModel):
    # Here the action is expected to be the true value of the action
    def __init__(self, action):
        super().__init__([-action, action])

    # Here the action is expected to be in range (0,N) (You may need to convert it in your likelihood expression)
    def compute_likelihood(self, action, observation):
        # Write down all possible cases (4)
        # (1) Goal to the right, positive action
        if observation['operator_state']['Goal'][0] > observation['task_state']['Position'][0] and action > 0 :
            return .99
        # (2) Goal to the right, negative action
        elif observation['operator_state']['Goal'][0] > observation['task_state']['Position'][0] and action < 0 :
            return .01
        # (3) Goal to the left, positive action
        if observation['operator_state']['Goal'][0] < observation['task_state']['Position'][0] and action > 0 :
            return .01
        # (2) Goal to the left, negative action
        elif observation['operator_state']['Goal'][0] < observation['task_state']['Position'][0] and action < 0 :
            return .99
        elif observation['operator_state']['Goal'][0] == observation['task_state']['Position'][0]:
            return 0
        else:
            print("warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition")
            raise NotImplementedError
