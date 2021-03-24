from collections import OrderedDict
import numpy

# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)

class InferenceEngine:
    def __init__(self, buffer_depth = 0, init_values = 0):
        self.buffer = None
        self.buffer_depth = buffer_depth
        self.init_values = init_values
    def add_observation(self, observation):
        if self.buffer_depth == 0:
            return
        if self.buffer is None:
            self.buffer = self.init_values * numpy.ones(shape = (len(observation), self.buffer_depth))
        # shift data to right and replace the first entry by the new observation
        self.buffer = numpy.roll(self.buffer, 1)
        self.buffer[0,:] = numpy.array(observation)

    def infer(self):
        # do something with information inside buffer
        if self.host.role[0] == 0:
            return self.host.observation['operator_state'], 0
        elif self.host.role[0] == 1:
            return self.host.observation['assistant_state'], 0


# The operatormodel is not updated with this assistant
class GoalInferenceWithOperatorModelGiven(InferenceEngine):
    def __init__(self, operator_model):
        super().__init__()
        self.operator_model = operator_model

    def reset(self):
        self.potential_targets = self.host.bundle.game_state['task_state']['Targets']

    def generate_candidate_operator_observation(self, observation, potential_target):
        observation['operator_state'] = OrderedDict({'Goal': [potential_target]})
        return observation


    def infer(self):
        observation = self.host.observation
        state = observation['assistant_state']
        old_beliefs = state['Beliefs']
        operator_action = state['OperatorAction'][0]

        for nt,t in enumerate(self.potential_targets):
            candidate_observation = self.generate_candidate_operator_observation(observation, t)
            old_beliefs[nt] *= self.operator_model.compute_likelihood(operator_action, candidate_observation)

        if sum(old_beliefs) == 0:
            print("warning: beliefs sum up to 0 after updating. I'm resetting to uniform to continue behavior. You should check if the behavior model makes sense. Here are the latest results from the model")
            old_beliefs = [1 for i in old_beliefs]
        new_beliefs = [i/sum(old_beliefs) for i in old_beliefs]
        state['Beliefs'] = new_beliefs
        return state, 0
