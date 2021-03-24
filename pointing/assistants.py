from core.agents import BaseAgent, BayesianBeliefAssistant
from core.helpers import sort_two_lists

import gym
import numpy
from collections import OrderedDict
import math

class ConstantCDGain(BaseAgent):
    def __init__(self, gain):
        self.gain = gain
        super().__init__(   "assistant",
                            [gym.spaces.Box(low = numpy.array([-10]), high = numpy.array([10]))],
                            [None]
                            )

    def sample(self):
        return [self.gain]

    def finit(self):
        return

    def reset(self):
        return


class BIGGain(BayesianBeliefAssistant):
    def __init__(self, operator_model, threshold = 0.5):
        # If one posterior probability goes above threshold, then the gain will be computed so as to reach it immediately.

        super().__init__(   [gym.spaces.Box(low = numpy.array([-10]), high = numpy.array([10]))],
                            [None],
                            operator_model,
                            observation_engine = None)
        self.operator_model = operator_model
        self.threshold = threshold

    def PYy_Xx(self, operator_action, position, targets, beliefs):
        pYy__Xx = 0
        for t,b in zip(targets, beliefs):
            observation = self.generate_candidate_observation(t, position)
            try:
                pYy__Xx += self.operator_model.compute_likelihood(operator_action, observation)*b
            except TypeError:
                print(operator_action, observation, b)
        return pYy__Xx


    def HY__Xx(self, position, targets, beliefs):
        H = 0
        for operator_action in self.operator_model.actions:
            pYy_Xx = self.PYy_Xx(operator_action, position, targets, beliefs)
            if pYy_Xx != 0:
                H += -pYy_Xx * math.log(pYy_Xx,2)
        return H

    def HY__OoXx(self, position, targets, beliefs):
        H = 0
        for operator_action in self.operator_model.actions:
            for t,b in zip(targets, beliefs):
                observation = self.generate_candidate_observation(t, position)
                pYy__OoXx = self.operator_model.compute_likelihood(operator_action, observation)
                if pYy__OoXx != 0:
                    H += -b*pYy__OoXx*math.log(pYy__OoXx,2)
        return H

    def IG(self, position, targets, beliefs):
        return self.HY__Xx(position, targets, beliefs) - self.HY__OoXx(position, targets, beliefs)

    def find_best_pos(self):
        targets = self.targets
        beliefs = self.state['Beliefs']
        hp, hp_target = max(beliefs), targets[beliefs.index(max(beliefs))]
        if hp > self.threshold:
            return [hp_target], [None]
        else:
            IG_storage = [self.IG( pos, targets, beliefs) for pos in range(self.bundle.task.state['Gridsize'][0])]
            _IG, pos = sort_two_lists(IG_storage, list(range(self.bundle.task.state['Gridsize'][0])))
            pos.reverse(), _IG.reverse()
            return pos, _IG


    def sample_discrete(self):
        new_pos, probabilities = self.find_best_pos()
        return new_pos[0]


    def generate_candidate_observation(self, target, position):
        observation = OrderedDict({})
        observation['operator_state'] = OrderedDict({'Goal': [target]})
        observation['task_state'] = OrderedDict({'Position': [position]})
        return observation



    def sample(self, mode = 'gain'):
        ret = self.sample_discrete()
        if mode == 'gain':
            operator_action = self.state['OperatorAction'][0]
            position = self.bundle.task.state['Position'][0]
            ret =  (ret - position)/operator_action
        else:
            pass
        return [ret]
