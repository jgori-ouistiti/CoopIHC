from core.agents import GaussianContinuousBeliefOperator
from core.observation import NoisyRuleObservationEngine, BaseOperatorObservationRule

import gym
import numpy



class ChenEye(GaussianContinuousBeliefOperator):
    def __init__(self, swapping_std):
        self.beliefdim = 2
        self.swapping_std = swapping_std
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]

        noiserules = [('task_state', 'Targets', 0, self.eccentric_noise)]
        observation_engine = NoisyRuleObservationEngine(BaseOperatorObservationRule, noiserules)

        super().__init__(self.action_space,  [None, None], observation_engine, dim = self.beliefdim)
        self.append_state('Fixation', [gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)])


        def yms(internal_observation):
            ## specify here what part of the internal observation will be used as observation sample
            return internal_observation['task_state']['Targets'][0]
        self.inference_engine.attach_yms(yms)


    def finit(self):
        pass
    
    def reset(self):
        # Initialize here the start position of the eye
        self.modify_state('Fixation', value = numpy.array([0,0]))
        self.eccentric_noise()
        self.modify_state("MuBelief", value = numpy.array([0 for i in range(self.beliefdim)]))
        #
        Sigma = self.Sigma*numpy.eye(self.beliefdim)
        # Sigma = self.sigma*numpy.ones((self.beliefdim, self.beliefdim))

        # Initialize here the covariance Matrix with Sigma from eccentric noise observation
        self.modify_state("SigmaBelief", value = Sigma)
        self.inference_engine.reset()

    def eccentric_noise(self):
        target = self.bundle.game_state['task_state']['Targets'][0]
        position = self.bundle.game_state['operator_state']['Fixation']
        eccentricity = numpy.sqrt(numpy.sum((target-position)**2))
        sigma = self.swapping_std * eccentricity
        self.Sigma = numpy.array([[sigma, 0], [0, sigma]])
        return numpy.random.normal(0, sigma, 2)
