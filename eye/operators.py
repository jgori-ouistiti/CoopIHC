from core.agents import GaussianContinuousBeliefOperator
from core.observation import RuleObservationEngine, base_operator_engine_specification

import eye.noise

import gym
import numpy



class ChenEye(GaussianContinuousBeliefOperator):
    """ The Eye model of Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021., --> with a real 2D implementation.

    :param swapping_std: noise in the eye observation process

    :meta public:
    """
    def __init__(self, swapping_std):
        self.beliefdim = 2
        self.swapping_std = swapping_std
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)]

        noiserules = {('task_state', 'Targets'): (self.eccentric_noise_gen, None)}
        observation_engine = RuleObservationEngine(base_operator_engine_specification, noiserules)

        super().__init__(self.action_space,  [None, None], observation_engine, dim = self.beliefdim)
        self.append_state('Fixation', [gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=numpy.float64)])
        


        def yms(internal_observation):
            ## specify here what part of the internal observation will be used as observation sample
            return internal_observation['task_state']['Targets'][0]

        self.inference_engine.attach_yms(yms)


    def finit(self):
        pass

    def reset(self):
        """ Reset the fixation at the center (0;0), reset the prior belief

        :meta public:
        """
        # Initialize here the start position of the eye
        self.modify_state('Fixation', value = numpy.array([0,0]))
        self.eccentric_noise_gen()
        self.modify_state("MuBelief", value = numpy.array([0 for i in range(self.beliefdim)]))
        #
        Sigma = self.Sigma*numpy.eye(self.beliefdim)
        # Sigma = self.sigma*numpy.ones((self.beliefdim, self.beliefdim))

        # Initialize here the covariance Matrix with Sigma from eccentric noise observation
        self.modify_state("SigmaBelief", value = Sigma)
        self.inference_engine.reset()


    def eccentric_noise_gen(self, observation, *args):
        """ Define eccentric noise that will be used in the noisyrule

        .. warning::

            this also sets self.Sigma to pass the covariance matrix used in the likelihood.

        :return: noise samples drawn according to the 2D centered Gaussian with covariance matrix specified by the eccentric noise.

        :meta public:
        """
        target = observation['task_state']['Targets'][0]
        position = observation['operator_state']['Fixation']
        self.Sigma = eye.noise.eccentric_noise(target, position, self.swapping_std)
        return numpy.random.multivariate_normal( numpy.zeros(shape = (2,)), self.Sigma)

    def sample(self):
        """ The policy is to select the most likely position of the target as next fixation.

        :meta public:
        """
        return self.state['MuBelief']
