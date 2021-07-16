from core.agents import BaseAgent
from core.observation import RuleObservationEngine, base_operator_engine_specification
from core.policy import LinearFeedback
from core.space import State, StateElement
from core.inference import LinearGaussianContinuous
import eye.noise
from scipy.linalg import toeplitz

import gym
import numpy



class ChenEye(BaseAgent):
    """ The Eye model of Chen, Xiuli, et al. "An Adaptive Model of Gaze-based Selection" Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021., --> with a real 2D implementation.

    :param perceptualnoise: noise in the eye observation process

    :meta public:
    """
    def __init__(self, perceptualnoise, oculomotornoise, dimension = 2, *args, **kwargs):
        self.dimension = dimension
        self.perceptualnoise = perceptualnoise
        self.oculomotornoise = oculomotornoise


        agent_policy = kwargs.get('agent_policy')
        if agent_policy is None:
            action_state = State()
            action_state['action'] = StateElement(
                        values = [None],
                        spaces = [gym.spaces.Box(low=-1, high=1, shape=(self.dimension, ))],
                        possible_values = [None],
                        mode = 'warn'
                                        )

            def noise_function(self, action, observation, *args):
                oculomotornoise = args[0]
                noise_obs = State()
                noise_obs['task_state'] = State()
                noise_obs['task_state']['Targets'] = action
                noise_obs['task_state']['Fixation'] = observation['task_state']['Fixation']
                noise = self.host.eccentric_noise_gen(noise_obs, oculomotornoise)[0]
                return noise

            agent_policy = LinearFeedback(
                ('operator_state','belief'),
                0,
                action_state,
                noise_function = noise_function,
                noise_function_args = (self.oculomotornoise,)
            )



        observation_engine = kwargs.get('observation_engine')

        if observation_engine is None:
            extraprobabilisticrules = {('task_state', 'Targets'): (self._eccentric_noise_gen, ())}

            observation_engine = RuleObservationEngine(base_operator_engine_specification,                                  extraprobabilisticrules = extraprobabilisticrules)


        inference_engine = kwargs.get('inference_engine')
        if inference_engine is None:

            def provide_likelihood_to_inference_engine(self):
                ## specify here what part of the internal observation will be used as observation sample
                observation = self.buffer[-1]
                mu = observation['task_state']['Targets']['values'][0]
                sigma = self.host.Sigma + 1e-4*toeplitz([1] + [0.1 for i in range(self.host.dimension -1)]) # Avoid null sigma
                return mu, sigma

            inference_engine = LinearGaussianContinuous(provide_likelihood_to_inference_engine)



        state = kwargs.get('state')
        if state is None:
            belief = StateElement(
                values = [None, None],
                spaces = [gym.spaces.Box(low=-1, high=1, shape=(self.dimension, )), gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (self.dimension,self.dimension))],
                possible_values = [[None], [None]],
                mode = 'warn'
            )
            state = State()
            state['belief'] = belief

        super().__init__('operator',
                            state = state,
                            policy = agent_policy,
                            observation_engine = observation_engine,
                            inference_engine = inference_engine
                            )


        self.handbook['render_mode'] = ['plot', 'text', 'log']
        _oculomotornoise = {'value': oculomotornoise, 'meaning': 'perceptual noise related to the eccentricity of the target w/r current fixation'}
        _perceptualnoise = {'value': perceptualnoise, 'meaning': 'motor noise related to the distance convered by the fixation'}
        _dimension = {'value': dimension, 'meaning': 'Dimension of the Fixation used.'}
        self.handbook['parameters'] = [_oculomotornoise, _perceptualnoise, _dimension]


    def finit(self):
        pass

    def reset(self, dic = None):
        """ Reset the fixation at the center (0;0), reset the prior belief

        :meta public:
        """
        # call reset before so that the code below gets executed when no arguments are provided to reset (no forced reset)
        if dic is None:
            super().reset()

        # Initialize here the start position of the eye as well as initial uncertainty
        observation = State()
        observation['task_state'] = State()
        observation['task_state']['Targets'] = self.bundle.task.state['Targets']
        observation['task_state']['Fixation'] = self.bundle.task.state['Fixation']
        # Initialize with a huge Gaussian noise so that the first observation massively outweights the prior. Put more weight on the pure variance components to ensure that it will behave well.
        Sigma = toeplitz([1000] + [100 for i in range(self.dimension -1)])
        self.state['belief']['values'] = [numpy.array([0 for i in range(self.dimension)]), Sigma]

        # call reset after, so that the code below gets overwritten by arguments provided in reset (forced reset)
        if dic is not None:
            super().reset(dic = dic)

    def _eccentric_noise_gen(self, _obs, observation, *args):
        # clip the noisy observation --> this makes sense e.g. on a screen, were we know that the observation has to be on screen at all times.
        noise = self.eccentric_noise_gen(observation,   *args)[0]
        return numpy.clip(  _obs[0] + noise,
                        observation['task_state']['Targets']['spaces'][0].low, observation['task_state']['Targets']['spaces'][0].high), noise

    def eccentric_noise_gen(self, observation, *args):
        """ Define eccentric noise that will be used in the noisyrule

        :return: noise samples drawn according to the 2D centered Gaussian with covariance matrix specified by the eccentric noise.

        :meta public:
        """
        try:
            noise_std = args[0]
        except IndexError:
            noise_std = self.perceptualnoise
        target = observation['task_state']['Targets']['values'][0]
        position = observation['task_state']['Fixation']['values'][0]
        Sigma = eye.noise.eccentric_noise(target, position, noise_std)
        noise = numpy.random.multivariate_normal( numpy.zeros(shape = (self.dimension,)), Sigma)
        self.Sigma = Sigma
        return noise, Sigma


    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'
        try:
            axtask, axoperator, axassistant = args
            self.inference_engine.render(axtask, axoperator, axassistant, mode = mode)
        except ValueError:
            self.inference_engine.render(mode = mode)
