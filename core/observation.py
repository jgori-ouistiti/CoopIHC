from collections import OrderedDict
import numpy
from core.helpers import flatten

class ObservationEngine:
    def __init__(self, type):
        if type == 'rule':
            self.type = 'rule'
        elif type == 'noisyrule':
            self.type = 'noisyrule'
        elif type == 'process':
            self.type = process

class RuleObservationEngine(ObservationEngine):
    def __init__(self, rule):
        super().__init__('rule')
        self.rule = rule


    # Observation engines are currently 'conflicted' between observing the flattened vector output state, or observing the OrderedDict of the game_state. I'm not sure what is best here and this might need to be changed at some point.
    # def observe(self, host, game_state):
    #     observation = OrderedDict({})
    #     indices = list(zip(*host.game_state_indices))[1]
    #     obs_indx = OrderedDict({})
    #     index = 0
    #     for nk,(k,v) in enumerate(list(self.rule.items())):
    #         if v == None:
    #             pass
    #         elif v == 'all':
    #             # observation[k] = game_state[slice(index, index + indices[nk], None)]
    #             observation.extend(game_state[slice(index, index + indices[nk], None)])
    #             obs_indx[k] = slice(index, index + indices[nk], None)
    #         elif isinstance(v, slice):
    #             # observation[k] = game_state[slice(index + v.start, index + v.stop, v.step)]
    #             observation.extend(game_state[slice(index + v.start, index + v.stop, v.step)])
    #             obs_indx[k] = slice(index + v.start, index + v.stop, v.step)
    #         else:
    #             raise NotImplementedError
    #         index += indices[nk]
    #     self.obs_indx = obs_indx
    #     return (observation, 0)

    def observe(self, host, game_state):
        observation = OrderedDict({})
        for nk,(k,v) in enumerate(list(self.rule.items())):
            if v == None:
                pass
            elif v == 'all':
                observation[k] = game_state[k]
            elif isinstance(v, slice):
                observation[k] = game_state[k][v]
            else:
                raise NotImplementedError
        return (observation, 0)


class NoisyRuleObservationEngine(RuleObservationEngine):
    def __init__(self, rule, noiserule):
        raise NotImplementedError ### Have to redo this, just like above with OrderedDict observation
        super(RuleObservationEngine, self).__init__('noisyrule')
        self.rule = rule
        self.noiserule = noiserule
    def observe(self, host, game_state):
        observation, reward = super().observe(host, game_state)
        if self.noiserule[0] == 'Gaussian':
            mu, sigma = self.noiserule[1]
            noise = numpy.random.normal(mu, sigma, len(observation))
        elif self.noiserule[0] == 'CustomGaussian':
            indices = list(zip(*host.game_state_indices))[1]
            index = 0
            noise = []
            noiseruledic = self.noiserule[1]
            for nk, (k,v) in enumerate(list(self.rule.items())):
                if v == None:
                    pass
                elif v == 'all':
                    substate_len = len(game_state[slice(index, index + indices[nk], None)])
                elif isinstance(k, slice):
                    substate_len = len(game_state[slice(index + k.start, index + k.stop, k.step)])
                else:
                    raise NotImplementedError
                mu, sigma = noiseruledic[k]
                if len(mu) == substate_len: # If the noiserule specifies mu, sigma for each component
                    for m,s in zip(mu,sigma):
                        noise += [numpy.random.normal(m,s,1)]
                elif len(mu) == 1:
                    noise += [numpy.random.normal(mu, sigma, substate_len)]
                index += indices[nk]
        elif self.noiserule[0] == 'Custom':
            func, kwargs = self.noiserule[1:]
            noise = func(observation, **kwargs)


        obs = [u+v for u,v in zip(observation, flatten(noise))]
        return obs, reward




# Rules
OracleObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', 'all'), ('assistant_state', 'all') ])
BaseBlindRule = OrderedDict([('b_state', 'all'), ('task_state', None), ('operator_state', None), ('assistant_state', None) ])
TaskObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', None), ('assistant_state', None) ])
BaseOperatorObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', 'all'), ('assistant_state', None) ])
BaseAssistantObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', None), ('assistant_state', 'all') ])
CustomRule = OrderedDict([('b_state', 'all'), ('task_state', slice(1,3, None)), ('operator_state', 'all'), ('assistant_state', None) ])

# NoisyRules
CustomGaussianNoiseRule = ('CustomGaussian', OrderedDict([('b_state', ([0], [0])), ('task_state', ([0,0], [1,1])), ('operator_state', ([0], [0])), ('assistant_state', ([0], [1])) ]) )
GaussianNoiseRule = ('Gaussian', (0,1))

def func(observation, arguments = {}):
    noise = [ni for ni, i in enumerate(observation)]
    return noise
CustomNoiseRule = ('Custom', func, {})
