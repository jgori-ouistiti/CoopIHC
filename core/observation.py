from collections import OrderedDict
import numpy
from core.helpers import flatten
import copy
from core.space import State
from core.core import Handbook

from loguru import logger

class BaseObservationEngine:
    """ Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    The only requirement for an Observation Engine is that it has a type (either base, rule, process) and that it has a function called observe with the signature below.

    All Observation Engines are subclassed from this main class, but you are really not inheriting much... This is mostly here for potential future changes.

    :meta public:
    """
    def __init__(self):
        self.type = 'base'
        self.handbook = Handbook({'name': self.__class__.__name__, 'render_mode': [], 'parameters': []})

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        return self.host.inference_engine.buffer[-1]

    @property
    def action(self):
        return self.host.policy.action_state['action']

    @property
    def unwrapped(self):
        return self

    def observe(self, game_state):
        return copy.deepcopy(game_state), 0


    def reset(self):
        pass



# ========================== Some observation engine specifications

oracle_engine_specification =       [('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('user_state', 'all'),
                                    ('assistant_state', 'all'),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

blind_engine_specification =        [('turn_index', 'all'),
                                    ('task_state', None),
                                    ('user_state', None),
                                    ('assistant_state', None),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_task_engine_specification =         [('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('user_state', None),
                                    ('assistant_state', None),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_user_engine_specification  =    [ ('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('user_state', 'all'),
                                    ('assistant_state', None),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_assistant_engine_specification  =   [ ('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('user_state', None),
                                    ('assistant_state', 'all'),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

custom_example_specification =      [('turn_index', 'all'),
                                    ('task_state', 'substate1', 'all'),
                                    ('user_state', 'substate1', slice(0,2,1)),
                                    ('assistant_state', 'None'),
                                    ('user_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

class RuleObservationEngine(BaseObservationEngine):
    """ Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    :meta public:
    """
    def __init__(self, deterministic_specification = base_task_engine_specification, extradeterministicrules = {}, extraprobabilisticrules = {}, mapping = None):
        super().__init__()
        self.type = 'rule'
        self.deterministic_specification = deterministic_specification
        self.extradeterministicrules = extradeterministicrules
        self.extraprobabilisticrules = extraprobabilisticrules
        self.mapping = mapping


        _deterministic_specification = {"name": 'deterministic_specification', "value": str(deterministic_specification), "meaning": 'Indicates which states are observable'}
        _extradeterministicrules =  {"name": 'extradeterministicrules', "value": str(extradeterministicrules), "meaning": 'Extra deterministic rules, applied on a substate basis'}
        _extraprobabilisticrules = {"name": 'extraprobabilisticrules', "value": str(extraprobabilisticrules), "meaning": 'Extra probabilistic rules, applied on a substate basis'}



        self.handbook['parameters'].extend([_deterministic_specification, _extradeterministicrules, _extraprobabilisticrules] )




    def observe(self, game_state):
        if self.mapping is None:
            self.mapping = self.create_mapping(game_state)
        obs = self.apply_mapping(game_state)
        return obs, 0

    def apply_mapping(self, game_state):
        observation = State()
        for substate, subsubstate, _slice, _func, _args, _nfunc, _nargs in self.mapping:
            if observation.get(substate) is None:
                observation[substate] = State()
            _obs = copy.copy(game_state[substate][subsubstate]['values'][_slice])
            if _func:
                if _args:
                    _obs = _func(_obs, game_state, *_args)
                else:
                    _obs = _func(_obs, game_state)
            else:
                _obs = _obs
            if _nfunc:
                if _nargs:
                    _obs, noise = _nfunc(_obs, game_state, *_nargs)
                else:
                    _obs, noise = _nfunc(_obs, game_state)
                logger.info('Noise applied for {}/{}: {}'.format(substate, subsubstate, str(noise)))
            else:
                _obs = _obs


            observation[substate][subsubstate] = copy.copy(game_state[substate][subsubstate])
            observation[substate][subsubstate]['values'] = [_obs]

        return observation

    def create_mapping(self, game_state):
        observation_engine_specification, extradeterministicrules, extraprobabilisticrules = self.deterministic_specification, self.extradeterministicrules, self.extraprobabilisticrules
        mapping = []
        for substate, *rest in observation_engine_specification:
            subsubstate = rest[0]
            if substate == 'turn_index':
                continue
            if subsubstate == 'all':
                for key, value in game_state[substate].items():
                    value = value['values']
                    v = extradeterministicrules.get((substate, key))
                    if v is not None:
                        f,a = v
                    else:
                        f,a = None, None
                    w = extraprobabilisticrules.get((substate, key))
                    if w is not None:
                        g,b = w
                    else:
                        g,b = None, None
                    mapping.append( (substate, key, slice(0, len(value), 1), f, a, g, b) )
            elif subsubstate is None:
                pass
            else:
                # Outdated
                v = extradeterministicrules.get((substate, subsubstate))
                if v is not None:
                    f,a = v
                else:
                    f,a = None, None
                w = extraprobabilisticrules.get((substate, subsubstate))
                if w is not None:
                    g,b= w
                else:
                    g,b = None, None
                _slice = rest[1]
                if _slice == 'all':
                    mapping.append( (substate, subsubstate, slice(0, len(game_state[substate][subsubstate]), 1), f, a , g , b) )
                elif isinstance(_slice, slice):
                    mapping.append( (substate, subsubstate, _slice, f, a, g, b) )
        return mapping



# ===================== Passing extra rules ==============
# each rule is a dictionnary {key:value} with
#     key = (state, substate)
#     value = (function, args)
#     Be careful: args has to be a tuple, so for a single argument arg, do (arg,)
# An exemple
# obs_matrix = {('task_state', 'x'): (core.observation.f_obs_matrix, (C,))}
# extradeterministicrules = {}
# extradeterministicrules.update(obs_matrix)



# ==================== Deterministic functions
# A linear combination of observation components

def observation_linear_combination(_obs, game_state, C):
    return C @ _obs[0]


# ==================== Noise functions
# Additive Gaussian Noise where D shapes the Noise
def additive_gaussian_noise(_obs, gamestate, D, *args):
    try:
        mu,sigma = args
    except ValueError:
        mu, sigma = numpy.zeros(_obs.shape), numpy.eye( max(_obs.shape) )
    return _obs + D @ numpy.random.multivariate_normal(mu, sigma, size = 1).reshape(-1,1), D @ numpy.random.multivariate_normal(mu, sigma, size = 1).reshape(-1,1)



class CascadedObservationEngine(BaseObservationEngine):
    def __init__(self, engine_list):
        super().__init__()
        self.engine_list = engine_list
        self.type = 'multi'


        _engine_list = {"name": 'engine_list', "value": str(engine_list), "meaning": 'A list of observation engines to be cascaded'}

        self.handbook['parameters'].extend([_engine_list])

    def __content__(self):
        return { self.__class__.__name__:         {"Engine{}".format(ni): i.__content__() for ni, i in enumerate(self.engine_list)} }

    def observe(self, game_state):
        game_state = copy.deepcopy(game_state)
        rewards = 0
        for engine in self.engine_list:
            new_obs, new_reward = engine.observe(game_state)
            game_state.update(new_obs)
            rewards += new_reward

        return game_state, rewards


class WrapAsObservationEngine(BaseObservationEngine):
    def __init__(self, obs_bundle):
        self.type = 'process'
        self.bundle = obs_bundle
        self.bundle.reset()

    def __content__(self):
        return { 'Name': self.__class__.__name__, 'Bundle': self.bundle.__content__() }

    @property
    def unwrapped(self):
        return self.bundle.unwrapped

    @property
    def game_state(self):
        return self.bundle.game_state

    def reset(self, *args, **kwargs):
        return self.bundle.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.bundle.step(*args, **kwargs)

    def observe(self, game_state):
        pass
        # Do something
        # return observation, rewards

    def __str__(self):
        return '{} <[ {} ]>'.format(self.__class__.__name__, self.bundle.__str__())

    def __repr__(self):
        return self.__str__()
