from collections import OrderedDict
import numpy
from core.helpers import flatten
import copy
from core.space import State

class ObservationEngine:
    """ Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    The only requirement for an Observation Engine is that it has a type (either base, rule, process) and that it has a function called observe with the signature below.

    All Observation Engines are subclassed from this main class, but you are really not inheriting much... This is mostly here for potential future changes.

    :meta public:
    """
    def __init__(self):
        self.type = 'base'


    def observe(self, game_state):
        return copy.deepcopy(game_state), 0




# ========================== Some observation engine specifications

oracle_engine_specification =       [('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('operator_state', 'all'),
                                    ('assistant_state', 'all'),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

blind_engine_specification =        [('turn_index', 'all'),
                                    ('task_state', None),
                                    ('operator_state', None),
                                    ('assistant_state', None),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_task_engine_specification =         [('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('operator_state', None),
                                    ('assistant_state', None),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_operator_engine_specification  =    [ ('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('operator_state', 'all'),
                                    ('assistant_state', None),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

base_assistant_engine_specification  =   [ ('turn_index', 'all'),
                                    ('task_state', 'all'),
                                    ('operator_state', None),
                                    ('assistant_state', 'all'),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

custom_example_specification =      [('turn_index', 'all'),
                                    ('task_state', 'substate1', 'all'),
                                    ('operator_state', 'substate1', slice(0,2,1)),
                                    ('assistant_state', 'None'),
                                    ('operator_action', 'all'),
                                    ('assistant_action', 'all')
                                    ]

class RuleObservationEngine(ObservationEngine):
    """ Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    :meta public:
    """
    def __init__(self, deterministic_specification = base_task_engine_specification, extradeterministicrules = {}, extraprobabilisticrules = {}, mapping = None):
        self.type = 'rule'
        self.deterministic_specification = deterministic_specification
        self.extradeterministicrules = extradeterministicrules
        self.extraprobabilisticrules = extraprobabilisticrules
        self.mapping = mapping



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
            _obs = game_state[substate][subsubstate]['values'][_slice]
            if _func:
                if _args is not None:
                    _obs = _func(_obs, *_args)
                else:
                    _obs = _func(_obs)
            else:
                _obs = _obs
            if _nfunc:
                if _nargs is not None:
                    _obs = _nfunc(_obs, *_nargs)
                else:
                    _obs = _nfunc(_obs)
            else:
                _obs = _obs

            observation[substate][subsubstate] = game_state[substate][subsubstate]
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
                print(substate, game_state[substate])
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



# ==================== Determinstic functions
# A linear combination of observation components

def f_obs_matrix(_obs, C):
    list_converted = False
    if isinstance(_obs, (float, int)):
        # This case should never occur
        return NotImplementedError
    elif isinstance(_obs, list):
        list_converted = True
        _obs = numpy.array(_obs)
    if not isinstance(_obs, numpy.ndarray):
        return NotImplementedError
    _obs = C @ _obs
    if list_converted:
        _obs = _obs.tolist()
    return _obs


# ==================== Noise functions
# Additive Gaussian Noise (Wiener process) where C selects substates and D shapes the Noise
def agn(_obs, D, *args):
    list_converted = False
    if isinstance(_obs, (float, int)):
        # This case should never occur
        return NotImplementedError
    elif isinstance(_obs, list):
        list_converted = True
        _obs = numpy.array(_obs)
    if not isinstance(_obs, numpy.ndarray):
        return NotImplementedError
    try:
        mu,sigma = args
    except ValueError:
        mu, sigma = numpy.zeros(_obs.shape), numpy.eye( max(_obs.shape) )
    _obs = _obs + D @ numpy.random.multivariate_normal(mu, sigma, size = 1).reshape(-1,1)
    if list_converted:
        _obs = _obs.tolist()
    return _obs

# agn_mapping = {('task_state', 'x'): (agn, (C, D, numpy.array([0,0]).reshape(-1,), numpy.sqrt(timestep)*numpy.eye(2)))}


# observation_engine_specification = [
#     ('task_state', 'x', slice(0,2,1) ),
#     ('operator_state', 'all'),
#     ('assistant_state', None)   ]
#
# noiserules = {('task_state', 'x'): (awgn, None)}
#
#
# # Create mappings
#
#
#
#
# mapping = [     ('task_state', 'x', slice(0,1,1), func1, args),
#                 ('operator_state', 'y', slice(0,2,1), func2, args)
#                 ]
# # Order mapping according to game_state order
#
#
#
#
#
# # Apply mapping


# class RuleObservationEngine(ObservationEngine):
#     """ A Rule Observation Engine.
#
#     A rule Observation Engine is a deterministic observation engine. The rule specifies which substates are fully visible and which are fully invisible.
#
#     :param rule: (OrderedDict) The observation rule, with (key,value) pairs where key is the label of the substate and the following possible value:
#
#     * if value == 'all', then the whole substate is observed
#     * if value == None, then none of the substate is observed
#     * if type(value) == slice(), then only part of the substate as specified by slice is observed
#
#     .. warning::
#
#         Note to self: verify if using a slice as value still works
#
#     See below for predefined, common observation rules.
#
#     .. code-block:: python
#
#         OracleObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', 'all'), ('assistant_state', 'all') ])
#         BaseBlindRule = OrderedDict([('b_state', 'all'), ('task_state', None), ('operator_state', None), ('assistant_state', None) ])
#         TaskObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', None), ('assistant_state', None) ])
#         BaseOperatorObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', 'all'), ('assistant_state', None) ])
#         BaseAssistantObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('operator_state', None), ('assistant_state', 'all') ])
#         CustomRule = OrderedDict([('b_state', 'all'), ('task_state', slice(1,3, None)), ('operator_state', 'all'), ('assistant_state', None) ])
#
#     :meta public:
#
#     """
#     def __init__(self, rule):
#         super().__init__('rule')
#         self.rule = rule
#
#     def observe(self, game_state):
#         """ Observe the game state.
#
#         :params game_state: (OrderedDict) the game_state to be observed.
#
#         :return: observation (OrderedDict) obtained by applying the rule, reward (float) associated with the observation.
#
#         :meta public:
#         """
#         observation = OrderedDict({})
#         for nk,(k,v) in enumerate(list(self.rule.items())):
#             if v == None:
#                 pass
#             elif v == 'all':
#                 observation[k] = game_state[k]
#             # Not tested
#             elif isinstance(v, slice):
#                 observation[k] = game_state[k][v]
#             else:
#                 raise NotImplementedError
#         return (observation, 0)
#
#
#
# class NoisyRuleObservationEngine(RuleObservationEngine):
#     """ A noisy Rule Observation Engine.
#
#     An observation is obtained by passing through a rule observation engine first, and then additive noise.
#
#     The rule specifies the rule observation engine, while the noiserules specify the additive noise. Several noise rules can be passed at once.
#
#     :param rule: (OrderedDict) see RuleObservationEngine
#     :param noiserules: (list). A list of tuples where each tuple specifies a noiserule. A noiserule is expressed as (substate, subsubstate, index, method)
#
#     Example noiserules:
#
#     .. code-block:: python
#
#         noiserules = [('task_state', 'Targets', 0, method)]
#
#         def method():
#             return numpy.random.multivariate_normal( numpy.zeros(shape = (2,)), numpy.ones(shape = (2,)))
#
#     :meta public:
#     """
#     def __init__(self, rule, noiserules):
#         super(RuleObservationEngine, self).__init__('noisyrule')
#         self.rule = rule
#         self.noiserules = noiserules
#
#
#     ## Maybe clipping should be enforced here, to ensure the noise doesn't make the signal go out of bounds.
#     def observe(self, game_state):
#         """ Observe the game_state.
#
#         Call the super() method to apply the deterministic rule and apply the additive noise specified by the noiserules.
#
#         .. warning::
#
#             Clipping is not enforced, the noise signal can go out of bounds.
#
#         :param game_state: (OrderedDict) the state of the game
#
#         observation (OrderedDict) obtained by applying the rule, reward (float) associated with the observation.
#
#         :meta public:
#         """
#         observation, reward = super().observe( game_state)
#         # Don't forget to use copies here, otherwise the noise will be propagated to the game_state as well
#         obs = copy.deepcopy(observation)
#         for substate, subsubstate, index, method, args in self.noiserules:
#             obs[substate][subsubstate][index] += method(args)
#         return obs, reward


class ProcessObservationEngine(ObservationEngine):
    """ Not impleted yet.
    """
    def __init__(self, bundle):
        super().__init__('process')
        self.bundle = bundle

    def observe(self, game_state):
        raise NotImplementedError








#
# CustomGaussianNoiseRule = ('CustomGaussian', OrderedDict({
#     'b_state': ('all', None),
#     'task_state': ('all', None),
#     'operator_state': ('all', {'mu': [0], 'sigma': [1]}),
#     'assistant_state': ('all', None)
#         }))
#
# # SubstateGaussianNoiseRule = ('SubstateGaussian', OrderedDict([('b_state', ([0], [0])), ('task_state', ([0,0], [1,1])), ('operator_state', ([0], [0])), ('assistant_state', ([0], [1])) ]) )
# GaussianNoiseRule = ('Gaussian', (0,1))
#
# def func(observation, arguments = {}):
#     noise = [ni for ni, i in enumerate(observation)]
#     return noise
# CustomNoiseRule = ('Custom', func, {})
