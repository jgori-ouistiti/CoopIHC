import numpy


# ========================== Some observation engine specifications

oracle_engine_specification = [
    ("turn_index", "all"),
    ("task_state", "all"),
    ("user_state", "all"),
    ("assistant_state", "all"),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

blind_engine_specification = [
    ("turn_index", "all"),
    ("task_state", None),
    ("user_state", None),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_task_engine_specification = [
    ("turn_index", "all"),
    ("task_state", "all"),
    ("user_state", None),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_user_engine_specification = [
    ("turn_index", "all"),
    ("task_state", "all"),
    ("user_state", "all"),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_assistant_engine_specification = [
    ("turn_index", "all"),
    ("task_state", "all"),
    ("user_state", None),
    ("assistant_state", "all"),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

custom_example_specification = [
    ("turn_index", "all"),
    ("task_state", "substate1", "all"),
    ("user_state", "substate1", slice(0, 2, 1)),
    ("assistant_state", "None"),
    ("user_action", "all"),
    ("assistant_action", "all"),
]


# ===================== Passing extra rules ==============
# each rule is a dictionnary {key:value} with
#     key = (state, substate)
#     value = (function, args)
#     Be careful: args has to be a tuple, so for a single argument arg, do (arg,)
# An exemple
# obs_matrix = {('task_state', 'x'): (coopihc.observation.f_obs_matrix, (C,))}
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
        mu, sigma = args
    except ValueError:
        mu, sigma = numpy.zeros(_obs.shape), numpy.eye(max(_obs.shape))
    return _obs + D @ numpy.random.multivariate_normal(mu, sigma, size=1).reshape(
        -1, 1
    ), D @ numpy.random.multivariate_normal(mu, sigma, size=1).reshape(-1, 1)
