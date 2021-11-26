from .BaseObservationEngine import BaseObservationEngine
from .RuleObservationEngine import RuleObservationEngine
from .CascadedObservationEngine import CascadedObservationEngine
from .WrapAsObservationEngine import WrapAsObservationEngine

import numpy
from coopihc.space import State, StateElement, Space

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


# ================= Examples ===================
class ExampleObservationEngine(BaseObservationEngine):
    def __init__(self, observable_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable_state = observable_state

    def observe(self, game_state):
        return game_state[self.observable_state], 0


if __name__ == "__main__":
    x = StateElement(
        values=None,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    # Discrete substate. Provide Space([range]). Value is optional
    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], dtype=numpy.int)]))

    # Define a State, composed of two substates previously defined
    s1 = State(substate_x=x, substate_y=y)

    # Define a super-State that is composed of the State previously defined
    S = State()

    a = StateElement(
        values=None, spaces=Space([numpy.array([0, 1, 2], dtype=numpy.int8)])
    )

    s2 = State(substate_a=a)
    S["substate1"] = s1
    S["substate_2"] = s2
    obs_engine = ExampleObservationEngine("substate1")
    print("Game state before observation")
    print(S)
    print("Observation")
    print(obs_engine.observe(S)[0])
