from coopihc.space import State
from .BaseObservationEngine import BaseObservationEngine
from coopihc.observation import base_task_engine_specification
import copy


class RuleObservationEngine(BaseObservationEngine):
    """Base Class for Observation Engine.

    Does nothing but specify a type for the observation engine and return the full game state.

    :meta public:
    """

    def __init__(
        self,
        deterministic_specification=base_task_engine_specification,
        extradeterministicrules={},
        extraprobabilisticrules={},
        mapping=None,
    ):
        super().__init__()
        # self.type = 'rule'
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
        for (
            substate,
            subsubstate,
            _slice,
            _func,
            _args,
            _nfunc,
            _nargs,
        ) in self.mapping:
            if observation.get(substate) is None:
                observation[substate] = State()
            _obs = copy.copy(game_state[substate][subsubstate]["values"][_slice])
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

            else:
                _obs = _obs

            observation[substate][subsubstate] = copy.copy(
                game_state[substate][subsubstate]
            )
            observation[substate][subsubstate]["values"] = [_obs]

        return observation

    def create_mapping(self, game_state):
        (
            observation_engine_specification,
            extradeterministicrules,
            extraprobabilisticrules,
        ) = (
            self.deterministic_specification,
            self.extradeterministicrules,
            self.extraprobabilisticrules,
        )
        mapping = []
        for substate, *rest in observation_engine_specification:
            subsubstate = rest[0]
            if substate == "turn_index":
                continue
            if subsubstate == "all":
                for key, value in game_state[substate].items():
                    value = value["values"]
                    v = extradeterministicrules.get((substate, key))
                    if v is not None:
                        f, a = v
                    else:
                        f, a = None, None
                    w = extraprobabilisticrules.get((substate, key))
                    if w is not None:
                        g, b = w
                    else:
                        g, b = None, None
                    mapping.append((substate, key, slice(0, len(value), 1), f, a, g, b))
            elif subsubstate is None:
                pass
            else:
                # Outdated
                v = extradeterministicrules.get((substate, subsubstate))
                if v is not None:
                    f, a = v
                else:
                    f, a = None, None
                w = extraprobabilisticrules.get((substate, subsubstate))
                if w is not None:
                    g, b = w
                else:
                    g, b = None, None
                _slice = rest[1]
                if _slice == "all":
                    mapping.append(
                        (
                            substate,
                            subsubstate,
                            slice(0, len(game_state[substate][subsubstate]), 1),
                            f,
                            a,
                            g,
                            b,
                        )
                    )
                elif isinstance(_slice, slice):
                    mapping.append((substate, subsubstate, _slice, f, a, g, b))
        return mapping


# ===================== Passing extra rules ==============
# each rule is a dictionnary {key:value} with
#     key = (state, substate)
#     value = (function, args)
#     Be careful: args has to be a tuple, so for a single argument arg, do (arg,)
# An exemple
# obs_matrix = {('task_state', 'x'): (coopihc.observation.f_obs_matrix, (C,))}
# extradeterministicrules = {}
# extradeterministicrules.update(obs_matrix)
