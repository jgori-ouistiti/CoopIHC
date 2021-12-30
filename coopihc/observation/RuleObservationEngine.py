from coopihc.space.State import State
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.observation.utils import base_task_engine_specification
import copy
import numpy


class RuleObservationEngine(BaseObservationEngine):
    """RuleObservationEngine [summary]

    An observation engine that is specified by rules regarding each particular substate, using a so called mapping.

    A mapping is any iterable where an item is:

    (substate, subsubstate, _slice, _func, _args, _nfunc, _nargs)

    where     observation = _nfunc(_func(state[substate][subsubstate][_slice], _args), _nargs)

    Example usage:

    .. code-block:: python

        observation_engine = RuleObservationEngine(
            base_user_engine_specification,
            extraprobabilisticrules=extraprobabilisticrules,
        )

    There are several rules:

    1. Deterministic rules, which specify at a high level which states are observable or not, e.g.

    .. code-block :: python

        base_user_engine_specification = [
                ("turn_index", "all"),
                ("task_state", "all"),
                ("user_state", "all"),
                ("assistant_state", None),
                ("user_action", "all"),
                ("assistant_action", "all"),
            ]

    2. Extra deterministic rules, which add some specific rules

    .. code-block:: python

        obs_matrix = {('task_state', 'x'): (coopihc.observation.f_obs_matrix, (C,))}
        extradeterministicrules = {}
        extradeterministicrules.update(obs_matrix)

    3. Extra probabilistic rules, which are used to e.g. add noise


    .. code-block :: python

        extraprobabilisticrules = {
            ("task_state", "targets"): (self._eccentric_noise_gen, ())
        }



    .. warning ::

        This observation engine is likely very slow, due to may copies.


    :param deterministic_specification: deterministic rules, defaults to base_task_engine_specification
    :type deterministic_specification: list(tuples), optional
    :param extradeterministicrules: extra deterministic rules, defaults to {}
    :type extradeterministicrules: dict, optional
    :param extraprobabilisticrules: extra probablistic rules, defaults to {}
    :type extraprobabilisticrules: dict, optional
    :param mapping: mapping, defaults to None
    :type mapping: iterable, optional
    """

    def __init__(
        self,
        *args,
        deterministic_specification=base_task_engine_specification,
        extradeterministicrules={},
        extraprobabilisticrules={},
        mapping=None,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.deterministic_specification = deterministic_specification
        self.extradeterministicrules = extradeterministicrules
        self.extraprobabilisticrules = extraprobabilisticrules
        self.mapping = mapping

    def observe(self, game_state):
        """observe

        Wrapper around apply_mapping for interfacing with bundle.

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State`
        :return: (observation, obs reward)
        :rtype: tuple(`State<coopihc.space.State.State`, float)
        """
        if self.mapping is None:
            self.mapping = self.create_mapping(game_state)
        obs = self.apply_mapping(game_state)
        return obs, 0

    def apply_mapping(self, game_state):
        """apply_mapping

        Apply the rule mapping

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State`
        :return: observation
        :rtype: `State<coopihc.space.State.State`
        """
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

            _obs = copy.copy((game_state[substate][subsubstate][_slice]))
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
            observation[substate][subsubstate] = _obs

        return observation

    def create_mapping(self, game_state):
        """create_mapping

        Create mapping from the high level rules specified in the Rule Engine.

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State>`
        :return: Mapping
        :rtype: iterable
        """
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
                    # deal with ints
                    try:
                        _len = len(value)
                    except TypeError:
                        _len = 1
                    mapping.append((substate, key, slice(0, _len, 1), f, a, g, b))
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
