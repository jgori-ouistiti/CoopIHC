from coopihc.space.State import State
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.observation.utils import base_task_engine_specification
import copy


class RuleObservationEngine(BaseObservationEngine):
    """RuleObservationEngine


    An observation engine that is specified by rules regarding each particular substate, using a so called mapping. Example usage is given below:

    .. code-block:: python

        obs_eng = RuleObservationEngine(mapping=mapping)
        obs, reward = obs_eng.observe(game_state=example_game_state())


    A mapping is any iterable where an item is:

    (substate, subsubstate, _slice, _func, _args, _nfunc, _nargs)

    The elements in this mapping are applied to create a particular component of the observation space, as follows

    .. code-block:: python

        observation_component = _nfunc(_func(state[substate][subsubstate][_slice], _args), _nargs)

    For example, a valid mapping for the ``example_game_state`` mapping that states that everything should be observed except the game information is as follows:

    .. code-block:: python

        from coopihc.space.utils import example_game_state
        print(example_game_state())

        # Define mapping
        mapping = [
            ("task_state", "position", slice(0, 1, 1), None, None, None, None),
            ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
            ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
            ("assistant_state", "beliefs", slice(0, 8, 1), None, None, None, None),
            ("user_action", "action", slice(0, 1, 1), None, None, None, None),
            ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
        ]

        # Apply mapping
        obseng = RuleObservationEngine(mapping=mapping)
        obseng.observe(example_game_state())

    As a more complex example, suppose we want to have an observation engine that behaves as above, but which doubles the observation on the ("user_state", "goal") StateElement. We also want to have a noisy observation of the ("task_state", "position") StateElement. We would need the following mapping:

    .. code-block:: python

       def f(observation, gamestate, *args):
            gain = args[0]
            return gain * observation

        def g(observation, gamestate, *args):
            return random.randint(0, 1) + observation

        mapping = [
            ("task_state", "position", slice(0, 1, 1), None, None, g, ()),
            ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
            ("user_state", "goal", slice(0, 1, 1), f, (2,), None, None),
            ("user_action", "action", slice(0, 1, 1), None, None, None, None),
            ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
        ]

    .. note::

        It is important to respect the signature of the functions you pass in the mapping (viz. f and g's signatures).


    Typing out a mapping may be a bit laborious and hard to comprehend for collaborators; there are some shortcuts that make defining this engine easier.

    Example usage:

    .. code-block:: python

        obs_eng = RuleObservationEngine(
            deterministic_specification=engine_specification,
            extradeterministicrules=extradeterministicrules,
            extraprobabilisticrules=extraprobabilisticrules,
        )

    There are three types of rules:

    1. Deterministic rules, which specify at a high level which states are observable or not, e.g.

    .. code-block :: python

        engine_specification = [
            ("game_info", "all"),
            ("task_state", "targets", slice(0, 1, 1)),
            ("user_state", "all"),
            ("assistant_state", None),
            ("user_action", "all"),
            ("assistant_action", "all"),
        ]

    2. Extra deterministic rules, which add some specific rules to specific substates

    .. code-block:: python

        def f(observation, gamestate, *args):
            gain = args[0]
            return gain * observation

        f_rule = {("user_state", "goal"): (f, (2,))}
        extradeterministicrules = {}
        extradeterministicrules.update(f_rule)

    3. Extra probabilistic rules, which are used to e.g. add noise

    .. code-block :: python

        def g(observation, gamestate, *args):
            return random.random() + observation

        g_rule = {("task_state", "position"): (g, ())}
        extraprobabilisticrules = {}
        extraprobabilisticrules.update(g_rule)





    .. warning ::

        This observation engine handles deep copies, to make sure operations based on observations don't mess up the actual states. This might be slow though.


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

    def observe(self, game_state=None):
        """observe

        Wrapper around apply_mapping for interfacing with bundle.

        :param game_state: game state
        :type game_state: :py:class:`State <coopihc.space.State.State>`
        :return: (observation, obs reward)
        :rtype: tuple(:py:class:`State <coopihc.space.State.State>`, float)
        """
        game_state = super().observe(game_state=game_state)[0]

        if self.mapping is None:
            self.mapping = self.create_mapping(game_state)
        obs = self.apply_mapping(game_state)
        return obs, 0

    def apply_mapping(self, game_state):
        """apply_mapping

        Apply the rule mapping

        :param game_state: game state
        :type game_state: :py:class:`State <coopihc.space.State.State>`
        :return: observation
        :rtype: :py:class:`State <coopihc.space.State.State>`
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
                    _obs = _nfunc(_obs, game_state, *_nargs)

                else:
                    _obs = _nfunc(_obs, game_state)

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
        :type game_state: :py:class:`State <coopihc.space.State.State>`
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
            # if substate == "turn_index":
            #     continue
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
                try:
                    _slice = rest[1]
                except IndexError:
                    _slice = "all"

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
