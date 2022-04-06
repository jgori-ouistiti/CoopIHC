"""This module provides tests for the BaseObservationEngine class of the
coopihc package."""

from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.base.elements import example_game_state
import random
import pytest


def test_imports():
    """Tests the different import ways for the  obseng."""
    from coopihc import RuleObservationEngine
    from coopihc.observation import RuleObservationEngine
    from coopihc.observation.RuleObservationEngine import RuleObservationEngine


def test_init():
    """Tries to initialize an  obseng and checks the expected
    properties and methods."""
    assert empty_init()
    test_properties()
    test_methods()


def test_properties():
    """Tests the expected properties for a minimal  obseng."""
    obseng = RuleObservationEngine()
    # Attributes
    assert hasattr(obseng, "deterministic_specification")
    assert hasattr(obseng, "extradeterministicrules")
    assert hasattr(obseng, "extraprobabilisticrules")
    assert hasattr(obseng, "mapping")
    # Property functions
    # assert hasattr(obseng, "turn_number") # Only available with bundle
    assert hasattr(obseng, "observation")
    assert hasattr(obseng, "action")


def test_methods():
    """Tests the expected methods for a minimal  obseng."""
    obseng = RuleObservationEngine()
    # Public methods
    assert hasattr(obseng, "observe")
    assert hasattr(obseng, "reset")
    assert hasattr(obseng, "apply_mapping")
    assert hasattr(obseng, "create_mapping")
    # Private methods
    assert hasattr(obseng, "__content__")


def empty_init():
    """Returns True if trying to initialize an  obseng
    without any arguments fails."""
    try:
        RuleObservationEngine()
        return True
    except TypeError:
        return False


def test_reset():
    obseng = RuleObservationEngine()
    obseng.reset()
    return True


def test_observationengine():
    """Tests the methods provided by the  obseng class."""
    test_imports()
    test_init()
    test_reset()


def test_create_deterministic_all_mapping():

    engine_specification = [
        ("game_info", "all"),
        ("task_state", "all"),
        ("user_state", "all"),
        ("assistant_state", "all"),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]

    obs_eng = RuleObservationEngine(deterministic_specification=engine_specification)
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("assistant_state", "beliefs", slice(0, 8, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


# ============================= Create Mappings ======================
def test_create_deterministic_all_mapping_remove_one():

    engine_specification = [
        ("game_info", "all"),
        ("task_state", "all"),
        ("user_state", "all"),
        ("assistant_state", None),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]

    obs_eng = RuleObservationEngine(deterministic_specification=engine_specification)
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


def test_create_deterministic_all_mapping_remove_subsubstate():
    engine_specification = [
        ("game_info", "all"),
        ("task_state", "targets"),
        ("user_state", "all"),
        ("assistant_state", None),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]
    obs_eng = RuleObservationEngine(deterministic_specification=engine_specification)
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


def test_create_deterministic_all_mapping_remove_subsubstate_slice():
    engine_specification = [
        ("game_info", "all"),
        ("task_state", "targets", slice(0, 1, 1)),
        ("user_state", "all"),
        ("assistant_state", None),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]
    obs_eng = RuleObservationEngine(deterministic_specification=engine_specification)
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 1, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


def test_create_deterministic_all_mapping_extradeterministic():
    engine_specification = [
        ("game_info", "all"),
        ("task_state", "all"),
        ("user_state", "all"),
        ("assistant_state", None),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]

    def f(observation, gamestate, *args):
        gain = args[0]
        return gain * observation

    f_rule = {("user_state", "goal"): (f, (2,))}
    extradeterministicrules = {}
    extradeterministicrules.update(f_rule)

    obs_eng = RuleObservationEngine(
        deterministic_specification=engine_specification,
        extradeterministicrules=extradeterministicrules,
    )
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), f, (2,), None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


def test_create_deterministic_all_mapping_extradeterministic_extraprobabilistic():
    engine_specification = [
        ("game_info", "all"),
        ("task_state", "all"),
        ("user_state", "all"),
        ("assistant_state", None),
        ("user_action", "all"),
        ("assistant_action", "all"),
    ]

    def f(observation, gamestate, *args):
        gain = args[0]
        return gain * observation

    f_rule = {("user_state", "goal"): (f, (2,))}
    extradeterministicrules = {}
    extradeterministicrules.update(f_rule)

    def g(observation, gamestate, *args):
        return random.random() + observation

    g_rule = {("task_state", "position"): (g, ())}
    extraprobabilisticrules = {}
    extraprobabilisticrules.update(g_rule)

    obs_eng = RuleObservationEngine(
        deterministic_specification=engine_specification,
        extradeterministicrules=extradeterministicrules,
        extraprobabilisticrules=extraprobabilisticrules,
    )
    mapping = obs_eng.create_mapping(example_game_state())
    assert mapping == [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "position", slice(0, 1, 1), None, None, g, ()),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), f, (2,), None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]


def test_create_mapping():
    test_create_deterministic_all_mapping()
    test_create_deterministic_all_mapping_remove_one()
    test_create_deterministic_all_mapping_remove_subsubstate()
    test_create_deterministic_all_mapping_remove_subsubstate_slice()
    test_create_deterministic_all_mapping_extradeterministic()
    test_create_deterministic_all_mapping_extradeterministic_extraprobabilistic()


# ====================== Apply mappings =======================


def test_apply_deterministic_all_mapping():
    mapping = [
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("assistant_state", "beliefs", slice(0, 8, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]
    obseng = RuleObservationEngine(mapping=mapping)
    obs, reward = obseng.observe(game_state=example_game_state())
    # print(dict.__repr__(example_game_state()))
    # print(example_game_state())
    egs = example_game_state()
    del egs["game_info"]
    assert egs == obs


def test_apply_deterministic_all_mapping_remove_one():
    mapping = [
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]
    obseng = RuleObservationEngine(mapping=mapping)
    obs, reward = obseng.observe(game_state=example_game_state())
    # print(dict.__repr__(example_game_state()))
    # print(example_game_state())
    egs = example_game_state()
    del egs["game_info"]
    del egs["assistant_state"]
    assert egs == obs


def test_apply_deterministic_all_mapping_remove_subsubstate():
    mapping = [
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]
    obseng = RuleObservationEngine(mapping=mapping)
    obs, reward = obseng.observe(game_state=example_game_state())
    # print(dict.__repr__(example_game_state()))
    # print(example_game_state())
    egs = example_game_state()
    del egs["game_info"]
    del egs["assistant_state"]
    del egs["task_state"]["position"]
    assert egs == obs


def test_apply_deterministic_all_mapping_remove_subsubstate_slice():
    mapping = [
        ("task_state", "targets", slice(0, 1, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]
    obseng = RuleObservationEngine(mapping=mapping)
    obs, reward = obseng.observe(game_state=example_game_state())
    # print(dict.__repr__(example_game_state()))
    # print(example_game_state())
    egs = example_game_state()
    del egs["game_info"]
    del egs["assistant_state"]
    del egs["task_state"]["position"]
    egs["task_state"]["targets"] = egs["task_state"]["targets"][0, {"space": True}]
    assert egs == obs


def test_apply_deterministic_all_mapping_extradeterministic():
    def f(observation, gamestate, *args):
        gain = args[0]
        return gain * observation

    mapping = [
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), f, (2,), None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]

    obseng = RuleObservationEngine(mapping=mapping)
    obs, reward = obseng.observe(game_state=example_game_state())
    egs = example_game_state()
    del egs["game_info"]
    del egs["assistant_state"]
    egs["user_state"]["goal"] = 2 * egs["user_state"]["goal"]
    assert egs == obs


def test_apply_deterministic_all_mapping_extradeterministic_extraprobabilistic():
    goal = []
    for i in range(100):

        def f(observation, gamestate, *args):
            gain = args[0]
            return gain * observation

        def g(observation, gamestate, *args):
            return random.randint(0, 1) + observation

        mapping = [
            ("task_state", "position", slice(0, 1, 1), None, None, None, None),
            ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
            ("user_state", "goal", slice(0, 1, 1), f, (1,), g, ()),
            ("user_action", "action", slice(0, 1, 1), None, None, None, None),
            ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
        ]

        obs_eng = RuleObservationEngine(mapping=mapping)
        obs, reward = obs_eng.observe(game_state=example_game_state())
        egs = example_game_state()
        del egs["game_info"]
        del egs["assistant_state"]
        goal.append(obs["user_state"]["goal"].squeeze().tolist())
        assert obs["user_state"]["goal"] <= egs["user_state"]["goal"] + 1
        assert obs["user_state"]["goal"] >= egs["user_state"]["goal"]
        del egs["user_state"]["goal"]
        del obs["user_state"]["goal"]
        assert egs == obs

    assert set(goal) == set([0, 1])


def test_apply_mapping():
    test_apply_deterministic_all_mapping()
    test_apply_deterministic_all_mapping_remove_subsubstate()
    test_apply_deterministic_all_mapping_remove_subsubstate_slice()
    test_apply_deterministic_all_mapping_extradeterministic()
    test_apply_deterministic_all_mapping_extradeterministic_extraprobabilistic()


# =========================== Shortcuts ==================


def test_oracle():
    from coopihc.observation.utils import oracle_engine_specification

    obs_eng = RuleObservationEngine(
        deterministic_specification=oracle_engine_specification
    )
    gamestate = example_game_state()
    obs, reward = obs_eng.observe(game_state=gamestate)
    assert obs == gamestate


def test_blind():
    from coopihc.observation.utils import blind_engine_specification

    obs_eng = RuleObservationEngine(
        deterministic_specification=blind_engine_specification
    )
    gamestate = example_game_state()
    obs, reward = obs_eng.observe(game_state=gamestate)
    del gamestate["task_state"]
    del gamestate["user_state"]
    del gamestate["assistant_state"]
    assert obs == gamestate


def test_basetask():
    from coopihc.observation.utils import base_task_engine_specification

    obs_eng = RuleObservationEngine(
        deterministic_specification=base_task_engine_specification
    )
    gamestate = example_game_state()
    obs, reward = obs_eng.observe(game_state=gamestate)
    del gamestate["user_state"]
    del gamestate["assistant_state"]
    assert obs == gamestate


def test_baseuser():
    from coopihc.observation.utils import base_user_engine_specification

    obs_eng = RuleObservationEngine(
        deterministic_specification=base_user_engine_specification
    )
    gamestate = example_game_state()
    obs, reward = obs_eng.observe(game_state=gamestate)
    del gamestate["assistant_state"]
    assert obs == gamestate


def test_baseassistant():
    from coopihc.observation.utils import base_assistant_engine_specification

    obs_eng = RuleObservationEngine(
        deterministic_specification=base_assistant_engine_specification
    )
    gamestate = example_game_state()
    obs, reward = obs_eng.observe(game_state=gamestate)
    del gamestate["user_state"]
    assert obs == gamestate


def test_preimplemented_rules():
    test_oracle()
    test_blind()
    test_basetask()
    test_baseuser()
    test_baseassistant()


def test_observe():
    # Default to basetask engine
    obseng = RuleObservationEngine()
    with pytest.raises(AttributeError):
        obseng.observe()

    _example_state = example_game_state()
    obs = obseng.observe(_example_state)[0]  # remove reward
    del _example_state["user_state"]
    del _example_state["assistant_state"]

    assert _example_state.equals(obs, mode="hard")


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_observationengine()
    test_create_mapping()
    test_apply_mapping()
    test_preimplemented_rules()
    test_observe()
