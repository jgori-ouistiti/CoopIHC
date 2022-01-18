import numpy


# ========================== Some observation engine specifications

oracle_engine_specification = [
    ("game_info", "all"),
    ("task_state", "all"),
    ("user_state", "all"),
    ("assistant_state", "all"),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

blind_engine_specification = [
    ("game_info", "all"),
    ("task_state", None),
    ("user_state", None),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_task_engine_specification = [
    ("game_info", "all"),
    ("task_state", "all"),
    ("user_state", None),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_user_engine_specification = [
    ("game_info", "all"),
    ("task_state", "all"),
    ("user_state", "all"),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]

base_assistant_engine_specification = [
    ("game_info", "all"),
    ("task_state", "all"),
    ("user_state", None),
    ("assistant_state", "all"),
    ("user_action", "all"),
    ("assistant_action", "all"),
]
