from pointing.envs import SimplePointingTask
from coopihc.bundle import _DevelopTask, SinglePlayUserAuto, PlayAssistant, PlayNone
from eye.envs import ChenEyePointingTask
from eye.users import ChenEye
from coopihc.observation import (
    ObservationEngine,
    RuleObservationEngine,
    CascadedObservationEngine,
)
from pointing.users import CarefulPointer
from pointing.assistants import BIGGain

import gym
import numpy
import copy

# Add a state to the SimplePointingTask to memorize the old position
class OldPositionMemorizedSimplePointingTask(SimplePointingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memorized = None

    def reset(self, dic={}):
        super().reset(dic=dic)
        self.state["OldPosition"] = copy.deepcopy(self.state["Position"])

    def user_step(self, *args, **kwargs):
        self.memorized = copy.deepcopy(self.state["Position"])
        obs, rewards, is_done, _doc = super().user_step(*args, **kwargs)
        obs["OldPosition"] = self.memorized
        return obs, rewards, is_done, _doc

    def assistant_step(self, *args, **kwargs):
        self.memorized = copy.deepcopy(self.state["Position"])
        obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
        obs["OldPosition"] = self.memorized
        return obs, rewards, is_done, _doc


pointing_task = OldPositionMemorizedSimplePointingTask(
    gridsize=31, number_of_targets=8, mode="position"
)
# bundle = _DevelopTask(pointing_task)
# bundle.reset()

fitts_W = 4e-2
fitts_D = 0.8
perceptualnoise = 0.09
oculomotornoise = 0.09
task = ChenEyePointingTask(fitts_W, fitts_D, dimension=1)
user = ChenEye(perceptualnoise, oculomotornoise, dimension=1)
obs_bundle = SinglePlayUserAuto(task, user, start_at_action=True)

# reset_dic = {'task_state':
#                 {   'Targets': .5,
#                     'Fixation': -.5    }
#             }
# obs_bundle.reset(reset_dic)


class ChenEyeObservationEngineWrapper(ObservationEngine):
    """Not impleted yet."""

    def __init__(self, obs_bundle):
        super().__init__()
        self.type = "process"
        self.obs_bundle = obs_bundle
        self.obs_bundle.reset()

    def observe(self, game_state):
        # Cast to the box of the obs bundle
        target = game_state["task_state"]["Position"].cast(
            self.obs_bundle.game_state["task_state"]["Targets"], inplace=False
        )
        fixation = game_state["task_state"]["OldPosition"].cast(
            self.obs_bundle.game_state["task_state"]["Fixation"], inplace=False
        )
        reset_dic = {"task_state": {"Targets": target, "Fixation": fixation}}

        self.obs_bundle.reset(dic=reset_dic)
        is_done = False
        rewards = 0
        while True:
            obs, reward, is_done, _doc = self.obs_bundle.step()
            rewards += reward
            if is_done:
                break
        obs["task_state"]["Fixation"].cast(
            game_state["task_state"]["OldPosition"], inplace=True
        )
        obs["task_state"]["Targets"].cast(
            game_state["task_state"]["Position"], inplace=True
        )
        return game_state, rewards


cursor_tracker = ChenEyeObservationEngineWrapper(obs_bundle)
base_user_engine_specification = [
    ("turn_index", "all"),
    ("task_state", "all"),
    ("user_state", "all"),
    ("assistant_state", None),
    ("user_action", "all"),
    ("assistant_action", "all"),
]
default_observation_engine = RuleObservationEngine(
    deterministic_specification=base_user_engine_specification,
)

observation_engine = CascadedObservationEngine(
    [cursor_tracker, default_observation_engine]
)
binary_user = CarefulPointer(observation_engine=observation_engine)
BIGpointer = BIGGain()
# bundle = PlayAssistant(pointing_task, binary_user, BIGpointer)


bundle = PlayNone(pointing_task, binary_user, BIGpointer)
game_state = bundle.reset()
bundle.render("plotext")
rewards = []
while True:
    reward, is_done, reward_list = bundle.step()
    rewards.append(reward_list)
    bundle.render("plotext")
    if is_done:
        break
