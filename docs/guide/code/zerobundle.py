from pointing.envs import SimplePointingTask
from pointing.users import CarefulPointer
from pointing.assistants import ConstantCDGain, BIGGain

from core.bundle import PlayNone, PlayAssistant

import matplotlib.pyplot as plt
# ===================== First example =====================

# task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
# binary_user = CarefulPointer()
# unitcdgain = ConstantCDGain(1)
#
# bundle = PlayNone(task, binary_user, unitcdgain)
# game_state = bundle.reset()
# bundle.render('plotext')
# while True:
#     sum_rewards, is_done, rewards = bundle.step()
#     bundle.render('plotext')
#     if is_done:
#         bundle.close()
#         break

# ===================== Second example =====================

# task = SimplePointingTask(gridsize = 31, number_of_targets = 10)
# user = CarefulPointer()
# assistant = ConstantCDGain(1)
#
# bundle = PlayAssistant(task, user, assistant)
#
# game_state = bundle.reset()
# bundle.render('plotext')
# # The heuristic is as follows: Start with a high gain. The user should always give the same action. If at some point it changes, it means the user went past the target and that the cursor is very close to the target. If that is the case, divide the gain by 2, but never less than 1.
#
# # Start off with a high gain
# gain = 4
# # init for the adaptive algorithm
# sign_flag = game_state["user_action"]['action']['human_values'][0]
# observation = game_state
# _return = 0
# while True:
#     # Check whether the user action changed:
#     sign_flag = sign_flag * observation["user_action"]['action']['human_values'][0]
#     # If so, divide gain by 2
#     if sign_flag == -1:
#         gain = max(1,gain/2)
#     # Apply assistant action
#     observation, sum_rewards, is_done, rewards = bundle.step([gain])
#     _return += sum_rewards
#     bundle.render('plotext')
#
#     if is_done:
#         bundle.close()
#         break
#
# print(_return)

# ================= Third example ======================

task = SimplePointingTask(gridsize = 31, number_of_targets = 10, mode = 'position')
binary_user = CarefulPointer()

BIGpointer = BIGGain()

bundle = PlayNone(task, binary_user, BIGpointer)

game_state = bundle.reset()
bundle.render('plotext')
plt.tight_layout()

while True:
    sum_rewards, is_done, rewards = bundle.step()
    bundle.render('plotext')

    if is_done:
        break
