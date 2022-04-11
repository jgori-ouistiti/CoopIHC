from coopihc.interactiontask.ExampleTask import CoordinatedTask
from coopihc.agents.ExampleUser import PseudoRandomUser, PseudoRandomUserWithParams
from coopihc.agents.ExampleAssistant import (
    CoordinatedAssistant,
    CoordinatedAssistantWithInference,
    CoordinatedAssistantWithRollout,
)

from coopihc.bundle.Bundle import Bundle
import copy
import pytest

# [start-user-model]
user = PseudoRandomUser()
user_model = PseudoRandomUser()  # The same as user
assistant = CoordinatedAssistant(user_model=user_model)

bundle = Bundle(task=CoordinatedTask(), user=user, assistant=assistant)

# bundle.reset(go_to=3)
print(bundle.game_state)
while True:
    obs, rewards, is_done = bundle.step()
    # print(bundle.game_state)
    if is_done:
        break

# [end-user-model]
# [start-user-model-mismatch]
user = PseudoRandomUserWithParams(p=[1, 5, 7])
user_model = PseudoRandomUserWithParams(p=[5, 5, 7])  # Model mismatch

assistant = CoordinatedAssistant(user_model=user_model)

bundle = Bundle(task=CoordinatedTask(), user=user, assistant=assistant)

bundle.reset(go_to=3)
# print(bundle.game_state)
while True:
    obs, rewards, is_done = bundle.step()
    print(bundle.game_state)
    if is_done:
        break
# [end-user-model-mismatch]
# pytest.skip("below not working, but tmp anyways", allow_module_level=True)
# [start-user-model-inference]
# user = PseudoRandomUserWithParams(p=[1, 5, 7])
# user_model = PseudoRandomUserWithParams(p=[5, 5, 7])  # Model mismatch

# assistant = CoordinatedAssistantWithInference(user_model=user_model)

# bundle = Bundle(task=CoordinatedTask(), user=user, assistant=assistant)

# bundle.reset(go_to=2)
# # print(bundle.game_state)
# while True:
#     obs, rewards, is_done = bundle.step()
#     if is_done:
#         break

# [end-user-model-inference]


# [start-user-model-rollout2]
# user = PseudoRandomUserWithParams(p=[1, 5, 7])
# user_model = PseudoRandomUserWithParams(p=[5, 5, 7])  # Model mismatch

# simulated_bundle = Bundle(task = CoordinatedTask(), user = copy.deepcopy(user_model), )

# assistant = CoordinatedAssistantWithInferenceRollouts(user_model=user_model, )

# bundle = Bundle(task=CoordinatedTask(), user=user, assistant=assistant)

# bundle.reset(turn=2)
# print(bundle.game_state)
# while True:
#     obs, rewards, is_done = bundle.step()
#     if is_done:
#         break
# [end-user-model-rollout2]


# [start-user-model-rollout]
task = CoordinatedTask()
task_model = CoordinatedTask()

user = PseudoRandomUserWithParams(p=[1, 5, 7])
user_model = copy.deepcopy(user)


assistant = CoordinatedAssistantWithRollout(task_model, user_model, [5, 7])
bundle = Bundle(task=task, user=user, assistant=assistant)
bundle.reset()
while True:
    obs, rewards, is_done = bundle.step()
    if is_done:
        break
# [end-user-model-rollout]
