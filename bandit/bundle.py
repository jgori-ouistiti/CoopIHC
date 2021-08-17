from envs import MultiBanditTask
from operators import RandomOperator
from assistants import DummyAssistant

from core.bundle import PlayNone


N = 2
P = [0.5, 0.75]
T = 5

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# Operator definition
random_operator = RandomOperator(N=N)

# Assistant definition
dummy_assistant = DummyAssistant()

# Bundle definition
bundle = PlayNone(multi_bandit_task, random_operator, dummy_assistant)

game_state = bundle.reset()
bundle.render('text')

rewards = []

while True:
    _, reward, is_done, _ = bundle.step()
    rewards.append(reward)
    bundle.render('text')

    if is_done:
        break

print()
print(f"Total rewards: {sum(rewards)}")
print(f"All rewards: {rewards}")
print()
