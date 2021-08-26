from envs import MultiBanditTask
from operators import RandomOperator, WSLS, RW

from core.agents import DummyAssistant

from core.bundle import PlayNone


N = 2
P = [0.5, 0.75]
T = 2

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# Operator definition
random_operator = RandomOperator()
wsls = WSLS(epsilon=0.1)
rw = RW(q_alpha=0.1, q_beta=2.0)

# Assistant definition
dummy_assistant = DummyAssistant()

# Bundle definition
# bundle = PlayNone(multi_bandit_task, random_operator, dummy_assistant)
# bundle = PlayNone(multi_bandit_task, wsls, dummy_assistant)
bundle = PlayNone(multi_bandit_task, rw, dummy_assistant)

bundle.reset()
bundle.render('text')

choices = []
rewards = []

while True:
    game_state, reward, is_done, _ = bundle.step()

    choice = game_state["task_state"]["last_action"]["values"][0]

    choices.append(choice)
    rewards.append(reward)

    bundle.render('text')

    print()
    print(f"Total rewards: {sum(rewards)}")
    print(f"All choices: {choices}")
    print(f"All rewards: {rewards}")
    print()

    if is_done:
        break
