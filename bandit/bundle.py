from envs import MultiBanditTask
from agents import RandomPlayer, WSLS, RW

from core.bundle import Bundle


N = 2
P = [0.5, 0.75]
T = 5

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# Operator definition
random_player = RandomPlayer()
wsls = WSLS(epsilon=0.1)
rw = RW(q_alpha=0.1, q_beta=2.0)

# Bundle definition
# bundle = Bundle(task=multi_bandit_task, user=random_player)
bundle = Bundle(task=multi_bandit_task, user=wsls)
# bundle = Bundle(task=multi_bandit_task, user=rw)

bundle.reset()
bundle.render("text")

choices = []
rewards = []

while True:
    game_state, round_rewards, is_done = bundle.step()

    choice = game_state["task_state"]["last_action"]["values"][0][0][0]

    choices.append(choice)
    rewards.append(round_rewards["first_task_reward"])

    bundle.render("text")

    print()
    print(f"Total rewards: {sum(rewards)}")
    print(f"All choices: {choices}")
    print(f"All rewards: {rewards}")
    print()

    if is_done:
        break
