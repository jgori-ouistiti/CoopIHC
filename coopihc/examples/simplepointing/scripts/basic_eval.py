from coopihc import SimplePointingTask, CarefulPointer, ConstantCDGain
from coopihc import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8)
binary_user = CarefulPointer(error_rate=0.05)
unitcdgain = ConstantCDGain(1)
bundle = Bundle(task=task, user=binary_user, assistant=unitcdgain)
game_state = bundle.reset()
bundle.render("plotext")
k = 0
while True:
    k += 1
    game_state, rewards_dic, is_done = bundle.step(
        user_action=None, assistant_action=None
    )
    print(rewards_dic)
    bundle.render("plotext")
    if is_done:
        bundle.close()
        break
