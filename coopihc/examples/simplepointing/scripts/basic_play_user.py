from coopihc import SimplePointingTask, CarefulPointer, ConstantCDGain
from coopihc import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8)
binary_user = CarefulPointer(error_rate=0)
unitcdgain = ConstantCDGain(1)
bundle = Bundle(task=task, user=binary_user, assistant=unitcdgain)
game_state = bundle.reset(go_to=1)
bundle.render("plot")
k = 0
while True:
    k += 1
    game_state, rewards, is_done = bundle.step(
        user_action=binary_user.take_action()[0], assistant_action=None
    )
    game_state, rewards, is_done = bundle.step()
    bundle.render("plot")
    if is_done:
        bundle.close()
        break
