from coopihc import SimplePointingTask, CarefulPointer, BIGGain
from coopihc import Bundle

import matplotlib.pyplot as plt

task = SimplePointingTask(gridsize=31, number_of_targets=8, mode="position")
binary_user = CarefulPointer(error_rate=0.05)
BIGpointer = BIGGain()

bundle = Bundle(task=task, user=binary_user, assistant=BIGpointer)
game_state = bundle.reset()
# bundle.render("plot")
# plt.tight_layout()
k = 0
# plt.savefig("/home/juliengori/Pictures/img_tmp/biggain_{}.png".format(k))
while True:
    game_state, rewards, is_done = bundle.step(user_action=None, assistant_action=None)
    # bundle.render("plot")
    k += 1
    # plt.savefig("/home/juliengori/Pictures/img_tmp/biggain_{}.png".format(k))
    if is_done:
        bundle.close()
        break
