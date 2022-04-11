from coopihc import SimplePointingTask, CarefulPointer, ConstantCDGain
from coopihc import Bundle


task = SimplePointingTask(gridsize=31, number_of_targets=8)
user = CarefulPointer(error_rate=0.05)
assistant = ConstantCDGain(1)
bundle = Bundle(task=task, user=user, assistant=assistant)

game_state = bundle.reset()
bundle.render("plotext")
