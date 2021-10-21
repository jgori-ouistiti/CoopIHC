from pointing.envs import SimplePointingTask
from pointing.users import CarefulPointer
from pointing.assistants import ConstantCDGain
from core.bundle import PlayNone

task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
user = CarefulPointer()
assistant = ConstantCDGain(1)

bundle = PlayNone(task, user, assistant)
game_state = bundle.reset()
bundle.render('plotext')
