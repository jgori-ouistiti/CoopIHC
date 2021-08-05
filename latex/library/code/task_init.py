from pointing.envs import SimplePointingTask
from pointing.operators import CarefulPointer
from pointing.assistants import ConstantCDGain
from core.bundle import PlayNone

task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
operator = CarefulPointer()
assistant = ConstantCDGain(1)

bundle = PlayNone(task, operator, assistant)
game_state = bundle.reset()
bundle.render('plotext')
