from eye.envs import ChenEyePointingTask
from eye.operators import ChenEye
from core.bundle import SinglePlayOperatorAuto


from loguru import logger

# log = ['terminal', 'file']
log = ['file']
# log = []
if 'terminal' not in log:
    logger.remove()
if 'file' in log:
    try:
        import os
        os.remove('logs/chen_eye_2d.log')
    except FileNotFoundError:
        pass
    logger.add('logs/chen_eye_2d.log', format = "{time} {level} {message}")



fitts_W = 4e-2
fitts_D = 0.8
perceptualnoise = 0.1
oculomotornoise = 0.02
task = ChenEyePointingTask(fitts_W, fitts_D)
operator = ChenEye(perceptualnoise, oculomotornoise)
bundle = SinglePlayOperatorAuto(task, operator)
bundle.reset()
bundle.render('plotext')
while True:
    obs, reward, is_done, _ = bundle.step()
    bundle.render('plotext')
    if is_done:
        break
