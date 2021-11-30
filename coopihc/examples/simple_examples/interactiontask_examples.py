import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))

from coopihc.interactiontask.ExampleTask import ExampleTask

et = ExampleTask()
