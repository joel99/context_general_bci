from enum import Enum

from task_registry import ExperimentalTaskLoader, ExperimentalTaskRegistry
from tasks.passive_icms import ICMSLoader
from tasks.nlb import MazeLoader, RTTLoader

class ExperimentalTask(Enum):
    passive_icms = ICMSLoader.name
    maze = MazeLoader.name
    rtt = RTTLoader.name