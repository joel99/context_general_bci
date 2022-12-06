from enum import Enum

from task_registry import ExperimentalTaskLoader, ExperimentalTaskRegistry
from tasks.passive_icms import ICMSLoader,
from tasks.nlb import NLBLoader

class ExperimentalTask(Enum):
    passive_icms = ICMSLoader.name
    maze = 'maze'
    rtt = 'random_target_task'