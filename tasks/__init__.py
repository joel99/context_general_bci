from enum import Enum

# Exports
from .task_registry import ExperimentalTaskLoader, ExperimentalTaskRegistry
from .passive_icms import ICMSLoader
from .nlb import MazeLoader, RTTLoader

r"""
    Dependency notes:
    - ContextInfo stores ExperimentalTask to key into ExperimentalTaskRegistry. (data.py uses this)
    - ExperimentalTask requires ContextInfo to load. (context info has array info)

"""

class ExperimentalTask(Enum):
    passive_icms = ICMSLoader.name
    maze = MazeLoader.name
    rtt = RTTLoader.name