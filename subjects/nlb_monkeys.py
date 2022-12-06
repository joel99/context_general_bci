from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from subjects.array_registry import SortedArrayInfo, SubjectInfo, SubjectArrayRegistry

@SubjectArrayRegistry.register
class Jenkins(SubjectInfo):
    _arrays = {
        'main': SortedArrayInfo(_max_channels=137)
    }