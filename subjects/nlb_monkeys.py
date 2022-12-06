
from subjects.array_registry import SortedArrayInfo, SubjectInfo, SubjectArrayRegistry

@SubjectArrayRegistry.register
class Jenkins(SubjectInfo):
    name = "Jenkins"
    _arrays = {
        'main': SortedArrayInfo(_max_channels=137)
    }