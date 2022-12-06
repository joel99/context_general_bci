from subjects import SubjectName, SortedArrayInfo, SubjectInfo, SubjectArrayRegistry

@SubjectArrayRegistry.register
class Jenkins(SubjectInfo):
    name = SubjectName.jenkins
    _arrays = {
        'main': SortedArrayInfo(_max_channels=137)
    }

@SubjectArrayRegistry.register
class Indy(SubjectInfo):
    name = SubjectName.indy
    _arrays = {
        'main': SortedArrayInfo(_max_channels=137) # TODO
    }