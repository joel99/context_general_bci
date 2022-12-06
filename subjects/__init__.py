from enum import Enum
from .array_registry import SubjectArrayRegistry, SubjectInfo, ArrayInfo, ArrayID

# These import lines ensure registration
import subjects.pitt_chicago as pitt_chicago

# ? Should we be referencing this instance or the class in calls? IDK
subject_array_registry = SubjectArrayRegistry()

class SubjectNames:
    # We refer to names instead of classes to make converting to singleton pattern easier
    CRS07 = pitt_chicago.CRS07.name
    CRS02b = pitt_chicago.CRS02b.name
    BCI02 = pitt_chicago.BCI02.name