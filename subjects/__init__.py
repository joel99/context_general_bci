from enum import Enum
from array_registry import SubjectArrayRegistry, SubjectInfo, ArrayInfo, ArrayID

# These import lines ensure registration
import pitt_chicago

# ? Should we be referencing this instance or the class in calls? IDK
# ? Preferring the class for now as it provides typing
subject_array_registry = SubjectArrayRegistry()

class SubjectName(Enum):
    CRS07 = pitt_chicago.CRS07.__name__
    CRS02b = pitt_chicago.CRS02b.__name__
    BCI02 = pitt_chicago.BCI02.__name__