import math
from dataclasses import dataclass

@dataclass(frozen=True)
class FelicificVector:
    intensity: float      # [-1, 1]
    duration: float       # [0, 1]
    certainty: float      # [0, 1]
    propinquity: float    # [0, 1]
    extent: float         # [-1, 1]

@dataclass(frozen=True)
class PriorityVector:
    intensity: float
    duration: float
    certainty: float
    propinquity: float
    extent: float

def distance(a: FelicificVector, b: PriorityVector) -> float:
    return math.sqrt(
        (a.intensity - b.intensity) ** 2 +
        (a.duration - b.duration) ** 2 +
        (a.certainty - b.certainty) ** 2 +
        (a.propinquity - b.propinquity) ** 2 +
        (a.extent - b.extent) ** 2
    )
