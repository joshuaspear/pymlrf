from dataclasses import dataclass
from typing import List
from ..SerialisableConfig import SerialisableConfig


__all__ = ["DiscreteFeature", "ContinuousFeature"]


@dataclass
class DiscreteFeature:
    
    unique_values: List[float] = None
    

@dataclass        
class ContinuousFeature:
    pass