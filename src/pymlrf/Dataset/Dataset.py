from typing import Dict, Union, List

from ..FileSystem import DirectoryHandler, FileHandler
from ..SerialisableConfig import SerialisableConfig
from .Feature import (DiscreteFeature, ContinuousFeature)

__all__ = ["Dataset"]


class Dataset(DirectoryHandler, SerialisableConfig):
    
    serialisable_attrs = [
        "features"
        ]
    
    def __init__(
        self, 
        loc: str, 
        config_fh:FileHandler,
        features: Dict[str, Union[DiscreteFeature, ContinuousFeature]] = None
        ):
        DirectoryHandler.__init__(self, loc)
        SerialisableConfig.__init__(self, config_fh)
        self.features = features
        