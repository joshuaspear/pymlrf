from typing import Dict, Union, List

from ..FileSystem import DirectoryHandler, FileHandler
from ..SerialisableConfig.base import SerialisableConfig
from .Feature import (DiscreteFeature, ContinuousFeature)

__all__ = ["Dataset"]


class Dataset(DirectoryHandler, SerialisableConfig):
    
    def __init__(
        self, 
        loc: str, 
        config_fh:FileHandler
        ):
        DirectoryHandler.__init__(self, loc)
        SerialisableConfig.__init__(self, config_fh)
        