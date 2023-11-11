from typing import Dict
import pickle

from ..FileSystem import FileHandler
from .base import SerialisableConfig

__all__ = ["SerialisablePickleConfig"]

class SerialisablePickleConfig(SerialisableConfig):
    
    def __init__(self, config_fh: FileHandler) -> None:
        super().__init__(config_fh=config_fh)
        
    def decode(self, path:str) -> Dict:
        with open(path, "rb") as f:
            res = pickle.load(f)
        return res
        
    def write(self):
        res = scpe.encode(self)
        with open(self.config_fh.path, "wb") as f:
            pickle.dump(res, f)

class SerialisableConfigPickleEncoder:
    
    def encode(self, sc:SerialisableConfig):
        return {key:getattr(sc, key) for key in sc.serialisable_attrs}

scpe = SerialisableConfigPickleEncoder()
