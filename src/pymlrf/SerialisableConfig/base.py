from abc import abstractmethod
from typing import Any, Dict
from json import JSONEncoder, JSONDecoder
import pickle

from ..FileSystem import FileHandler

__all__ = ["SerialisableConfig"]

class SerialisableConfig:
    
    serialisable_attrs = []
    
    def __init__(self, config_fh: FileHandler) -> None:
        self.config_fh = config_fh
    
    @abstractmethod
    def decode(self, path:str) -> Dict:
        pass
    
    def read(self):
        f = self.decode(self.config_fh.path)
        _ms_keys = []
        for key in f.keys():
            if hasattr(self, key):
                setattr(self, key, f[key])  
            else:
                _ms_keys.append(key)
        if len(_ms_keys) > 0:
            raise AttributeError(
                f"Config does not have attributes: {_ms_keys}"
                )
    
    @abstractmethod
    def write(self):
       pass