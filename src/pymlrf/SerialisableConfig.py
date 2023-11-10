from abc import abstractmethod
from typing import Any, Dict
from json import JSONEncoder, JSONDecoder

from .FileSystem import FileHandler

__all__ = ["SerialisableConfig"]

class SerialisableConfig:
    
    serialisable_attrs = []
    
    def __init__(self, config_fh: FileHandler) -> None:
        self.config_fh = config_fh
    
    # def read(self, path):
    #     with open(path, "r") as f:
    #         json_string = f.read()
    #     f:Dict = JSONDecoder().decode(json_string)
    #     _ms_keys = []
    #     for key in f.keys():
    #         if hasattr(self, key):
    #             setattr(self, key, f[key])  
    #         else:
    #             _ms_keys.append(key)
    #     if len(_ms_keys) > 0:
    #         raise AttributeError(
    #             f"Config does not have attributes: {_ms_keys}"
    #             )
    
    # def write(self, path): 
    #     json_string = dje.encode(self)
    #     with open(path, "w") as f:
    #         f.write(json_string)
    
    def decode(self, json_string:str) -> Dict:
        return JSONDecoder().decode(json_string)
    
    def encode(self) -> str:
        return dje.encode(self)
    
    def read(self):
        with open(self.config_fh.path, "r") as f:
            json_string = f.read()
        f = self.decode(json_string)
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
    
    def write(self):
        json_string = self.encode()
        with open(self.config_fh.path, "w") as f:
            f.write(json_string)
  
             

class SerialisableConfigJsonEncoder(JSONEncoder):
    
    def default(self, sc:SerialisableConfig):
        return {key:getattr(sc, key) for key in sc.serialisable_attrs}

dje = SerialisableConfigJsonEncoder()
        