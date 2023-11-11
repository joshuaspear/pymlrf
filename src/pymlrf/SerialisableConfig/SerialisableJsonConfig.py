from typing import Any, Dict
from json import JSONEncoder, JSONDecoder

from ..FileSystem import FileHandler
from .base import SerialisableConfig

__all__ = ["SerialisableJsonConfig"]

class SerialisableJsonConfig(SerialisableConfig):
    
    def __init__(self, config_fh: FileHandler) -> None:
        super().__init__(config_fh=config_fh)
        
    def decode(self, path:str) -> Dict:
        return scjd.decode(path)
        
    def write(self):
        json_string = scje.encode(self)
        with open(self.config_fh.path, "w") as f:
            f.write(json_string)
  
             
class SerialisableConfigJsonEncoder(JSONEncoder):
    
    def default(self, sc:SerialisableConfig):
        return {key:getattr(sc, key) for key in sc.serialisable_attrs}

scje = SerialisableConfigJsonEncoder()

class SerialisableConfigJsonDecoder:
    
    def decode(self, path:str) -> Any:
        with open(path, "r") as f:
            json_string = f.read()
        return JSONDecoder().decode(json_string)

scjd = SerialisableConfigJsonDecoder()
