from .SerialisableConfig import SerialisableConfig

class PyMlObj:
    
    def __init__(self) -> None:
        pass
    
    def from_config(self, sc:SerialisableConfig):
        for i in sc.serialisable_attrs:
            setattr(self, i, getattr(sc, i))