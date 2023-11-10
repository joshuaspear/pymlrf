import os
import time

__all__ = ["FileHandler"]

class FileHandler:
    
    def __init__(self, path:str) -> None:
        self.path = path
    
    @property
    def is_created(self):
        return os.path.isfile(self.path)
        
    def delete(self):
        os.remove(self.path)
        while self.is_created:
            time.sleep(1)
