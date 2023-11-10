import os
import shutil
import time

__all__ = ["DirectoryHandler"]

class DirectoryHandler:
    
    def __init__(self, loc:str):
        self.loc = loc
    
    @property
    def is_created(self):
        return os.path.exists(self.loc)
    
    @property
    def is_empty(self):
        return len(os.listdir(self.loc)) == 0 
        
    def create(self):
        if not self.is_created:
            os.makedirs(self.loc)
        else:
            raise Exception
    
    def delete(self):
        shutil.rmtree(self.loc)
        while self.is_created:
            time.sleep(1)
    
    def clear(self):
        self.delete()
        self.create()
            

            
        
    