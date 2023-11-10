import os
import logging
from .Tracker import ModelTracker
from typing import Dict

logger = logging.getLogger("dtr_renal")

__all__ = [
    "get_create_tracker",
    "Option"
]

def get_create_tracker(tracker_path:str, updt_kwargs:Dict={}, 
                       u_id:str="model_name"):
    mt = ModelTracker(path=tracker_path, u_id=u_id)
    if mt.is_created:
        logger.info("Tracker identified. Importing...")
        # TODO: Implement the ability to specify what type of tracker to import i.e. dataframe/csv/json etc
        
        mt.read(**updt_kwargs)
    else:
        logger.info("Could not find tracker at location, creating new tracker")
    return mt

class Option:
    
    def __init__(self, option):
        if option not in ["overwrite", "duplicate", "error"]:
            raise TypeError("Values should only be one of overwrite, duplicate or error")
        self.option = option

    def __repr__(self):
        return str(self._option)