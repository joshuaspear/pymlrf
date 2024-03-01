import os
import logging
from typing import Any, Callable, Dict

from ..FileSystem.DirectoryHandler import DirectoryHandler
from .Tracker import Tracker
from .utils import Option


logger = logging.getLogger("pymlrf")

__all__ = ["Experiment"]

class Experiment(DirectoryHandler):
    
    def __init__(self, exp_name: str, parent_loc: str, mt: Tracker):
        super().__init__(loc=os.path.join(parent_loc, exp_name))
        self._mt = mt
        self._exp_name = exp_name
        self._status = False
        self._in_tracker = False
    
    @property
    def in_tracker(self):
        if self._status:
            return self._in_tracker
        else:
            raise Exception("Check status first")
        
    @property
    def is_created(self):
        if self._status:
            return super().is_created
        else:
            raise Exception("Check status first")

    def status_check(self):
        self._in_tracker = self._mt.check_model_exists(u_id=self._exp_name)
        self._status = True
    
    def delete(self):
        if self.in_tracker:
            self._mt.drop_model(u_id=self._exp_name)
            
        if self.is_created:
            super().delete()
            
                    
    def run(
        self, 
        option:Option, 
        func:Callable[[Any], Dict[str,Any]],
        force_columns=True,
        meta_data:Dict[str,Any] = {},
        *args, 
        **kwargs
        ) -> Dict[str,Any]:
        
        if option.option == "overwrite":
            self.delete()
            self.create()
        if option.option == "update":
            pass
        elif option.option == "duplicate":
            raise ValueError
        elif option.option == "error":
            if self.is_created or self.in_tracker:
                raise Exception(
                    f""" Either directory exists or experiment in tracker
                    Experiment directory exists: {self.is_created}
                    Experiment in tracker: {self.in_tracker}
                    """
                    )
            else:
                self.create()
        res = {**meta_data}
        res[self._mt.u_id] = self._exp_name
        try:
            res = {**res, **func(*args, **kwargs)}
            self._mt.update_tracker_w_dict(
                row_dict=res, 
                force_columns=force_columns
                )
        except Exception as e:    
            self._mt.write_run_error(u_id=self._exp_name)
            
            raise e
        return res