import numpy as np
from typing import Callable, Literal

from .. import logger

__all__ = [
    "EarlyStopper", 
    "EarlyStopperPassThru",
    "PercEpsImprove"
]


def pss_thr_stp_func_false(x, y):
    return False

class EarlyStoppingException(Exception):
    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        

class EarlyStopper:
    
    def __init__(self, stopping_func:Callable, 
                 action:Literal["capture", "stop"]) -> None:
        self.__values = []
        self.__epochs = []
        self.__stopping_func = stopping_func
        self.optimal_epoch = 0
        if action == "stop":
            self.__end_fnc = self.__stp
        elif action == "capture":
            self.__end_fnc = self.__cptr
        else:
            raise Exception("action parameter must be one of capture or stop")
        
    def __stp(self):
        self.__cptr()
        raise Exception("Stopping criteria reached")
    
    def __cptr(self):
        self.optimal_epoch = self.__epochs[-1]
        self.__stopping_func = pss_thr_stp_func_false
        
    
    def update(self, value, epoch):
        self.__values.append(value)
        self.__epochs.append(epoch)
        
    def evaluate(self):
        res = self.__stopping_func(self.__values[-1], self.__values[-2])
        if res: 
            self.__end_fnc()
            
    def assess(self, value, epoch):
        self.update(value=value, epoch=epoch)
        try:
            self.evaluate()
        except IndexError:
            logger.debug("First epoch so skipping early stopping evaluation")
            pass
    
    def get_min_epoch(self):
        return self.__epochs[np.argmin(self.__values)]
            

class EarlyStopperPassThru(EarlyStopper):
    
    def __init__(self) -> None:
        super().__init__(stopping_func=pss_thr_stp_func_false, action="stop")
        
    def assess(self, value, epoch):
        self.update(value, epoch)


# ***** Stopping criteria *****

class PercEpsImprove:
    
    def __init__(self, eps) -> None:
        self.__eps = eps
    
    def __call__(self, curr, prev):
        # Assumes the previous loss is greater than the current loss. If it is 
        # not, i.e. the validation loss begins to increase, the training will
        # terminate. If the percentage change is loss is less than epsilon, the
        # training will terminate. 
        return (prev-curr)/prev < self.__eps