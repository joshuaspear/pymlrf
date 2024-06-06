from abc import abstractmethod
import numpy as np
from typing import Callable, Literal, Union, List

from ... import logger

__all__ = [
    "EarlyStopper", 
    "PassThruStoppingCriteria",
    "PercEpsImprove",
    "StoppingCriteria",
    "EarlyStoppingException"
]

class EarlyStoppingException(Exception):
    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class StoppingCriteria:
    """Generic class to support stopping functions.
    Stopping functions should:
        1. Be supplied with the current value and current epoch and determine 
        whether training should be halted;
        2. Track the best epoch and best value
    """
    
    def __init__(self) -> None:
        self.__optimal_epoch:Union[int,None] = None
        self.update:bool = True
    
    @abstractmethod
    def __call__(
        self, 
        value:float, 
        epoch:int
        )->bool:
        """Using the current value and epoch, determines whether the training 
        should be halted

        Args:
            value (float): _description_
            epoch (int): _description_

        Returns:
            bool: _description_
        """
        pass
    
    @property
    def optimal_epoch(self):
        return self.__optimal_epoch
    
    @optimal_epoch.setter
    def optimal_epoch(self, val:Union[int,None]):
        if self.update:
            if val is not None:
                assert val >= 1, "Epoch must be greater than 0"
            self.__optimal_epoch = val
        else:
            pass        


class PercEpsImprove(StoppingCriteria):
    
    __dir_lkp = {
        "ls": lambda a,b: a<b,
        "gr": lambda a,b: a>b
    }
    
    def __init__(
        self, 
        eps, 
        direction:Literal["ls", "gr"] = "ls"
        ) -> None:
        super().__init__()
        self.__eps = eps
        self.__dir = self.__dir_lkp[direction]
        self.__previous_value:Union[float,None] = None        
    
    def __evaluate(
        self,
        value:float, 
        epoch:int
        )->bool:
        # If the percentage change in loss between the current and previous 
        # epoch is less than epsilon, the training loss will terminate.
        res = self.__dir(
            (self.__previous_value-value)/self.__previous_value, 
            self.__eps
            )
        if res:
            # The best epoch is the epoch before the current one since the 
            # current one did not induce a large enough change therefore don't 
            # update
            self.update = False
            pass
        else:
            self.optimal_epoch = epoch
        return res
    
    
    def __call__(
        self, 
        value:float, 
        epoch:int
        )->bool:
        if self.optimal_epoch is None:
            self.optimal_epoch = epoch
            res = False
        else:
            res = self.__evaluate(value=value,epoch=epoch)
        self.__previous_value = value
        return res


class PassThruStoppingCriteria(StoppingCriteria):
    
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(
        self, 
        value:float, 
        epoch:int
        )->bool:
        self.optimal_epoch = epoch    
        return False


class EarlyStopper:
    
    def __init__(
        self, 
        stopping_func:StoppingCriteria, 
        action:Literal["capture", "stop"]
        ) -> None:
        self.__values:List[float] = []
        self.__epochs:List[int] = []
        self.__stopping_func = stopping_func
        assert action in ["stop","capture"], "action parameter must be one of capture or stop"
        self.action = action
    
    def update(self, value:float, epoch:int):
        """Logs the supplied value.

        Args:
            value (float): Value of loss associated with epoch
            epoch (int): Epoch to log
        """
        self.__values.append(value)
        self.__epochs.append(epoch)
        
    def evaluate(self, value:float, epoch:int):
        """Runs the supplied stopping function, if the stopping function
        evaluates to true, runs the function to end the training run.
            Currently the only ending functions are 'stop' which terminates the 
            run or 'capture' which logs the associated epoch as optimal but 
            continues the training run.

        Args:
            value (float): Current loss value
            epoch (int): Current epoch
        """
        res = self.__stopping_func(value, epoch)
        if res:
            if self.action == "stop":
                raise EarlyStoppingException("Stopping criteria reached")
            
    def assess(self, value:float, epoch:int):
        """Takes as input the loss value associated with the supplied epoch.
            1. Logs the values;
            2. Evaluates whether the training loop should be stopped according 
            to the supplied stopping function

        Args:
            value (float): Value of loss associated with epoch
            epoch (int): Epoch to log
        """
        self.update(value=value, epoch=epoch)
        self.evaluate(value=value,epoch=epoch)
            
    def get_min_epoch(self):
        return self.__epochs[np.argmin(self.__values)]
    
    @property
    def optimal_epoch(self):
        return self.__stopping_func.optimal_epoch
            