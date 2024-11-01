from typing import Protocol, Tuple, Literal, Any, runtime_checkable
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict

from .Structs.torch.Dataset import DatasetOutput

@runtime_checkable
class TrainSingleEpochProtocol(Protocol):
        
    def __call__(
        self, 
        model:torch.nn.Module, 
        data_loader:DataLoader, 
        gpu:bool, 
        optimizer:torch.optim.Optimizer,
        criterion:torch.nn.modules.loss, 
        logger:logging.Logger
        ) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
        ...

@runtime_checkable
class ValidateSingleEpochProtocol(Protocol):
    
    def __call__(
        self, 
        model:torch.nn.Module, 
        data_loader:DataLoader,
        gpu:Literal[True, False], 
        criterion:torch.nn.Module
        ) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
        ...

@runtime_checkable
class GenericDataLoaderProtocol(Protocol):
    
    def __iter__(self)->"GenericDataLoaderProtocol":
        ...

    def __next__(self)->DatasetOutput:
        ...

@runtime_checkable
class CriterionProtocol(Protocol):

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        ...