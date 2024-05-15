from dataclasses import dataclass
import torch
from typing import Dict

__all__ = [
    "DatasetOutput"
    ]

@dataclass
class DatasetOutput:
    """Class for defining standarised output from DataLoaders

    Args:
        input (Dict[str,torch.Tensor]): The input to the model forward
        output (Dict[str,torch.Tensor]): The output to be passed to the loss 
        function
    """
    input:Dict[str,torch.Tensor]
    output:Dict[str,torch.Tensor]
        
