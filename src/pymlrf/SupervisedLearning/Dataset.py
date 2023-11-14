from abc import abstractmethod
import torch

__all__ = [
    "DatasetOutput"
    ]

class DatasetOutput:
    
    def __init__(self, input:torch.Tensor, output:torch.Tensor) -> None:
        """Class for defining standarised output from DataLoaders

        Args:
            input (Any): The input to the model forward
            output (Any): The output to be passed to the loss function
        """
        self.input = input
        self.output = output
        
