import torch
from typing import Any, Dict

__all__ = [
    "BalancedLoss",
    "PassLoss",
    "DuelInputLoss",
    "RegDuelInputLoss",
    "StableBenchCriteria"
    ]

class BalancedLoss:
    
    def __init__(self, loss_lkp:Dict) -> None:
        """Class for handling MultiLoss outputs

        Args:
            loss_lkp (Dict[str:torch.nn.modules.loss]): Dictionary containing 
            the required losses and lookup values. The lookup values should
            match those passed to the call method in act and pred parameters
        """
        self.loss_lkp = loss_lkp
        

    def __call__(self, pred:Dict[str, torch.Tensor],
                 act:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluates the input values against the loss functions specified in 
        the init.

        Args:
            act (Dict[str:torch.Tensor]): Dictionary of the 
            form {"name_of_loss": actual_values}
            The keys should match the keys provided in the loss_lkp parameter, 
            specified in the init
            pred (Dict[str:torch.Tensor]): Dictionary of the 
            form {"name_of_loss": predicted_values}
            The keys should match the keys provided in the loss_lkp parameter, 
            specified in the init

        Returns:
            Dict[str:Any]: Dictionary of evaluated results. The keys will match
            those provided in the multi_loss parameter
        """
        loss = 0
        for key in self.loss_lkp.keys():
            loss += self.loss_lkp[key](pred[key], act[key])
        out_loss = torch.mean(loss)
        return out_loss
    
    
class PassLoss:
    
    def __init__(self):
        pass
    
    def __call__(self, pred, act) -> Any:
        return pred
    
    
class DuelInputLoss:
    
    def __init__(self, loss_func) -> None:
        self.__loss_func = loss_func
    
    def __call__(self, pred, act):
        return self.__loss_func(pred[0], pred[1])
    

class RegDuelInputLoss:
    
    def __init__(self, loss_func) -> None:
        self.__loss_func = loss_func
        
    
    def __call__(self, pred, act):
        return (self.__loss_func(pred[0], pred[1])*pred[2]).mean()
    


class StableBenchCriteria:
    
    def __init__(self, loss_func):
        self.__loss_func = loss_func
    
    def __call__(self, pred, act):
        
        return self.__loss_func(torch.exp(pred[2]), act["cross_ent"])
    