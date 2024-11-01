import logging
import numpy as np
import os
import pickle
from typing import Callable, Literal, Union, Optional
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.autograd.set_detect_anomaly(True)

from ...types import (
    TrainSingleEpochProtocol, 
    ValidateSingleEpochProtocol,
    GenericDataLoaderProtocol,
    CriterionProtocol
    )
from .Metric import MetricOrchestrator
from .EarlyStopper import (
    EarlyStopper, EarlyStoppingException, PassThruStoppingCriteria
    )
from ...utils import set_seed

__all__ = [
    "train_single_epoch",
    "validate_single_epoch",
    "train"
]

def train_single_epoch(
    model:torch.nn.Module, 
    data_loader:GenericDataLoaderProtocol, 
    gpu:bool, 
    optimizer:torch.optim.Optimizer,
    criterion:CriterionProtocol, 
    logger:logging.Logger
    ):
    
    model.train()
    losses = []
    preds = []
    range_gen = tqdm(
        enumerate(data_loader),
        total=len(data_loader)
        #desc=f"Epoch {int(epoch)}/{epochs}",
        )
    for i, vals in range_gen:
        
        input_vals = vals.input
        output_vals = vals.output
        if gpu:
            input_vals = {
                val:input_vals[val].cuda() for val in input_vals
                }
            output_vals = {
                val:output_vals[val].cuda() for val in output_vals
                }
        else:
            input_vals = {val:Variable(input_vals[val]) for val in input_vals}
            output_vals = {val:Variable(output_vals[val]) 
                           for val in output_vals}
        
        optimizer.zero_grad()

        # Compute output
        output = model(**input_vals)
        preds.append(output)
        train_loss = criterion(output, output_vals)
        losses.append(train_loss.item())

        # losses.update(train_loss.data[0], g.size(0))
        # error_ratio.update(evaluation(output, target).data[0], g.size(0))

        try: 
            # compute gradient and do SGD step
            train_loss.backward()
            
            optimizer.step()
        except RuntimeError as e:
            logger.debug("Runtime error on training instance: {}".format(i))
            raise e
    return losses, preds
            
def validate_single_epoch(
    model:torch.nn.Module, 
    data_loader:GenericDataLoaderProtocol,
    gpu:Literal[True, False], 
    criterion:CriterionProtocol
    ):
    
    model.eval()
    losses = []
    preds = []
    with torch.no_grad():
        for i, vals in enumerate(data_loader):

            # Prepare input data
            input_vals = vals.input
            output_vals = vals.output
            if gpu:
                input_vals = {
                    val:input_vals[val].cuda() for val in input_vals
                    }
                output_vals = {
                    val:output_vals[val].cuda() for val in output_vals
                    }
            else:
                input_vals = {
                    val:Variable(input_vals[val]) for val in input_vals
                    }
                output_vals = {
                    val:Variable(output_vals[val]) for val in output_vals
                    }

            # Compute output
            output = model(**input_vals)

            # Logs
            losses.append(criterion(output, output_vals).item())
            preds.append(output)
    return losses, preds



def train(
    model:torch.nn.Module, 
    train_data_loader:DataLoader,
    val_data_loader:DataLoader, 
    gpu:bool, 
    optimizer:torch.optim.Optimizer,
    criterion:CriterionProtocol, 
    epochs:int, 
    logger:logging.Logger, 
    save_dir:str, 
    scheduler:torch.optim.lr_scheduler.LRScheduler=None, 
    early_stopping_func:Union[Callable, None]=None, 
    es_action:Literal["stop", "capture"]="capture", 
    train_epoch_func:TrainSingleEpochProtocol = train_single_epoch, 
    val_epoch_func:ValidateSingleEpochProtocol = validate_single_epoch,
    seed: int = None,
    mo: MetricOrchestrator = MetricOrchestrator(),
    val_criterion:Optional[CriterionProtocol] = None,
    preds_save_type:Optional[Literal["pickle","csv"]] = None
    ) -> MetricOrchestrator:
    
    if seed is not None:
        set_seed(n=seed)
    # data_loader should output dictionaries of parameters
    
    # # Assert the dataloader provides an output of type DatasetOutput
    # first_val = data_loader.__next__()
    # if not isinstance(first_val, DatasetOutput):
    #     raise Exception("Dataloader should provide instances of type DatasetOutput")
    
    mo.add_metric(
        nm="epoch_train_loss",
        rll_trans={}
        )
    mo.add_metric(
        nm="epoch_val_loss",
        rll_trans={}
        )
    
    if early_stopping_func:
        es = EarlyStopper(stopping_func=early_stopping_func, action=es_action)
    else:
        es = EarlyStopper(
            stopping_func=PassThruStoppingCriteria(),
            action=es_action
            )
    
    logger.info("Running epochs: {}".format(epochs))
    # Add model to cuda device
    if gpu:
        model.cuda()

    if val_criterion is None:
        val_criterion = criterion
        
    
    for epoch in np.arange(1,epochs+1):
        try:
            logger.info("Running training epoch")
            train_loss_val, train_preds =  train_epoch_func(
                model=model, data_loader=train_data_loader, gpu=gpu, 
                optimizer=optimizer, criterion=criterion,logger=logger)
            epoch_train_loss = np.mean(train_loss_val)
            del train_loss_val           
            logger.info("epoch {}\t training loss : {}".format(
                    epoch, epoch_train_loss))
            val_loss_val, val_preds = val_epoch_func(
                model=model, data_loader=val_data_loader, gpu=gpu, 
                criterion=val_criterion)
            logger.info("Running validation")
            epoch_val_loss = np.mean(val_loss_val)
            del val_loss_val
            logger.info("epoch {}\t validation loss : {} ".format(
                    epoch, epoch_val_loss))
            
            mo.update_metrics(metric_value_dict={
                "epoch_train_loss":{"label":"epoch_{}".format(epoch), 
                                    "value":epoch_train_loss},
                "epoch_val_loss":{"label":"epoch_{}".format(epoch), 
                                "value":epoch_val_loss}
            })
            
            if scheduler:
                scheduler.step()            
                
            chkp_pth = os.path.join(save_dir, "mdl_chkpnt_epoch_{}.pt".format(
                epoch))
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss,
                }, chkp_pth)
            
            if preds_save_type is not None:
                if preds_save_type == "pickle":
                    for k in train_preds.keys():
                        with open(
                            os.path.join(
                                save_dir, 
                                f"epoch_{epoch}_train_preds_{k}.pkl"
                                ), "wb"
                            ) as file:
                            pickle.dump(train_preds[k], file)
                    for k in val_preds.keys():
                        with open(
                            os.path.join(
                                save_dir, 
                                f"epoch_{epoch}_val_preds_{k}.pkl"
                                ), "wb"
                            ) as file:
                            pickle.dump(val_preds[k], file)
                        
                elif preds_save_type == "csv":
                    for k in train_preds.keys():
                        np.savetxt(
                            os.path.join(
                                save_dir, 
                                f"epoch_{epoch}_train_preds_{k}.csv"
                                ), 
                            train_preds[k].detach().cpu().float().numpy(), 
                            delimiter=","
                            )
                    for k in val_preds.keys():
                        np.savetxt(
                            os.path.join(
                                save_dir, 
                                f"epoch_{epoch}_val_preds_{k}.csv"
                                ), 
                            val_preds[k].detach().cpu().float().numpy(), 
                            delimiter=","
                            )
                else:
                    raise ValueError(
                        "preds_save_type must be either None, csv or pickle"
                        )
            
            es.assess(value=epoch_val_loss, epoch=epoch)
            
        except EarlyStoppingException:
            logger.info(
                "Early stopping criteria reached. Terminating training run"
                )
        if early_stopping_func:
            fnl_epoch = es.optimal_epoch
        else:
            fnl_epoch = es.get_min_epoch()
    return mo, fnl_epoch