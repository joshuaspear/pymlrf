import logging
import numpy as np
import os
import pickle
from typing import Callable, Literal, Union
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.autograd.set_detect_anomaly(True)

from .Metric import MetricOrchestrator
from .EarlyStopper import (
    EarlyStopper, EarlyStopperPassThru, EarlyStoppingException
    )
from ..utils import set_seed

__all__ = [
    "train_single_epoch",
    "validate_single_epoch",
    "train"
]

def train_single_epoch(
    model:torch.nn.Module, 
    data_loader:DataLoader, 
    gpu:bool, 
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.modules.loss, 
    logger:logging.Logger
    ):
    
    model.train()
    losses = []
    preds = []
    range_gen = tqdm(
        enumerate(data_loader),
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
    data_loader:DataLoader,
    gpu:Literal[True, False], 
    criterion:torch.nn.modules.loss
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
    criterion:torch.nn.modules.loss, 
    epochs:int, 
    logger:logging.Logger, 
    save_dir:str, 
    scheduler:torch.optim.lr_scheduler.LRScheduler=None, 
    early_stopping_func:Union[Callable, None]=None, 
    es_action:Literal["stop", "capture"]="capture", 
    train_epoch_func:Callable=train_single_epoch, 
    val_epoch_func:Callable=validate_single_epoch,
    seed: int = None
    ) -> MetricOrchestrator:
    
    if seed is not None:
        set_seed(n=seed)
    # data_loader should output dictionaries of parameters
    
    # # Assert the dataloader provides an output of type DatasetOutput
    # first_val = data_loader.__next__()
    # if not isinstance(first_val, DatasetOutput):
    #     raise Exception("Dataloader should provide instances of type DatasetOutput")
    
    # Resetting orchestrators for new training run
    mo = MetricOrchestrator()
    mo.reset_orchestrator()
    mo.setup_orchestrator(name_trans_dict={
        "epoch_train_loss":{}, "epoch_val_loss":{}})
    
    if early_stopping_func:
        es = EarlyStopper(stopping_func=early_stopping_func, action=es_action)
    else:
        es = EarlyStopperPassThru()
    
    logger.info("Running epochs: {}".format(epochs))
    # Add model to cuda device
    if gpu:
        model.cuda()
        
    for epoch in np.arange(1,epochs+1):
        try:
            logger.info("Running training epoch")
            train_loss_val, train_preds =  train_epoch_func(
                model=model, data_loader=train_data_loader, gpu=gpu, 
                optimizer=optimizer, criterion=criterion,logger=logger)
            epoch_train_loss = np.mean(train_loss_val)
                        
            logger.info("epoch {}\t training loss : {}".format(
                    epoch, epoch_train_loss))
            val_loss_val, val_preds = val_epoch_func(
                model=model, data_loader=val_data_loader, gpu=gpu, 
                criterion=criterion)
            
            logger.info("Running validation")
            epoch_val_loss = np.mean(val_loss_val)
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
            
            with open(os.path.join(save_dir, "epoch_{}_train_preds.pkl".format(
                epoch)), "wb") as file:
                pickle.dump(train_preds, file)
                
            with open(os.path.join(save_dir, "epoch_{}_val_preds.pkl".format(
                epoch)), "wb") as file:
                pickle.dump(val_preds, file)
            
            es.assess(value=epoch_val_loss, epoch=epoch)
            
        except EarlyStoppingException:
            logger.info(
                "Early stopping criteria reached. Terminating training run")
        logger.warn("Implement final epoch properly")
        if early_stopping_func:
            if es.optimal_epoch == 0:
                fnl_epoch = epochs-1
            else:
                fnl_epoch = es.optimal_epoch
        else:
            fnl_epoch = es.get_min_epoch()
    return mo, fnl_epoch