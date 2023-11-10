import logging
import pickle
import os
import torch
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List
from parse import parse
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.base import LearnableBase, load_learnable
from d3rlpy.metrics.evaluators import (make_batches, WINDOW_SIZE)
from d3rlpy.interface import QLearningAlgoProtocol
from offline_rl_ope.components import (
    ISWeightOrchestrator, BehavPolicy, D3RlPyDeterministic)
from offline_rl_ope.api.d3rlpy import (
    D3RlPyTorchAlgoPredict, FQECallback
)
from offline_rl_ope.OPEEstimators import ISEstimator
logger = logging.getLogger("dtr_renal")


class TrajInitialStateValueEstimationScorer:
    
    def __init__(self) -> None:
        self.cache = []
    
    def __call__(self, algo:QLearningAlgoProtocol, dataset: ReplayBuffer
                 ) -> float:
        total_values = []
        for episode in dataset.episodes:
            for batch in make_batches(episode, WINDOW_SIZE, 
                                      dataset.transition_picker):
                # estimate action-value in initial states
                first_obs = np.expand_dims(batch.observations[0], axis=0)
                actions = algo.predict(first_obs)
                values = algo.predict_value(first_obs, actions)
                total_values.append(values[0])
        self.cache = total_values
        return float(np.mean(total_values))
    

class _Evaluator:
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, dataset:ReplayBuffer, model:LearnableBase, 
                 discount:float, save_dir:str) -> Dict:
        raise NotImplementedError

class ISEstimatorWrapper(ISEstimator):
    
    def __init__(self, is_type:str, norm_weights: bool, clip: float = None,
                 cache_traj_rewards:bool=False, norm_kwargs:Dict[str,Any] = {}
                 ) -> None:
        super().__init__(norm_weights=norm_weights, clip=clip, 
                         cache_traj_rewards=cache_traj_rewards, 
                         norm_kwargs=norm_kwargs)
        self.is_type = is_type
        
    def predict(self, rewards: List[torch.Tensor], states: List[torch.Tensor], 
                actions: List[torch.Tensor], discount: float, 
                is_weight_calculator:ISWeightOrchestrator
                ) -> float:
        weights = is_weight_calculator[self.is_type].traj_is_weights
        is_msk = is_weight_calculator.weight_msk
        return  super().predict(rewards, states, actions, weights, discount, 
                                is_msk)


class RegIsPipeline(_Evaluator):
    
    def __init__(self, behav_est, is_types:List[str], 
                 is_scorers:Dict[str,ISEstimatorWrapper], gpu:bool=False,
                 lb:Dict[str,Callable]=None, lb_delta:float=0.05, 
                 debug_is_weights:bool=False) -> None:
        self.policy_be = BehavPolicy(
            policy_class=behav_est, collect_res=True)
        self.lb = lb
        self.lb_delta = lb_delta
        self.is_weight_calculator = ISWeightOrchestrator(
            *is_types, behav_policy=self.policy_be)
        self.is_scorers = is_scorers
        self.gpu = gpu
        self.debug_is_weights = debug_is_weights
    
    def __call__(self, dataset:ReplayBuffer, model:LearnableBase, 
                 discount:float, save_dir:str) -> Dict:
        policy_class = D3RlPyTorchAlgoPredict(predict_func=model.predict)
        eval_policy = D3RlPyDeterministic(policy_class=policy_class, 
                                          gpu=self.gpu, collect_res=True, 
                                          collect_act=True)
        
        states = [torch.Tensor(ep.observations) for ep in dataset.episodes]
        actions = [torch.Tensor(ep.actions) for ep in dataset.episodes]
        rewards = [torch.Tensor(ep.rewards) for ep in dataset.episodes]
        
        if len(actions[0].shape) < 2:
            actions = [torch.Tensor(act).view(-1,1) for act in actions]
        
        self.is_weight_calculator.update(states=states, actions=actions,
            eval_policy=eval_policy
        )
        
        if self.debug_is_weights:
            for is_type in self.is_weight_calculator.is_samplers.keys():
                np.savetxt(
                    os.path.join(save_dir, f"{is_type}_debug.csv"), 
                    self.is_weight_calculator[is_type].traj_is_weights, 
                    delimiter=","
                    )
            np.savetxt(
                os.path.join(save_dir, f"weight_msk_debug.csv"), 
                self.is_weight_calculator.weight_msk, 
                delimiter=","
                ) 
        
        res:Dict[str, float] = {}
        for scr in self.is_scorers:
            tmp_res:torch.Tensor = self.is_scorers[scr].predict(
                rewards=rewards, states=states, actions=actions,
                is_weight_calculator=self.is_weight_calculator, 
                discount=discount,
                )
            res[scr] = tmp_res.cpu().detach().numpy().item()
            if self.lb:
                for lb_key in self.lb.keys():
                    try:
                        logger.debug("Calculating {} bound".format(lb_key))
                        traj_rewards = self.is_scorers[scr].traj_rewards_cache
                        if len(traj_rewards) > 0:    
                            __nm = f"{scr}_{lb_key}"
                            try:
                                res[__nm] = self.lb[lb_key](
                                    traj_rewards.numpy(), self.lb_delta)
                            except ZeroDivisionError as e:
                                res[__nm] = np.nan
                        else:
                            res[__nm] = np.nan
                    except ValueError as e:
                        res[__nm] = np.nan

        logger.info("Saving evaluation outputs")
        with open(os.path.join(save_dir, "eval_policy_acts.pkl"), 
                    "wb") as file:
            pickle.dump(eval_policy.policy_actions, file)
        with open(os.path.join(save_dir, "eval_policy_preds.pkl"), 
                    "wb") as file:
            pickle.dump(eval_policy.policy_predictions, file)
        with open(os.path.join(save_dir, "be_policy_preds.pkl"), 
                    "wb") as file:
            pickle.dump(self.policy_be.policy_predictions, file)    
            
        return res

class FqePipeline(_Evaluator):
    
    def __init__(self, fqe_eval_cls, fqe_scorers, fqe_init_kwargs, 
                 fqe_fit_kwargs, fqe_impl_init, lb:Dict[str,Callable]=None, 
                 lb_delta:float=0.05
                 ) -> None:
        self.fqe_eval_cls = fqe_eval_cls
        self.fqe_scorers = fqe_scorers
        self.fqe_init_kwargs = fqe_init_kwargs
        self.fqe_fit_kwargs = fqe_fit_kwargs
        self.fqe_impl_init = fqe_impl_init
        self.lb = lb
        self.lb_delta = lb_delta
    
    
    def __call__(self, dataset: ReplayBuffer, model, discount:float,
                 save_dir:str) -> Dict:
        self.fqe_init_kwargs["gamma"] = discount
        
        fqe_callback = FQECallback(
            scorers=self.fqe_scorers, fqe_cls=self.fqe_eval_cls, 
            model_init_kwargs=self.fqe_init_kwargs, 
            model_fit_kwargs=self.fqe_fit_kwargs, dataset=dataset, 
            fqe_impl_init=self.fqe_impl_init)
        
        fqe_callback(algo=model, epoch=0, total_step=0)
        res:Dict[str,float] = {}
        for scr in fqe_callback.cache:
            res[scr] = fqe_callback.cache[scr]
        if self.lb:
            for lb_key in self.lb:
                try:
                    #for scr in self.fqe_scorers.keys():
                    scr = "init_state_val"
                    __nm = "{}_loss_{}".format(scr,lb_key)
                    try:
                        res[__nm] = self.lb[lb_key](
                            self.fqe_scorers[scr].cache, self.lb_delta)
                    except ZeroDivisionError as e:
                        res[__nm] = np.nan
                except ValueError as e:
                    logger.error(e)
                    res[__nm] = np.nan
            
        fqe_callback.clean_up()
        return res
        

def get_policy(algo, epoch, checkpoint_dir):
    model_chkp_pnts = [file for file in os.listdir(checkpoint_dir) 
                       if "model_" in file]
    
    model_chkp_pnts = pd.DataFrame({"file_nm":model_chkp_pnts})
    model_chkp_pnts["step"] = model_chkp_pnts["file_nm"].apply(
        lambda x: parse("model_{}.d3", x)[0])
    model_chkp_pnts["step"] = model_chkp_pnts["step"].astype(float)
    model_chkp_pnts = model_chkp_pnts.sort_values(by=["step"], ascending=True)
    model_chkp_pnts["epoch"] = range(1,len(model_chkp_pnts)+1)
    mdl_chk_pnt_file = model_chkp_pnts[
        model_chkp_pnts["epoch"] == epoch]["file_nm"].iloc[0]
    
    mdl_chk_pnt = os.path.join(checkpoint_dir, mdl_chk_pnt_file)
    
    algo = load_learnable(mdl_chk_pnt)
    return algo


def test_ope(
    algo, epoch:int, checkpoint_dir:str, save_dir:str, discount:float, 
    evaluators:List[_Evaluator], dataset:ReplayBuffer
    ):
    algo = get_policy(algo=algo, epoch=epoch, checkpoint_dir=checkpoint_dir)
    res = {}
    for evalu in evaluators:
        res.update(evalu(dataset=dataset, model=algo, discount=discount, 
                         save_dir=save_dir))
    return res