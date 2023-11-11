import d3rlpy
import pathlib
import os
import torch
from d3rlpy.dataset import BasicTransitionPicker
from d3rlpy.datasets import get_cartpole
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from d3rlpy.logging import FileAdapterFactory

from offline_rl_ope.api.d3rlpy import (ISCallback, ISEstimatorScorer)
from offline_rl_ope.components.Policy import BehavPolicy

from pymlrf.api.d3rlpy import (epochs_to_steps)
from pymlrf.ModelTracking import Experiment, Option, SerialisedTracker
from pymlrf.utils import set_seed

HERE = pathlib.Path(__file__).parent
root_dir = os.path.join(HERE, "tmp_rl_train_example")
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)


def training_loop(
    epochs, 
    n_obs, 
    batch_size, 
    gbt_policy_be, 
    dataset,
    gamma,
    algo,
    save_dir,
    seed=1
    ):
    set_seed(seed)
    # start training
    n_steps, n_steps_per_epoch = epochs_to_steps(
        epochs=epochs, n_obs=n_obs, batch_size=batch_size)
    
    # Get callbacks
    is_callback = ISCallback(
        is_types=["vanilla", "per_decision"],
        behav_policy=gbt_policy_be,
        dataset=dataset,
        eval_policy_kwargs = {
            "gpu": False, 
            "collect_act":True,
            "collect_res":False
        }
        )
    
    # Get scorers
    scorers = {}

    scorers.update({"vanilla_wis_loss": ISEstimatorScorer(
        discount=gamma, cache=is_callback, is_type="vanilla", 
        norm_weights=True)})

    scorers.update({"pd_wis_loss": ISEstimatorScorer(
        discount=gamma, cache=is_callback, is_type="per_decision", 
        norm_weights=True)})

    res = algo.fit(
        dataset=dataset, n_steps=n_steps, 
        n_steps_per_epoch=n_steps_per_epoch,
        logger_adapter=FileAdapterFactory(root_dir=save_dir),
        experiment_name="rl_train_example", 
        with_timestamp=False, 
        evaluators=scorers, 
        epoch_callback=is_callback
        )
    out_dict = {
        key:(value.item() if isinstance(value, torch.Tensor) else value)
        for key,value in res[-1][-1].items()
        }
    return out_dict 


def main():
    dataset = "hopper-medium-v0"

    # fix seed
    d3rlpy.seed(1)
    dataset, env = get_cartpole()
    gamma = 0.99
    # setup algorithm
    batch_size=256
    sac = d3rlpy.algos.DQNConfig(
        batch_size=batch_size,
        learning_rate=3e-4
    ).create()
    
    cql = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=3e-4
    ).create()


    
    class GbtEst:
    
        def __init__(self, estimator:MultiOutputClassifier) -> None:
            self.estimator = estimator
        
        def eval_pdf(self, indep_vals:np.array, dep_vals:np.array):
            probs = self.estimator.predict_proba(X=indep_vals)
            res = []
            for i,out_prob in enumerate(probs):
                tmp_res = out_prob[
                    np.arange(len(out_prob)),
                    dep_vals[:,i].squeeze().astype(int)
                    ]
                res.append(tmp_res.reshape(1,-1))
            res = np.concatenate(res, axis=0).prod(axis=0)
            return res

    behav_est = MultiOutputClassifier(OneVsRestClassifier(XGBClassifier(
        objective="binary:logistic")))

    # Fit the behaviour model
    observations = []
    actions = []
    tp = BasicTransitionPicker()
    for ep in dataset.episodes:
        for i in range(ep.transition_count):
            _transition = tp(ep,i)
            observations.append(_transition.observation.reshape(1,-1))
            actions.append(_transition.action)

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)

    behav_est.fit(X=observations, Y=actions.reshape(-1,1))

    gbt_est = GbtEst(estimator=behav_est)
    gbt_policy_be = BehavPolicy(policy_class=gbt_est, collect_res=False)

    tracker_path=os.path.join(root_dir,"model_tracker.json")
    tracker = SerialisedTracker(path=tracker_path)
    if tracker.is_created:
        tracker.read()
    else:
        print("Tracker not yet created at location")
    
    exp_1 = Experiment(exp_name="dqn_exp", parent_loc=root_dir, mt=tracker)
    exp_1.status_check()
    exp_2 = Experiment(exp_name="cql_exp", parent_loc=root_dir, mt=tracker)
    exp_2.status_check()
    
    epochs = 2
    n_obs = 100000
    
    exp_1.run(
        option=Option("overwrite"),
        func = training_loop,
        epochs=epochs, 
        n_obs=n_obs, 
        batch_size=batch_size, 
        gbt_policy_be=gbt_policy_be, 
        dataset=dataset,
        gamma=gamma,
        algo=sac,
        save_dir=exp_1.loc
        )
    
    exp_2.run(
        option=Option("overwrite"),
        func = training_loop,
        epochs=epochs, 
        n_obs=n_obs, 
        batch_size=batch_size, 
        gbt_policy_be=gbt_policy_be, 
        dataset=dataset,
        gamma=gamma,
        algo=cql,
        save_dir=exp_2.loc
    )

    tracker_df = tracker.tracker_to_pandas_df()
    print(tracker_df.head())
    tracker.write()
    
    exp_1.run(
        option=Option("error"),
        func = training_loop,
        epochs=epochs, 
        n_obs=n_obs, 
        batch_size=batch_size, 
        gbt_policy_be=gbt_policy_be, 
        dataset=dataset,
        gamma=gamma,
        algo=sac,
        save_dir=exp_1.loc
        )
    
    exp_2.run(
        option=Option("error"),
        func = training_loop,
        epochs=epochs, 
        n_obs=n_obs, 
        batch_size=batch_size, 
        gbt_policy_be=gbt_policy_be, 
        dataset=dataset,
        gamma=gamma,
        algo=cql,
        save_dir=exp_2.loc
    )

    tracker_df = tracker.tracker_to_pandas_df()
    print(tracker_df.head())
    tracker.write()


if __name__ == "__main__":
    main()    