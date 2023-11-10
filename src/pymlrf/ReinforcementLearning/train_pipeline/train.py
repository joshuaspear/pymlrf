import logging
from typing import Any, Callable, Dict, List
import math

from d3rlpy.dataset import ReplayBuffer
from d3rlpy.logging import FileAdapterFactory
from d3rlpy.algos import QLearningAlgoBase

logger = logging.getLogger("pymlrf")


def train_offline(
      algo:QLearningAlgoBase, 
      n_obs:int, 
      batch_size:int, 
      epochs:int, 
      train_data:ReplayBuffer, 
      model_sv_loc:str,
      *args,
      **kwargs
      ):
    # Total number of training steps: (n_obs/batch_size)*n_epochs
    n_steps_per_epoch = int(math.ceil(n_obs/batch_size))
    n_steps = int(math.ceil(n_steps_per_epoch*epochs))
    algo.fit(train_data, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch,
             logger_adapter=FileAdapterFactory(root_dir=model_sv_loc),
             experiment_name="d3rlpy_out", with_timestamp=False
             )
    return True

