import math
from typing import Tuple

__all__ = ["epochs_to_steps"]

def epochs_to_steps(
    epochs:int, 
    n_obs:int, 
    batch_size:int,
    logging_freq:int=1
    ) -> Tuple[int]:
    n_logging_epochs = epochs*logging_freq
    n_steps_per_epoch = int(math.ceil((n_obs/batch_size)))
    n_steps = int(math.ceil(n_steps_per_epoch*epochs))
    n_steps_per_epoch = int(math.ceil(n_steps_per_epoch/logging_freq))
    return n_steps, n_steps_per_epoch, n_logging_epochs