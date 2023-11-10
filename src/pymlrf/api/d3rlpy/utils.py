import math
from typing import Tuple

__all__ = ["epochs_to_steps"]

def epochs_to_steps(epochs:int, n_obs:int, batch_size:int) -> Tuple[int]:
    n_steps_per_epoch = int(math.ceil(n_obs/batch_size))
    n_steps = int(math.ceil(n_steps_per_epoch*epochs))
    return n_steps, n_steps_per_epoch