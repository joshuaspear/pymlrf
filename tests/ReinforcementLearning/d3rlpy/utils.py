import math
from typing import Tuple

import unittest
import os

from pymlrf.ReinforcementLearning.d3rlpy.utils import epochs_to_steps

class EpochsToStepsTest(unittest.TestCase):
    
    def test_epochs_to_steps_1(self):
        epochs = 10
        n_obs = 1000
        batch_size = 4
        logging_freq = 1
        p_n_steps, p_n_steps_per_epoch, p_n_logging_epochs = epochs_to_steps(
            epochs=epochs,
            n_obs=n_obs,
            batch_size=batch_size,
            logging_freq=logging_freq
            )
        assert p_n_steps == 2500
        assert p_n_steps_per_epoch == 250
        assert p_n_logging_epochs == 10
    
    def test_epochs_to_steps_2(self):
        epochs = 10
        n_obs = 1000
        batch_size = 4
        logging_freq = 2
        p_n_steps, p_n_steps_per_epoch, p_n_logging_epochs = epochs_to_steps(
            epochs=epochs,
            n_obs=n_obs,
            batch_size=batch_size,
            logging_freq=logging_freq
            )
        assert p_n_steps == 2500
        assert p_n_steps_per_epoch == 250
        assert p_n_logging_epochs == 20