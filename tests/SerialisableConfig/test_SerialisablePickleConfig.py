from dataclasses import dataclass
from typing import Any, List, Dict
import unittest
import os
import pickle

from pymlrf.SerialisableConfig import SerialisablePickleConfig 
from pymlrf.FileSystem import FileHandler

from ..config import sc_pickle_config, TEST_TMP_LOC

config_fh = FileHandler(path=sc_pickle_config)

class LoadPassWPropertySerialisableConfig(SerialisablePickleConfig):
    
    serialisable_attrs = ["state_space"]
    
    def __init__(self, config_fh = config_fh, state_space=None) -> None:
         super().__init__(config_fh)
         self.state_space = state_space

            

class LoadPassNoPropertySerialisableConfig(SerialisablePickleConfig):
    pass


class SerialisablePickleConfigTest(unittest.TestCase):

    def test_load_failure_no_property(self):
        dc = LoadPassNoPropertySerialisableConfig(
            config_fh=config_fh
        )
        with self.assertRaises(AttributeError) as context:
            dc.read()
        
    def test_load_pass_w_property(self):
        dc = LoadPassWPropertySerialisableConfig(
            config_fh=config_fh
        )
        dc.read()
        self.assertTrue(dc.state_space == "hello world")
    
    def test_write(self):
        temp_config = os.path.join(TEST_TMP_LOC, "sc_config_2.pkl")
        config_fh = FileHandler(path=temp_config)
        dc = LoadPassWPropertySerialisableConfig(
            config_fh=config_fh,
            state_space="test_state_space"
            )
        dc.write()
        
        with open(temp_config, "rb") as f:
            config = pickle.load(f)
        
        self.assertTrue(config["state_space"] == "test_state_space")
