from dataclasses import dataclass
from typing import Any, List, Dict
import unittest
import os
import json
import time

from pymlrf.SerialisableConfig.base import SerialisableConfig
from pymlrf.FileSystem import FileHandler

from ..config import sc_config, TEST_TMP_LOC

config_fh = FileHandler(path=sc_config)

class LoadPassWPropertySerialisableConfig(SerialisableConfig):
    
    serialisable_attrs = ["state_space"]
    
    def __init__(self, config_fh = config_fh, state_space=None) -> None:
         super().__init__(config_fh)
         self.state_space = state_space
         
    def decode(self, path:str) -> Dict:
        return {"state_space": "hello world"}
    
    def write(self):
       pass

            

class LoadPassNoPropertySerialisableConfig(SerialisableConfig):

    def decode(self, path:str) -> Dict:
        return {"state_space": "hello world"}
    
    def write(self):
       pass


class SerialisableConfigTest(unittest.TestCase):

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