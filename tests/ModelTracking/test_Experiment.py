import unittest
import os
import pandas as pd

from pymlrf.ModelTracking import Experiment

from ..config import TEST_TMP_LOC

# temp_mt_no_exp = pd.DataFrame([{"model_name":"test_exp2", "mse":0.9}])
# temp_mt = pd.DataFrame(
#     [{"model_name":"test_exp2", "mse":0.9},
#      {"model_name":"test_exp1", "mse":0.9}]
#     )


temp_exp_loc = os.path.join(TEST_TMP_LOC, "test_exp1")
os.mkdir(temp_exp_loc)

class ModelTrackerMoc:
    
    def __init__(self, has_model:bool):
        
        if has_model:
            self.check_model_exists = lambda u_id: True
        else:
            self.check_model_exists = lambda u_id: False
            
    def drop_model(self, u_id):
        return True

temp_mt_no_exp = ModelTrackerMoc(has_model=False) 
temp_mt = ModelTrackerMoc(has_model=True) 
            

class ExperimentTest(unittest.TestCase):
        
    def test_status(self):
        # In tracker and has loc
        exp = Experiment(exp_name="test_exp1", parent_loc=TEST_TMP_LOC, 
                         mt=temp_mt)
        exp.status_check()
        assert exp.is_created is True
        assert exp.in_tracker is True
        
        exp = Experiment(exp_name="test_exp2", parent_loc=TEST_TMP_LOC, 
                         mt=temp_mt)
        exp.status_check()
        assert exp.is_created is False
        assert exp.in_tracker is True
        
        exp = Experiment(exp_name="test_exp1", parent_loc=TEST_TMP_LOC, 
                         mt=temp_mt_no_exp)
        exp.status_check()
        assert exp.is_created is True
        assert exp.in_tracker is False
        
        exp = Experiment(exp_name="test_exp2", parent_loc=TEST_TMP_LOC, 
                         mt=temp_mt_no_exp)
        exp.status_check()
        assert exp.is_created is False
        assert exp.in_tracker is False
    
    def test_delete(self):
        temp_exp_loc = os.path.join(TEST_TMP_LOC, "test_exp2")
        os.mkdir(temp_exp_loc)
        exp = Experiment(exp_name="test_exp2", parent_loc=TEST_TMP_LOC, 
                         mt=temp_mt)
        exp.status_check()
        exp.delete()
        assert not os.path.isdir(temp_exp_loc)
    
    

