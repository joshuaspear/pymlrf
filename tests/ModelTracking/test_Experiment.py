import unittest
import os
import pandas as pd
import shutil
import time

from pymlrf.ModelTracking import Experiment
from pymlrf.ModelTracking import Tracker

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


def setup_data_for_rename():
    temp_exp_loc = os.path.join(TEST_TMP_LOC, "test_exp_rename")
    os.mkdir(temp_exp_loc)
    os.mkdir(os.path.join(temp_exp_loc, "random_folder_full"))
    with open(
        os.path.join(
            temp_exp_loc, 
            "random_folder_full", 
            "rand_file.txt"
            ),
        "w"
        ) as f:
        f.write("Test text")
    os.mkdir(os.path.join(temp_exp_loc, "random_folder_empty"))
    with open(
        os.path.join(
            temp_exp_loc, 
            "rand_file.txt"
            ),
        "w"
        ) as f:
        f.write("Test text")
    return temp_exp_loc

            

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
        
    def test_rename(self):
        temp_exp_loc = setup_data_for_rename()
        tracker = Tracker(u_id="model_name")
        tracker.rows = [
            {"model_name":"test_exp1","first_col":1,"second_col":2},
            {"model_name":"test_exp_rename","first_col":3,"second_col":4},
            ]
        exp = Experiment(
            exp_name="test_exp_rename",
            parent_loc=TEST_TMP_LOC, 
            mt=tracker
            )
        exp.status_check()
        exp.rename(new_exp_name="test_exp_rename_RENAME")
        assert not os.path.isdir(temp_exp_loc)
        assert os.path.isdir(
            os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME")
        )
        assert exp._exp_name == "test_exp_rename_RENAME"
        assert exp.loc == os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME")
        assert not tracker.check_model_exists("test_exp_rename")
        assert tracker.check_model_exists("test_exp_rename_RENAME")
        shutil.rmtree(os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME"))
        
        temp_exp_loc = setup_data_for_rename()
        tracker = Tracker(u_id="model_name")
        tracker.rows = [
            {"model_name":"test_exp1","first_col":1,"second_col":2}
            ]
        exp = Experiment(
            exp_name="test_exp_rename",
            parent_loc=TEST_TMP_LOC, 
            mt=tracker
            )
        exp.status_check()
        exp.rename(new_exp_name="test_exp_rename_RENAME")
        assert not os.path.isdir(temp_exp_loc)
        assert os.path.isdir(
            os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME")
        )
        assert exp._exp_name == "test_exp_rename_RENAME"
        assert not tracker.check_model_exists("test_exp_rename")
        assert not tracker.check_model_exists("test_exp_rename_RENAME")
        shutil.rmtree(os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME"))
        
        temp_exp_loc = os.path.join(TEST_TMP_LOC, "test_exp_rename")
        tracker = Tracker(u_id="model_name")
        tracker.rows = [
            {"model_name":"test_exp1","first_col":1,"second_col":2}
            ]
        exp = Experiment(
            exp_name="test_exp_rename",
            parent_loc=TEST_TMP_LOC, 
            mt=tracker
            )
        exp.status_check()
        exp.rename(new_exp_name="test_exp_rename_RENAME")
        assert not os.path.isdir(temp_exp_loc)
        assert not os.path.isdir(
            os.path.join(TEST_TMP_LOC, "test_exp_rename_RENAME")
        )
        assert exp._exp_name == "test_exp_rename_RENAME"
        assert not tracker.check_model_exists("test_exp_rename")
        assert not tracker.check_model_exists("test_exp_rename_RENAME")


