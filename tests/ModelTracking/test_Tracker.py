import unittest
import os
import pandas as pd

from pymlrf.ModelTracking import Tracker

from ..config import TEST_TMP_LOC

# temp_mt_no_exp = pd.DataFrame([{"model_name":"test_exp2", "mse":0.9}])
# temp_mt = pd.DataFrame(
#     [{"model_name":"test_exp2", "mse":0.9},
#      {"model_name":"test_exp1", "mse":0.9}]
#     )

tracker = Tracker(u_id="model_name")

tracker.rows = [
    {"model_name":2,"first_col":1,"second_col":2},
    {"model_name":1,"first_col":1,"second_col":2},
    {"model_name":3,"first_col":1,"second_col":2}
    ]
tracker.column_names = ["model_name","first_col","second_col"]

class TrackerTest(unittest.TestCase):
        
    def test_check_model_exists(self):
        # In tracker and has loc
        assert tracker.check_model_exists(u_id=1)
        assert not tracker.check_model_exists(u_id="1")
        assert not tracker.check_model_exists(u_id=4)
    
    def test_get_cur_row_index(self):
        
        assert tracker.get_cur_row_index(u_id=1) == 1
        assert tracker.get_cur_row_index(u_id=2) == 0
        assert tracker.get_cur_row_index(u_id=3) == 2
        
    def test_check_consistent_col_names(self):
        new_row = {"model_name":1,"first_col":1,"second_col":2}
        new_nms = tracker._get_check_consistent_col_names(
            new_row_col_names=list(new_row.keys())
            )
        assert len(new_nms) == 0
        new_row = {"model_name":1,"first_col":1,"second_col":2, "third_col":4}
        # new_nms = tracker._get_check_consistent_col_names(
        #     new_row_col_names=list(new_row.keys())
        #     )
        new_nms = tracker._get_check_consistent_col_names(
             new_row_col_names=list(new_row.keys()), force_columns=True
             )
        assert len(new_nms) == 1
        
    def test_update_tracker_w_dict(self):
        tracker = Tracker(u_id="model_name")
        tracker.rows = [
            {"model_name":2,"first_col":1,"second_col":2},
            {"model_name":1,"first_col":1,"second_col":2},
            {"model_name":3,"first_col":1,"second_col":2}
            ]
        tracker.column_names = ["model_name","first_col","second_col"]
        new_row = {"model_name":1,"first_col":10,"second_col":20}
        tracker.update_tracker_w_dict(new_row)
        upt_row = tracker.rows[tracker.get_cur_row_index(u_id=1)]
        assert upt_row["model_name"] == 1
        assert upt_row["first_col"] == 10
        assert upt_row["second_col"] == 20
        assert len(tracker.rows) == 3
        
        new_row = {"model_name":4,"first_col":10,"second_col":20}
        tracker.update_tracker_w_dict(new_row)
        upt_row = tracker.rows[tracker.get_cur_row_index(u_id=4)]
        assert upt_row["model_name"] == 4
        assert upt_row["first_col"] == 10
        assert upt_row["second_col"] == 20
        assert len(tracker.rows) == 4
        
    def test_rename_model(self):
        tracker = Tracker(u_id="model_name")
        tracker.rows = [
            {"model_name":2,"first_col":1,"second_col":2},
            {"model_name":1,"first_col":3,"second_col":4},
            {"model_name":3,"first_col":1,"second_col":2}
            ]
        tracker.column_names = ["model_name","first_col","second_col"]
        tracker.rename_model(
            u_id=1, new_u_id=100
        )
        assert tracker.check_model_exists(u_id=100)
        assert not tracker.check_model_exists(u_id=1)
        upt_row = tracker.rows[tracker.get_cur_row_index(u_id=100)]
        assert upt_row["model_name"] == 100
        assert upt_row["first_col"] == 3
        assert upt_row["second_col"] == 4
        assert len(tracker.rows) == 3
    
    

