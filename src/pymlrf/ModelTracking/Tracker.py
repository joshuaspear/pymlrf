import logging
import os
from typing import Any, Callable, Dict, List

import pandas as pd

from ..FileSystem import FileHandler

logger = logging.getLogger("pymlrf")

__all__ = [
    "Tracker",
    "SerialisedTracker"
]


class Tracker:
    
    _error_value:str = "RUN_ERROR"
    
    def __init__(self, u_id:str="model_name"):
        """Class representing a 'model tracker'. 
        self.rows is a list dictionaries where each dictionary is of the form 
        {column_name: value} and each dictionary represents an individual 
        experiment
        self.column_names is a list of unique column_name value from self.rows
        """
        self.rows:List[Dict[str,Any]] = []
        self.column_names:List[str] = []
        self.u_id:str = u_id
        
    def _get_check_consistent_col_names(self, new_row_col_names:list, 
                                        force_columns:bool=False) -> set:
        """Method is used check whether the column names provided in 
        new_row_col_names are consistent with the current names housed in 
        self.column_names. The method raises an exception if the new column 
        names don't align unless force_columns is set to True, in which case a 
        warning is provided.
        

        Args:
            new_row_col_names (list): List of column names to check
            force_columns (bool, optional): Option to force new column names in 
            and avoid exception. Defaults to False.

        Returns:
            set: The column names provided in new_row_col_names that are not 
            already in self.column_names
        """
        miss_frm_exst_col = set(self.column_names) - set(new_row_col_names)
        miss_frm_exst_col_wrn = "Column names {} already exist in the tracker however are missing from the new row".format(miss_frm_exst_col)
        nw_col_names = set(new_row_col_names) - set(self.column_names)
        nw_col_names_wrn = "{} are new column names in the row that are not already in the tracker".format(nw_col_names)
        if len(self.column_names) > 0:
            # Only check if column names are not empty
            if force_columns:
                if len(miss_frm_exst_col) > 0:
                    logger.warning(miss_frm_exst_col_wrn)
                if len(nw_col_names) > 0:
                    logger.warning(nw_col_names_wrn)
            else:
                assert(len(miss_frm_exst_col) == 0), miss_frm_exst_col_wrn
                assert(len(nw_col_names) == 0), nw_col_names_wrn
        return nw_col_names
    
    def write_run_error(self, u_id:str, overwrite:bool=True):
        if self.check_model_exists(u_id=u_id):
            if overwrite:
                row_dict = {col:self._error_value for col in self.column_names}
                row_dict[self.u_id] = u_id
                self.update_tracker_w_dict(row_dict=row_dict)
                
    def write_u_id(self, u_id_update:Callable):
        """Writes a column to each row named self.u_id according to the function
        provided in u_id_update. Useful when a tracker has previously been 
        defined with a different u_id to the one required.

        Args:
            u_id_update (Callable): Function which takes in a individual values
            of self.row i.e. a Dict[str,Any]. Should return an Any.
        """
        for row in self.rows:
            try:
                row[self.u_id]
                if row[self.u_id] is None:
                    row[self.u_id] = u_id_update(row)    
            except KeyError as e:
                row[self.u_id] = u_id_update(row)

    def update_tracker_w_dict(self, row_dict:dict, force_columns:bool=False):
        """Updates the self.rows and self.column_names with the new values 
        provided in row_dict

        Args:
            row_dict (dict): dictionary containing {column:values} to be added 
            to the tracker
            force_columns (bool, optional): Option to force new column names in 
            and avoid exception. Defaults to False.
        """
        try:
            row_idx = self.get_cur_row_index(u_id=row_dict[self.u_id])
            logger.warn(
                "Model already exists in tracker, overwriting relevant values")
            logger.debug(f"Inserting row with u_id: {row_dict}")
            old_row = self.rows.pop(row_idx)
            for i in row_dict:
                old_row[i] = row_dict[i]
            row_dict = old_row
        except KeyError as e:
            pass     
        new_row_col_names = [col for col in row_dict.keys()]
        nw_col_names = self._get_check_consistent_col_names(
            new_row_col_names=new_row_col_names, force_columns=force_columns)
        if len(nw_col_names) > 0:
            self.column_names += nw_col_names
        self.rows.append(row_dict)
        
    def check_model_exists(self, u_id:str):
        curr_model_nms = [rw[self.u_id] for rw in self.rows]
        res = u_id in curr_model_nms
        return res
    
    def get_cur_row_index(self, u_id:str):
        for idx, rw in enumerate(self.rows):
            if rw[self.u_id] == u_id:
                return idx
        raise KeyError("Model does not exist in tracker")

    def tracker_to_pandas_df(self)->pd.DataFrame:
        """Converts the values stored in self.rows and returns in the form of a 
        dataframe

        Returns:
            pd.DataFrame: Dataframe containing the values in self.rows
        """
        dict_df = pd.DataFrame.from_dict(self.rows)
        return dict_df
    
    def drop_model(self, u_id:str):
        curr_model_nms = [rw[self.u_id] for rw in self.rows]
        dupe_indices = [idx for idx, mdl_nm in enumerate(curr_model_nms) 
                        if mdl_nm == u_id]
        dupe_indices.sort(reverse=True)
        for idx in dupe_indices:
            del self.rows[idx]
            
    def rename_model(self, u_id:str, new_u_id:str):
        curr_model_nms = [rw[self.u_id] for rw in self.rows]
        dupe_indices = [idx for idx, mdl_nm in enumerate(curr_model_nms) 
                        if mdl_nm == u_id]
        dupe_indices.sort(reverse=True)
        for idx in dupe_indices:
            self.rows[idx][self.u_id] = new_u_id
        
    def import_existing_pandas_df_tracker(
        self, exstng_track_df:pd.DataFrame, **kwargs
        ):
        """Takes as an input a dataframe representing and model tracker and 
        updates self with values from the dataframe. kwargs should refer to 
        updating options defined in self.update_tracker_w_dict
        

        Args:
            exstng_track_df (pd.DataFrame): Pandas dataframe representing a 
            model tracker
        """
        exstng_track_dict = exstng_track_df.to_dict("records")
        for row in exstng_track_dict:
            self.update_tracker_w_dict(row, **kwargs)


class SerialisedTracker(Tracker, FileHandler):
    
    def __init__(self, path:str, u_id:str="model_name"):
        FileHandler.__init__(self, path=path)
        Tracker.__init__(self, u_id=u_id)
    
    def write(self, **kwargs):
        """Saves the tracker i.e. values in self.rows as a json. This is 
        performed via pandas. kwargs should contain options defined in 
        pd.DataFrame.to_json()

        Args:
            json_dir (str): File location of where to save the output json
        """
        dict_df = self.tracker_to_pandas_df()
        dict_df.to_json(self.path, **kwargs)

    def read(
        self, 
        imprt_kwargs:dict = {}, 
        rd_json_kwargs:dict = {}
        ):
        """Takes as an input a json representing and model tracker and updates 
        self with values from the json. This is performed via pandas.

        Args:
            existing_tracker_path (str): File location of the csv tracker
            imprt_kwargs (dict, optional): kwargs to provide to 
            self.import_existing_pandas_df_tracker. Defaults to {}.
            rd_json_kwargs (dict, optional): kwargs to provide to pd.read_json. 
            Defaults to {}.
        """
        exstng_track_df = pd.read_json(self.path, **rd_json_kwargs)
        self.import_existing_pandas_df_tracker(exstng_track_df, **imprt_kwargs)