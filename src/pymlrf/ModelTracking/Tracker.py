from abc import ABCMeta, abstractmethod
import logging
import os
from typing import Any, Callable, Dict, List
import time
import pandas as pd
import numpy as np
try:
    import sqlite3
except ImportError as e:
    pass
from contextlib import closing
try:
    import psycopg2
except ImportError as e:
    pass

from ..FileSystem import FileHandler

logger = logging.getLogger("pymlrf")

__all__ = [
    "TrackerBase",
    "Tracker",
    "SerialisedTracker",
    "SQLiteTracker",
    "PostgresTracker"
]

class TrackerBase(metaclass=ABCMeta):
    
    _error_value:str = "RUN_ERROR"
    
    def __init__(self, u_id:str="model_name"):
        self.column_names:List[str] = []
        self.u_id:str = u_id
    
    
    def write_run_error(self, u_id:str, overwrite:bool=True):
        if self.check_model_exists(u_id=u_id):
            if overwrite:
                row_dict = {col:self._error_value for col in self.column_names}
                row_dict[self.u_id] = u_id
                self.update_tracker_w_dict(row_dict=row_dict)
    
    def get_check_consistent_col_names(
        self, 
        new_row_col_names:list, 
        force_columns:bool=False
        ) -> set[str]:
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
    
    @abstractmethod    
    def write_u_id(
        self, 
        u_id_update:Any
        ):
        """Writes a column to each row named self.u_id according to the function
        provided in u_id_update. Useful when a tracker has previously been 
        defined with a different u_id to the one required.

        Args:
            u_id_update (Callable): Function which takes in a individual values
            of self.row i.e. a Dict[str,Any]. Should return an Any.
        """
        pass
    
    @abstractmethod
    def update_tracker_w_dict(
        self, 
        row_dict:dict, 
        force_columns:bool=False
        ):
        pass
    
    @abstractmethod
    def check_model_exists(self, u_id:str):
        pass
    
    @abstractmethod
    def drop_model(
        self, 
        u_id:str
        ):
        pass
    
    @abstractmethod
    def rename_model(
        self, 
        u_id:str, 
        new_u_id:str
        ):
        pass
    

class Tracker(TrackerBase):
    
    def __init__(self, u_id:str="model_name"):
        """Class representing a 'model tracker'. 
        self.rows is a list dictionaries where each dictionary is of the form 
        {column_name: value} and each dictionary represents an individual 
        experiment
        self.column_names is a list of unique column_name value from self.rows
        """
        TrackerBase.__init__(self,u_id=u_id)
        self.rows:List[Dict[str,Any]] = []
                
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
            logger.warning(
                "Model already exists in tracker, overwriting relevant values")
            logger.debug(f"Inserting row with u_id: {row_dict}")
            old_row = self.rows.pop(row_idx)
            for i in row_dict:
                old_row[i] = row_dict[i]
            row_dict = old_row
        except KeyError as e:
            pass     
        new_row_col_names = [col for col in row_dict.keys()]
        nw_col_names = self.get_check_consistent_col_names(
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
    
    def __init__(self, path:str, u_id:str="model_name", safe_write:bool=True):
        FileHandler.__init__(self, path=path)
        Tracker.__init__(self, u_id=u_id)
        self.safe_write = safe_write
    
    def write(self, **kwargs):
        """Saves the tracker i.e. values in self.rows as a json. This is 
        performed via pandas. kwargs should contain options defined in 
        pd.DataFrame.to_json()

        Args:
            json_dir (str): File location of where to save the output json
        """
        dict_df = self.tracker_to_pandas_df()
        tracker_exists = os.path.exists(self.path)
        run_safe_write = tracker_exists and self.safe_write
        if run_safe_write:
            path_split = self.path.split(".")
            backup_root = ".".join(path_split[:-1])
            extention = path_split[-1]
            backup_file_pth = f"{backup_root}_TEMP.{extention}"
            os.rename(self.path, backup_file_pth)
        dict_df.to_json(self.path, **kwargs)
        if run_safe_write:
            path_exists = os.path.exists(self.path)
            while not path_exists:
                logger.info("Tracker not fully written waiting before cleanup")
                time.sleep(1)
                path_exists = os.path.exists(self.path)
            os.remove(backup_file_pth)

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

def format_sql_value(value:Any)->str:
    if isinstance(value, bool):
        if value:
            value = "TRUE"
        else:
            value = "FALSE"
    elif isinstance(value, str):
        value = f"'{value}'"
    elif isinstance(value, int):
        value = f"{float(value)}"
    elif isinstance(value, float):
        value = f"{value}"
    else:
        value = f"'{str(value)}'"
        # raise ValueError(f"{value}, {type(value)}")
    return value

class SQLiteTracker(TrackerBase, FileHandler):
    
    def __init__(
        self, 
        path:str, 
        u_id:str="model_name"
        ):
        FileHandler.__init__(self, path=path)
        TrackerBase.__init__(self, u_id=u_id)
        assert path.split(".")[-1] == "db", f"save file must be a db"
        self.__valid = False
        if not self.is_created:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"CREATE TABLE tracker({self.u_id})")
            con.commit()
            cur.close()
            con.close()
        else:
            con = self.get_connection()
            cur = con.cursor()
            res = cur.execute("SELECT name FROM sqlite_master")
            con.commit()
            cur.close()
            con.close()
            if res is None:
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"CREATE TABLE tracker({self.u_id})")
                con.commit()
                cur.close()
                con.close()
        con = self.get_connection()
        cur = con.cursor()
        res = cur.execute("SELECT name FROM sqlite_master")
        con.commit()
        cur.close()
        con.close()
        assert "tracker" in res
        # self.__con = sqlite3.connect(self.path)
    
    def get_connection(self):
        return sqlite3.connect(self.path)
    
    def read(self):
        try:
            con = self.get_connection()
            cur = con.cursor()
            res = cur.execute("PRAGMA table_info('tracker')")
            out_names = [
                "cid", "name", "type", "notnull", "dflt_value", "pk"
                ]
            output = [
                {_k:_v for _k,_v in zip(out_names, _rw)} 
                for _rw in res.fetchall()
            ]
            con.commit()
            cur.close()
            con.close()
            self.column_names = [_rw["name"] for _rw in output]
            self.__valid = True
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
    
    def get_current_experiments(self)->List[str]:
        assert self.__valid, f"Run .read() first!"
        if len(self.column_names) > 0:
            try:
                con = self.get_connection()
                cur = con.cursor()
                res = cur.execute(f"SELECT {self.u_id} FROM tracker")
                res = [i[0] for i in res.fetchall()]
                con.commit()
                cur.close()
                con.close()
            except Exception as e:
                try:
                    cur.close()
                except NameError as _e:
                    pass
                try:
                    con.close()
                except NameError as _e:
                    pass
                raise e
            return res
        else:
            return []
        
    def check_model_exists(self, u_id:str):
        return u_id in self.get_current_experiments()
        
                    
    def write_u_id(self, u_id_update:str):
        """Writes a column to each row named self.u_id according to the function
        provided in u_id_update. Useful when a tracker has previously been 
        defined with a different u_id to the one required.

        Args:
            u_id_update (Callable): Function which takes in a individual values
            of self.row i.e. a Dict[str,Any]. Should return an Any.
        """
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"""
                ALTER TABLE tracker
                ADD {u_id_update};
            """)
            con.commit()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e

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
            new_row_col_names = [col for col in row_dict.keys()]
            nw_col_names = self.get_check_consistent_col_names(
                new_row_col_names=new_row_col_names, force_columns=force_columns)
            if len(nw_col_names) > 0:
                self.column_names += nw_col_names
                for _col in nw_col_names:
                    con = self.get_connection()
                    cur = con.cursor()
                    cur.execute(f"""
                        ALTER TABLE tracker
                        ADD {_col};
                    """)
                    con.commit()
                    cur.close()
                    con.close()
            if self.check_model_exists(u_id=row_dict[self.u_id]):
                logger.warning(
                    "Model already exists in tracker, overwriting relevant values")
                logger.debug(f"Inserting row with u_id: {row_dict[self.u_id]}")
                sql_row_fmt = []
                for _k in row_dict.keys():
                    if _k != self.u_id:
                        value = format_sql_value(row_dict[_k])
                        sql_row_fmt.append(
                            f"{_k} = {value}"
                        )
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"""
                    UPDATE tracker 
                    SET {", ".join(sql_row_fmt)}
                    WHERE {self.u_id} = '{row_dict[self.u_id]}';
                """)
                con.commit()
                cur.close()
                con.close()
            else:
                _col_nms_to_set = []
                _vals_to_set = []
                for _k in row_dict.keys():
                    _col_nms_to_set.append(_k)
                    _vals_to_set.append(format_sql_value(row_dict[_k]))
                for _col in self.column_names:
                    if _col not in _col_nms_to_set:
                        _col_nms_to_set.append(_col)
                        _vals_to_set.append("NULL")
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"""
                    INSERT INTO tracker {', '.join(_col_nms_to_set)}
                    VALUES {', '.join(_vals_to_set)};
                """)
                con.commit()
                cur.close()
                con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
        
    def drop_model(
        self, 
        u_id:str
        ):
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(
                f"DELETE FROM tracker WHERE {self.u_id} = '{u_id}';"
                )
            con.commit()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
    
    def rename_model(
        self, 
        u_id:str, 
        new_u_id:str
        ):
        raise NotImplementedError
    
def get_format(value: Any)->str:
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, float):
        return "double precision"
    elif isinstance(value, int):
        return "double precision"
    else:
        return "text"

class PostgresTracker(TrackerBase):
    
    def __init__(
        self,
        connection_str:str,
        table_name:str,
        table_schema:str = "tracking",
        u_id="model_name"
        ):
        assert table_name.islower()
        assert table_schema.islower()
        super().__init__(u_id)
        self.connection_str = connection_str
        self.table_name = table_name
        self.table_schema = table_schema
        
        # Create schema
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {table_schema};")
            con.commit()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
        
        # Check whether table exists
        _tbl_exists = self.check_table_exists()
        
        if not _tbl_exists:
            try:
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"CREATE TABLE IF NOT EXISTS {self.table_schema}.{self.table_name} ({self.u_id} TEXT PRIMARY KEY);")
                con.commit()
                cur.close()
                con.close()
            except Exception as e:
                try:
                    cur.close()
                except NameError as _e:
                    pass
                try:
                    con.close()
                except NameError as _e:
                    pass
                raise e
        
        self.column_names = self.get_column_names(
            schema = self.table_schema,
            table = self.table_name
        )
            
            
    def check_table_exists(self)->bool:
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema = '{self.table_schema}' AND table_name = '{self.table_name}'
            """)
            avail_tbls = cur.fetchall()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
        col_names = ["table_schema", "table_name"]
        avail_tbls = [
            {_k:_v for _k, _v in zip(col_names, avail_tbls[i])} 
            for i in range(len(avail_tbls))
        ]
        
        if len(avail_tbls) > 0:
            assert len(avail_tbls) == 1, f"Duplicate tables in database: {avail_tbls}"
            _tbl_exists = True
        else:
            _tbl_exists = False
        return _tbl_exists
    
    
    def get_connection(self)->psycopg2.extensions.connection:
        return psycopg2.connect(self.connection_str)
    
    
    def get_column_names(self, schema:str, table:str)->List[str]:
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{self.table_schema}' AND table_name = '{self.table_name}'
            """)
            result = cur.fetchall()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
        return [i[0] for i in result]
            
            
    def get_current_experiments(self)->List[str]:
        self.column_names = self.get_column_names(
            schema = self.table_schema,
            table = self.table_name
        )
        if len(self.column_names) > 0:
            try:
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"SELECT {self.u_id} FROM {self.table_schema}.{self.table_name}")
                res = cur.fetchall()
                cur.close()
                con.close()
                res = [i[0] for i in res]
            except Exception as e:
                try:
                    cur.close()
                except NameError as _e:
                    pass
                try:
                    con.close()
                except NameError as _e:
                    pass
                raise e
            return res
        else:
            return []
        
    def check_model_exists(self, u_id:str):
        return u_id in self.get_current_experiments()
        
                    
    def write_u_id(self, u_id_update:str):
        """Writes a column to each row named self.u_id according to the function
        provided in u_id_update. Useful when a tracker has previously been 
        defined with a different u_id to the one required.

        Args:
            u_id_update (Callable): Function which takes in a individual values
            of self.row i.e. a Dict[str,Any]. Should return an Any.
        """
        assert isinstance(u_id_update, str)
        assert u_id_update.islower()
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(f"""
                ALTER TABLE {self.table_schema}.{self.table_name}
                ADD {u_id_update} TEXT;
            """)
            con.commit()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e

    def update_tracker_w_dict(self, row_dict:Dict[str,Any], force_columns:bool=False):
        """Updates the self.rows and self.column_names with the new values 
        provided in row_dict

        Args:
            row_dict (dict): dictionary containing {column:values} to be added 
            to the tracker
            force_columns (bool, optional): Option to force new column names in 
            and avoid exception. Defaults to False.
        """
        self.column_names = self.get_column_names(
            schema = self.table_schema,
            table = self.table_name
        )
        try:
            new_row_col_names = [col for col in row_dict.keys()]
            nw_col_names = self.get_check_consistent_col_names(
                new_row_col_names=new_row_col_names, force_columns=force_columns)
            if len(nw_col_names) > 0:
                self.column_names += nw_col_names
                for _col in nw_col_names:
                    assert _col.islower()
                    con = self.get_connection()
                    cur = con.cursor()
                    _val_type = get_format(row_dict[_col])
                    cur.execute(f"""
                        ALTER TABLE {self.table_schema}.{self.table_name}
                        ADD {_col} {_val_type};
                    """)
                    con.commit()
                    cur.close()
                    con.close()
            if self.check_model_exists(u_id=row_dict[self.u_id]):
                logger.warning(
                    "Model already exists in tracker, overwriting relevant values")
                logger.debug(f"Inserting row with u_id: {row_dict[self.u_id]}")
                sql_row_fmt = []
                for _k in row_dict.keys():
                    if _k != self.u_id:
                        assert _k.islower()
                        value = format_sql_value(row_dict[_k])
                        sql_row_fmt.append(
                            f"{_k} = {value}"
                        )
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"""
                    UPDATE {self.table_schema}.{self.table_name} 
                    SET {", ".join(sql_row_fmt)}
                    WHERE {self.u_id} = '{row_dict[self.u_id]}';
                """)
                con.commit()
                cur.close()
                con.close()
            else:
                _col_nms_to_set = []
                _vals_to_set = []
                for _k in row_dict.keys():
                    assert _k.islower()
                    _col_nms_to_set.append(_k)
                    _vals_to_set.append(format_sql_value(row_dict[_k]))
                for _col in self.column_names:
                    assert _col.islower()
                    if _col not in _col_nms_to_set:
                        _col_nms_to_set.append(_col)
                        _vals_to_set.append("NULL")
                con = self.get_connection()
                cur = con.cursor()
                cur.execute(f"""
                    INSERT INTO {self.table_schema}.{self.table_name} ({', '.join(_col_nms_to_set)})
                    VALUES ({', '.join(_vals_to_set)});
                """)
                con.commit()
                cur.close()
                con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
        
    def drop_model(
        self, 
        u_id:str
        ):
        try:
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(
                f"DELETE FROM {self.table_schema}.{self.table_name} WHERE {self.u_id} = '{u_id}';"
                )
            con.commit()
            cur.close()
            con.close()
        except Exception as e:
            try:
                cur.close()
            except NameError as _e:
                pass
            try:
                con.close()
            except NameError as _e:
                pass
            raise e
    
    def rename_model(
        self, 
        u_id:str, 
        new_u_id:str
        ):
        raise NotImplementedError