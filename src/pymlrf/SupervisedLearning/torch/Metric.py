import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List

__all__ = [
    "Metric", 
    "MetricOrchestrator", 
    "mean_trans",
    "sum_trans"
]

# Metric class 
class Metric:
    
    def __init__(self, name:str) -> None:
        """Class for tracking the value of metric throughout training. The raw
        values of the metric are stored in self.value_dict, the transformations
        that should be applied to the metric on a rolling basis are stored in 
        self.roll_trans and the results of the rolling transformations are 
        stored in self.roll_trans_values

        Args:
            name (str): Name of the metric i.e. its identifier
        """
        self.name = name
        self.value_dict = {}
        self.roll_trans = {}
        self.roll_trans_values = {}
        
    def add_value(self, label:str, value:Any)->None:
        """Method to log a new value with the tracker. Any rolling 
        transformations are also applied each time a new value is logged. 

        Args:
            label (str): Identifier for the metric value for example "epoch_1"
            value (Any): The specific value of the metric for this update
        """
        self.value_dict[label] = value
        for trans_lab in self.roll_trans:
            self.roll_trans_values[trans_lab][label] = self.roll_trans[
                trans_lab](list(self.value_dict.values()))
        
    def add_roll_trans(
        self, 
        label:str, 
        trans:Callable
        )->None:
        """Method to add a new transformation which is applied to metrics on 
        a rolling basis. Note all transformations should be logged BEFORE
        values are tracked.

        Args:
            label (str): The identifier of the transformation.
            trans (Callable): A function/lambda which takes a list of numeric 
            values as an input
        """
        self.roll_trans[label] = trans
        self.roll_trans_values[label] = {}
        
    def values_to_df(self)->pd.DataFrame:
        """Compiles the raw metric values and transformed values into a pandas
        dataframe

        Returns:
            pd.DataFrame: Pandas dataframe with the raw values, transformed 
            values, a column for the name of the metric, with the index as the 
            labels of the values in self.value_dict
        """
        df_out = pd.DataFrame(self.roll_trans_values)
        df_out["raw_vals"] = pd.Series(self.value_dict)
        df_out["metric_name"] = self.name
        return df_out


# Orchestrator

class MetricOrchestrator:
    
    def __init__(self) -> None:
        """Class for handling the update of multiple metrics simultaneously
        """
        self.metrics:Dict[str,Metric] = {}
        
    def setup_orchestrator(
        self, 
        name_trans_dict:Dict[str, Dict[str,Callable]]
        ) -> None:
        """Method for specifying what metrics to be tracked along with 
        associated meta data

        Args:
            name_trans_dict (Dict[str, Callable]): A dictionary of the form 
            {*metric_name*: {*transformation_name*: *transformation_callable*}}
        """
        for metric in name_trans_dict:
            self.add_metric(
                nm=metric,
                rll_trans=name_trans_dict[metric]
                )
    
    def add_metric(self, nm, rll_trans) -> None:
        self.metrics[nm] = Metric(nm)
        if len(rll_trans) > 0:
            for trans in rll_trans:
                self.metrics[nm].add_roll_trans(
                    trans, rll_trans[trans])
    
    def update_metrics(self, metric_value_dict:Dict[str, Dict[str, Any]])->None:
        """Method for updating multiple metrics simulaneously

        Args:
            metric_value_dict (Dict[str, Dict[str, Any]]): A dictionary 
            containing the relevant update values of the form:
            {*metric_name*:{"label": *value_label*, "value": *value_value*}}
        """
        for metric in metric_value_dict:
            self.metrics[metric].add_value(
                metric_value_dict[metric]["label"], 
                metric_value_dict[metric]["value"])
            
    def all_metrics_to_df(self)->pd.DataFrame:
        """Method for compiling all tracked metrics into a single dataframe

        Returns:
            pd.DataFrame: Dataframe containing the values of all tracked 
            metrics. Refer to the Metric.values_to_df for more information
        """
        all_metric_df_lst = []
        for metric in self.metrics:
            all_metric_df_lst.append(self.metrics[metric].values_to_df())
        return pd.concat(all_metric_df_lst)
    
    def reset_orchestrator(self):
        self.metrics = {}

# Transormations
def mean_trans(input_list: List):
    return np.mean(input_list)

def sum_trans(input_list: List):
    return np.sum(input_list)

