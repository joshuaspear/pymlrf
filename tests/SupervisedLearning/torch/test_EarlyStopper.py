from dataclasses import dataclass
import unittest
from pymlrf.SupervisedLearning.torch.EarlyStopper import (
    PercEpsImprove, PassThruStoppingCriteria, EarlyStopper, 
    EarlyStoppingException
)
    

#from ..config import sc_config, TEST_TMP_LOC

test_values = [0.9,0.7,0.1,0.8]
test_epochs = [1,2,3,4]


class PassThruStoppingCriteriaTest(unittest.TestCase):
    
    def test_call(self):
        sc = PassThruStoppingCriteria()
        for value, epoch in zip(test_values,test_epochs):
            sc(value=value,epoch=epoch)
        assert sc.optimal_epoch == 4

eps_1_test_case = {
    "eps":0.001,
    # Epoch 1:
        # Optimal = 1 by default
    # Epoch 2:
        # (0.9-0.7)/0.9 = 0.2
        # 0.2 > 0.001
        # Optimal = 2
    # Epoch 3:
        # (0.7-0.1)/0.7 = 0.8
        # 0.8 > 0.001
        # Optimal = 3
    # Epoch 3:
        # (0.1-0.8)/0.1 = -7
        # Raise Exception
        # Optimal = 3
    "res_lkp": {
        1:1, 2:2, 3:3, 4:3
        }
}

eps_2_test_case = {
    "eps":0.5,
    # Epoch 1:
        # Optimal = 1 by default
    # Epoch 2:
        # (0.9-0.7)/0.9 = 0.2
        # Raise Exception
        # Optimal = 1
    # Epoch 3:
        # Optimal = 1
    # Epoch 3:
        # Optimal = 1
    "res_lkp": {
        1:1, 2:1, 3:1, 4:1
        }
}

eps_2 = 0.5

class PercEpsImproveTest(unittest.TestCase):

    def test_call_1(self):
        sc = PercEpsImprove(
            eps=eps_1_test_case["eps"]
        )
        for value, epoch in zip(test_values,test_epochs):
            sc(value=value,epoch=epoch)
            assert sc.optimal_epoch == eps_1_test_case["res_lkp"][epoch]

    def test_call_2(self):
        sc = PercEpsImprove(
            eps=eps_2_test_case["eps"]
        )
        for value, epoch in zip(test_values,test_epochs):
            sc(value=value,epoch=epoch)
            print(sc.optimal_epoch)
            assert sc.optimal_epoch == eps_2_test_case["res_lkp"][epoch]

class EarlyStopperPassTest(unittest.TestCase):
    
    def test_regression_stop(self):
        es = EarlyStopper(
            stopping_func=PassThruStoppingCriteria(),
            action="stop"
        )
        for value, epoch in zip(test_values,test_epochs):
            es.assess(value=value, epoch=epoch)
        
        assert es.optimal_epoch == 4
        
    def test_regression_capture(self):
        es = EarlyStopper(
            stopping_func=PassThruStoppingCriteria(),
            action="capture"
        )
        for value, epoch in zip(test_values,test_epochs):
            es.assess(value=value, epoch=epoch)
        
        assert es.optimal_epoch == 4
    
    
class EarlyStopperPercTest(unittest.TestCase):
    
    def test_regression_stop_1(self):
        es = EarlyStopper(
            stopping_func=PercEpsImprove(eps=eps_1_test_case["eps"], direction="ls"),
            action="stop"
        )
        for value, epoch in zip(test_values,test_epochs):
            try:
                es.assess(value=value, epoch=epoch)
            except EarlyStoppingException:
                pass
        assert es.optimal_epoch == eps_1_test_case["res_lkp"][epoch]
        
    def test_regression_capture_1(self):
        es = EarlyStopper(
            stopping_func=PercEpsImprove(eps=eps_1_test_case["eps"], direction="ls"),
            action="capture"
        )
        for value, epoch in zip(test_values,test_epochs):
            try:
                es.assess(value=value, epoch=epoch)
            except EarlyStoppingException as e:
                raise Exception("Throwing exception incorrectly")
        assert es.optimal_epoch == eps_1_test_case["res_lkp"][epoch]


    def test_regression_stop_2(self):
        es = EarlyStopper(
            stopping_func=PercEpsImprove(eps=eps_2_test_case["eps"], direction="ls"),
            action="stop"
        )
        for value, epoch in zip(test_values,test_epochs):
            try:
                es.assess(value=value, epoch=epoch)
            except EarlyStoppingException:
                pass
        assert es.optimal_epoch == eps_2_test_case["res_lkp"][epoch]
        
    def test_regression_capture_2(self):
        es = EarlyStopper(
            stopping_func=PercEpsImprove(eps=eps_2_test_case["eps"], direction="ls"),
            action="capture"
        )
        for value, epoch in zip(test_values,test_epochs):
            try:
                es.assess(value=value, epoch=epoch)
            except EarlyStoppingException as e:
                raise Exception("Throwing exception incorrectly")
        assert es.optimal_epoch == eps_2_test_case["res_lkp"][epoch]
    
