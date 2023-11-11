import pathlib
import os
import time
import json
import pickle

HERE = pathlib.Path(__file__).parent
TEST_TMP_LOC = os.path.join(HERE, "test_data")

sc_config = os.path.join(TEST_TMP_LOC, "sc_config_1.json")

with open(sc_config, "w") as f:
    config = {"state_space": "hello world"}
    json.dump(config, f)

while not os.path.isfile(sc_config):
    time.sleep(1)
    
sc_pickle_config = os.path.join(TEST_TMP_LOC, "sc_config_1.pkl")

with open(sc_pickle_config, "wb") as f:
    config = {"state_space": "hello world"}
    pickle.dump(config, f)

while not os.path.isfile(sc_pickle_config):
    time.sleep(1)
    

