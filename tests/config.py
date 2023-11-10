import pathlib
import os
import time
import json

HERE = pathlib.Path(__file__).parent
TEST_TMP_LOC = os.path.join(HERE, "test_data")

sc_config = os.path.join(TEST_TMP_LOC, "sc_config_1.json")

with open(sc_config, "w") as f:
    config = {"state_space": "hello world"}
    json.dump(config, f)

while not os.path.isfile(sc_config):
    time.sleep(1)
