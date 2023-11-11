# import unittest
# import os
# import json

# from pymlrf.Dataset import Dataset
# from pymlrf.FileSystem import FileHandler


# from ..config import TEST_TMP_LOC

# f_config = os.path.join(TEST_TMP_LOC, "f_config.json")
# with open(f_config, "w") as f:
#     config = {"features": None}
#     json.dump(config, f)

# config_fh = FileHandler(path=f_config)

    
# class DatasetTest(unittest.TestCase):

#     def test_from_file(self):
#         df = Dataset(loc=TEST_TMP_LOC, config_fh=config_fh)
#         df.read()
#         self.assertTrue(df.features is None)
        