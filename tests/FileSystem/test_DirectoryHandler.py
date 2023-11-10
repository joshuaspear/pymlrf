import unittest
import os

from pymlrf.FileSystem.DirectoryHandler import DirectoryHandler

from ..config import TEST_TMP_LOC

class DirectoryHandlerTest(unittest.TestCase):
        
    def test_create(self):
        test_loc = os.path.join(TEST_TMP_LOC, "dh_test_loc1")
        self.dh = DirectoryHandler(loc=test_loc)
        self.dh.create()
        assert os.path.isdir(test_loc)
    
    def test_clear(self):
        test_loc = os.path.join(TEST_TMP_LOC, "dh_test_loc2")
        os.mkdir(test_loc)
        test_file = os.path.join(test_loc, "test_file.txt")
        self.dh = DirectoryHandler(loc=test_loc)
        with open(test_file, "w") as f:
            f.write("TEST LINE")
    
    def test_delete(self):
        test_loc = os.path.join(TEST_TMP_LOC, "dh_test_loc3")
        os.mkdir(test_loc)
        self.dh = DirectoryHandler(loc=test_loc)
        self.dh.delete()
        assert not os.path.isdir(test_loc)
    

