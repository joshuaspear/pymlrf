import unittest
import os

from pymlrf.FileSystem import DirectoryHandler, FileHandler

from ..config import TEST_TMP_LOC

class FileHandlerTest(unittest.TestCase):
                
    def test_delete(self):
        test_file = os.path.join(TEST_TMP_LOC, "test_file.txt")
        fh = FileHandler(test_file)
        with open(test_file, "w") as f:
            f.write("TEST LINE")
        fh.delete()
        assert not os.path.isfile(test_file)
    

