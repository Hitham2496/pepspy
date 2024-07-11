import unittest
import numpy as np
from pepspy import Node

class TestNode(unittest.TestCase):

    def test_initialization(self):
        tensor = np.array([[1, 0], [0, 1]])
        # peps = PEPS([tensor])
        # self.assertEqual(len(peps.tensors), 1)
        # self.assertTrue(np.array_equal(peps.tensors[0], tensor))

if __name__ == "__main__":
    unittest.main()
