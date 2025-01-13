import unittest
import numpy as np
from src.tools.sparse_matrix import SparseMatrix

class TestSparseMatrix(unittest.TestCase):
    def test_from_dense(self):
        dense = np.array([[1, 0], [0, 1]])
        sparse = SparseMatrix.from_dense(dense)
        np.testing.assert_array_equal(sparse.to_dense(), dense)

    def test_dot(self):
        dense = np.array([[1, 0], [0, 1]])
        sparse = SparseMatrix.from_dense(dense)
        vector = np.array([1, 2])
        result = sparse.dot(vector)
        np.testing.assert_array_equal(result, vector)

if __name__ == '__main__':
    unittest.main()
