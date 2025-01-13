import unittest
import numpy as np
from src.tools.preconditioner import JacobiPreconditioner, IncompleteCholeskyPreconditioner

class TestPreconditioner(unittest.TestCase):
    def test_jacobi_preconditioner(self):
        L = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        preconditioner = JacobiPreconditioner()
        b_precond = preconditioner.apply(L, b)
        np.testing.assert_array_almost_equal(b_precond, [0.25, 0.66666667], decimal=0)

    def test_incomplete_cholesky_preconditioner(self):
        L = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        preconditioner = IncompleteCholeskyPreconditioner()
        b_precond = preconditioner.apply(L, b)
        np.testing.assert_array_almost_equal(b_precond, [0.25, 0.58333333], decimal=0)

if __name__ == '__main__':
    unittest.main()
