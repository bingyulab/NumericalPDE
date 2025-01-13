import unittest
import numpy as np
from src.grid.uniform_grid import UniformGrid


class TestGrid(unittest.TestCase):

    def test_uniform_grid(self):
        grid = UniformGrid(5, 5)
        X, Y = grid.generate_grid()
        self.assertEqual(X.shape, (5, 5))
        self.assertEqual(Y.shape, (5, 5))
        np.testing.assert_array_almost_equal(X[0, :], np.linspace(0, 1, 5))
        np.testing.assert_array_almost_equal(Y[:, 0], np.linspace(0, 1, 5))

    def test_uniform_grid_spacing(self):
        N = 5
        grid = UniformGrid(N, N)
        X, Y = grid.generate_grid()
        expected_spacing = 1 / (N - 1)
        # Check spacing between adjacent points
        self.assertTrue(np.allclose(np.diff(X, axis=1), expected_spacing))
        self.assertTrue(np.allclose(np.diff(Y, axis=0), expected_spacing))

    def test_uniform_grid_bounds(self):
        N = 5
        grid = UniformGrid(N, N)
        X, Y = grid.generate_grid()
        self.assertTrue(np.allclose(X, np.linspace(0, 1, N)[None, :]))
        self.assertTrue(np.allclose(Y, np.linspace(0, 1, N)[:, None]))


if __name__ == '__main__':
    unittest.main()
