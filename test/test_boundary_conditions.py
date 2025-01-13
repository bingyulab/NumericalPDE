import unittest
import numpy as np
from src.boundary.boundary_condition import DirichletBoundaryCondition, NeumannBoundaryCondition

class TestBoundaryConditions(unittest.TestCase):
    def test_dirichlet_boundary_condition(self):
        f = np.ones((5, 5))
        bc = DirichletBoundaryCondition(boundary_value=0)
        f_bc = bc.apply(f)
        self.assertTrue(np.all(f_bc[0, :] == 0))
        self.assertTrue(np.all(f_bc[-1, :] == 0))
        self.assertTrue(np.all(f_bc[:, 0] == 0))
        self.assertTrue(np.all(f_bc[:, -1] == 0))

    def test_neumann_boundary_condition(self):
        f = np.ones((5, 5))
        bc = NeumannBoundaryCondition(boundary_value=0)
        f_bc = bc.apply(f)
        self.assertTrue(np.all(f_bc[0, :] == f_bc[1, :]))
        self.assertTrue(np.all(f_bc[-1, :] == f_bc[-2, :]))
        self.assertTrue(np.all(f_bc[:, 0] == f_bc[:, 1]))
        self.assertTrue(np.all(f_bc[:, -1] == f_bc[:, -2]))

    def test_dirichlet_boundary_condition_no_change_interior(self):
        f = np.random.rand(5, 5)
        bc = DirichletBoundaryCondition(boundary_value=0)
        f_bc = bc.apply(f)
        # Check that interior points remain unchanged
        self.assertTrue(np.array_equal(f_bc[1:-1, 1:-1], f[1:-1, 1:-1]))

    def test_neumann_boundary_condition_no_change_interior(self):
        f = np.random.rand(5, 5)
        bc = NeumannBoundaryCondition(boundary_value=0)
        f_bc = bc.apply(f)
        # Check that interior points remain unchanged
        self.assertTrue(np.array_equal(f_bc[1:-1, 1:-1], f[1:-1, 1:-1]))

if __name__ == '__main__':
    unittest.main()
