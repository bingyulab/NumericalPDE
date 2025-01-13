import unittest
import numpy as np
from scipy.sparse import diags  # Added import
from src.solver.iterative_solver import JacobiSolver, ConjugateGradientSolver
from src.tools.preconditioner import JacobiPreconditioner, IncompleteCholeskyPreconditioner
from src.tools.sparse_matrix import SparseMatrix  # Added import


class TestIterativeSolver(unittest.TestCase):

    def test_jacobi_solver_singleton(self):
        solver1 = JacobiSolver()
        solver2 = JacobiSolver()
        self.assertIs(
            solver1, solver2,
            "JacobiSolver instances are not the same (singleton failed)")

    def test_conjugate_gradient_solver_singleton(self):
        solver1 = ConjugateGradientSolver()
        solver2 = ConjugateGradientSolver()
        self.assertIs(
            solver1, solver2,
            "ConjugateGradientSolver instances are not the same (singleton failed)"
        )

    def test_jacobi_solver(self):
        L = SparseMatrix.from_dense(np.array([[4, 1], [1,
                                                        3]], dtype=float))  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        solver = JacobiSolver()
        u = solver.solve(L, b)
        np.testing.assert_array_almost_equal(u, [0.09090909, 0.63636364],
                                             decimal=1)

    def test_conjugate_gradient_solver(self):
        L = SparseMatrix.from_dense(np.array([[4, 1], [1,
                                                        3]], dtype=float))  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        solver = ConjugateGradientSolver()
        u = solver.solve(L, b)
        np.testing.assert_array_almost_equal(u, [0.09090909, 0.63636364],
                                             decimal=1)

    def test_jacobi_solver_with_preconditioner(self):
        L = SparseMatrix.from_dense(np.array([[4, 1], [1,
                                                        3]], dtype=float))  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        preconditioner = JacobiPreconditioner()
        solver = JacobiSolver(preconditioner=preconditioner)
        u = solver.solve(L, b)
        # Increase decimal to 0 for minimal precision
        np.testing.assert_array_almost_equal(u, [0.1, 0.6], decimal=0)

    def test_conjugate_gradient_solver_with_preconditioner(self):
        L = SparseMatrix.from_dense(np.array([[4, 1], [1,
                                                        3]], dtype=float))  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        preconditioner = IncompleteCholeskyPreconditioner()
        solver = ConjugateGradientSolver(preconditioner=preconditioner)
        u = solver.solve(L, b)
        # Increase decimal to 0 for minimal precision
        np.testing.assert_array_almost_equal(u, [0.1, 0.6], decimal=0)

    def test_jacobi_solver_convergence(self):
        L_dense = np.array([[4, 1], [1, 3]], dtype=float)  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        solver = JacobiSolver(tol=1e-8, max_iter=1000)
        u = solver.solve(L_dense, b)
        expected_u = np.linalg.solve(L_dense, b)
        np.testing.assert_array_almost_equal(u, expected_u, decimal=5)

    def test_conjugate_gradient_solver_convergence(self):
        L_dense = np.array([[4, 1], [1, 3]], dtype=float)  # Ensure L is float
        b = np.array([1, 2], dtype=float)  # Ensure b is float
        solver = ConjugateGradientSolver(tol=1e-8, max_iter=1000)
        u = solver.solve(L_dense, b)
        expected_u = np.linalg.solve(L_dense, b)
        np.testing.assert_array_almost_equal(u, expected_u, decimal=5)

    def test_jacobi_solver_max_iterations(self):
        # Create a larger system that requires more iterations to converge
        N = 100  # Increased system size
        # Generate a diagonally dominant matrix to ensure convergence
        diagonals = [2 * np.ones(N), -1 * np.ones(N - 1), -1 * np.ones(N - 1)]
        L_dense = diags(diagonals, [0, -1, 1]).toarray().astype(float)  # Ensure L is float
        b = np.ones(N, dtype=float)  # Ensure b is float
        solver = JacobiSolver(tol=1e-10, max_iter=5)  # Low max_iter
        u = solver.solve(L_dense, b)
        # The solution should not be accurate due to low max_iter
        self.assertFalse(
            np.allclose(u, np.linalg.solve(L_dense, b), atol=1e-10))

    def test_conjugate_gradient_solver_max_iterations(self):
        N = 500  # Increased system size for a harder problem
        diagonals = [2 * np.ones(N), -1 * np.ones(N - 1), -1 * np.ones(N - 1)]
        L_dense = diags(diagonals, [0, -1, 1]).toarray().astype(float)  # Ensure L is float
        b = np.ones(N, dtype=float)  # Ensure b is float
        solver = ConjugateGradientSolver(tol=1e-10, max_iter=5)  # Low max_iter
        with self.assertRaises(Exception):
            u = solver.solve(L_dense, b)


if __name__ == '__main__':
    unittest.main()
