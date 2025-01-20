import unittest
import numpy as np
from src.solver.iterative_solver import JacobiSolver, ConjugateGradientSolver
from src.tools.preconditioner import JacobiPreconditioner, IncompleteCholeskyPreconditioner


class TestIterativeSolver(unittest.TestCase):

    def test_jacobi_solver_singleton(self):
        solver1 = JacobiSolver()
        solver2 = JacobiSolver()
        self.assertIsInstance(solver1, JacobiSolver)
        self.assertIsInstance(solver2, JacobiSolver)
        self.assertIsNot(solver1, solver2, "JacobiSolver instances should be distinct")

    def test_conjugate_gradient_solver_singleton(self):
        solver1 = ConjugateGradientSolver()
        solver2 = ConjugateGradientSolver()
        self.assertIsInstance(solver1, ConjugateGradientSolver)
        self.assertIsInstance(solver2, ConjugateGradientSolver)
        self.assertIsNot(solver1, solver2, "ConjugateGradientSolver instances should be distinct")

    def test_jacobi_solver(self):
        # Create a simple linear system
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        solver = JacobiSolver(tol=1e-10, max_iter=100)
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations > 0, "JacobiSolver should perform at least one iteration")

    def test_conjugate_gradient_solver(self):
        # Create a symmetric positive-definite linear system
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        solver = ConjugateGradientSolver(tol=1e-10, max_iter=100)
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations > 0, "ConjugateGradientSolver should perform at least one iteration")

    def test_jacobi_solver_with_preconditioner(self):
        # Create a simple linear system
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        preconditioner = JacobiPreconditioner(A)
        solver = JacobiSolver(tol=1e-10, max_iter=100)
        # Modify solver to accept preconditioner if applicable
        # Assuming JacobiSolver can accept a preconditioner
        # This depends on the actual implementation
        solver.preconditioner = preconditioner
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations > 0, "JacobiSolver with preconditioner should perform at least one iteration")

    def test_conjugate_gradient_solver_with_preconditioner(self):
        # Create a symmetric positive-definite linear system
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        preconditioner = IncompleteCholeskyPreconditioner(A)
        solver = ConjugateGradientSolver(tol=1e-10, max_iter=100)
        # Modify solver to accept preconditioner if applicable
        # Assuming ConjugateGradientSolver can accept a preconditioner
        solver.preconditioner = preconditioner
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations > 0, "ConjugateGradientSolver with preconditioner should perform at least one iteration")

    def test_jacobi_solver_convergence(self):
        # Create a diagonally dominant matrix to ensure convergence
        A = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]])
        b = np.array([12, 12, 12])
        solver = JacobiSolver(tol=1e-10, max_iter=100)
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations <= solver.max_iter, "JacobiSolver should converge within the maximum iterations")

    def test_conjugate_gradient_solver_convergence(self):
        # Create a symmetric positive-definite matrix
        A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        b = np.array([1, 2, 3])
        solver = ConjugateGradientSolver(tol=1e-10, max_iter=100)
        solution = solver.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected, decimal=6)
        self.assertTrue(solver.iterations <= solver.max_iter, "ConjugateGradientSolver should converge within the maximum iterations")

    def test_jacobi_solver_max_iterations(self):
        # Create a system that may not converge within a limited number of iterations
        A = np.array([[1, 2], [3, 4]])
        b = np.array([5, 11])
        solver = JacobiSolver(tol=1e-12, max_iter=5)  # Set low max_iter
        solution = solver.solve(A, b)
        # Since the system may not converge, just check that it does not exceed max_iter
        self.assertLessEqual(solver.iterations, solver.max_iter, "JacobiSolver should not exceed maximum iterations")

    def test_conjugate_gradient_solver_max_iterations(self):
        # Create a system that may not converge within a limited number of iterations
        A = np.array([[1, 0], [0, 1]])
        b = np.array([1, 1])
        solver = ConjugateGradientSolver(tol=1e-12, max_iter=1)  # Set low max_iter
        solution = solver.solve(A, b)
        # Even though the system is simple, check that iterations do not exceed max_iter
        self.assertLessEqual(solver.iterations, solver.max_iter, "ConjugateGradientSolver should not exceed maximum iterations")


if __name__ == '__main__':
    unittest.main()
