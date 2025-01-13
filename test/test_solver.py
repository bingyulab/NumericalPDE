import unittest
import numpy as np
from scipy.sparse import diags  # Added import
from src.solver.solver_factory import SolverFactory
from src.solver.direct_solver import DirectSolver, FastPoissonSolver
from src.solver.decomposition_solver import LUdecompositionSolver, CholeskySolver, LowRankApproximationSolver
from src.solver.multigrid_solver import MultigridSolver
from src.solver.graph_solver import GraphSolver
from src.solver.other_solver import PriorityQueueSolver
from src.problem.poisson_problem import PoissonProblem


class TestSolver(unittest.TestCase):

    def test_direct_solver(self):
        L = np.array([[3, 2], [2, 6]])
        b = np.array([2, -8])
        solver = DirectSolver()
        u = solver.solve(L, b)
        np.testing.assert_array_almost_equal(u, [2, -2], decimal=5)

    def test_fast_poisson_solver(self):
        N = 4
        L = diags([-4, 1, 1], [0, -1, 1], shape=(N, N)).toarray()
        b = np.ones(N)
        solver = FastPoissonSolver()
        u = solver.solve(L, b)
        self.assertEqual(u.shape, (N, ))

    def test_lu_decomposition_solver(self):
        N = 2  # Defined N
        L = diags([-4, 1, 1], [0, -1, 1], shape=(N, N)).toarray()
        b = np.ones(N)
        solver = LUdecompositionSolver()  # Use LUdecompositionSolver instead of FastPoissonSolver
        u = solver.solve(L, b)
        # Define the expected solution analytically or using a reliable method
        expected_u = np.linalg.solve(L, b)
        np.testing.assert_array_almost_equal(u, expected_u, decimal=5)

    def test_lu_decomposition_solver_correctness(self):
        L = np.array([[3, 2], [2, 6]])
        b = np.array([2, -8])
        solver = LUdecompositionSolver()
        u = solver.solve(L, b)
        np.testing.assert_array_almost_equal(u, [2, -2], decimal=5)

    def test_cholesky_solver(self):
        L = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        solver = CholeskySolver()
        u = solver.solve(L, b)
        np.testing.assert_array_almost_equal(u, [0.09090909, 0.63636364],
                                             decimal=5)

    def test_low_rank_approximation_solver(self):
        L = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        solver = LowRankApproximationSolver()
        u = solver.solve(L, b, rank=1)
        self.assertEqual(u.shape, (2, ))

    def test_multigrid_solver(self):
        L = diags([-4, 1, 1], [0, -1, 1], shape=(4, 4)).toarray()
        b = np.ones(4)
        solver = MultigridSolver()
        try:
            u = solver.solve(L, b)
            self.assertEqual(u.shape, (4, ))
        except np.linalg.LinAlgError as e:
            self.fail(f"MultigridSolver failed with LinAlgError: {e}")
        except ValueError as e:
            self.fail(f"MultigridSolver failed with ValueError: {e}")
        except Exception as e:
            self.fail(f"MultigridSolver failed with exception: {e}")

    def test_graph_solver(self):
        N = 4
        L = diags([-4, 1, 1], [0, -1, 1], shape=(N, N)).toarray()
        b = np.ones(N)
        solver = GraphSolver()
        u = solver.solve(L, b)
        self.assertEqual(u.shape, (N, ))

    def test_priority_queue_solver(self):
        L = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        solver = PriorityQueueSolver()
        u = solver.solve(L, b)
        self.assertEqual(u.shape, (2, ))

    def test_solver_factory(self):
        solver = SolverFactory.create_solver("direct")
        self.assertIsInstance(solver, DirectSolver)
        solver = SolverFactory.create_solver("fst")
        self.assertIsInstance(solver, FastPoissonSolver)
        solver = SolverFactory.create_solver("lu")
        self.assertIsInstance(solver, LUdecompositionSolver)
        solver = SolverFactory.create_solver("cholesky")
        self.assertIsInstance(solver, CholeskySolver)
        solver = SolverFactory.create_solver("low_rank")
        self.assertIsInstance(solver, LowRankApproximationSolver)
        solver = SolverFactory.create_solver("multigrid")
        self.assertIsInstance(solver, MultigridSolver)
        solver = SolverFactory.create_solver("graph")
        self.assertIsInstance(solver, GraphSolver)
        solver = SolverFactory.create_solver("priority_queue")
        self.assertIsInstance(solver, PriorityQueueSolver)

    def test_fast_poisson_solver_singular_matrix(self):
        problem = PoissonProblem(
            Nx=3,
            Ny=3,
            grid_type="uniform",
            solver_type="fst",
            boundary_condition_type="dirichlet",
            use_sparse=True
        )
        # Create a singular matrix (e.g., a zero matrix)
        L_singular = np.zeros((1, 1), dtype=float)  # Ensure L is float
        b = np.array([0], dtype=float)              # Ensure b is float
        problem.solver = problem.solver  # Ensure solver is set
        with self.assertRaises(Exception):
            problem.solver.solve(L_singular, b)

    
if __name__ == '__main__':
    unittest.main()
