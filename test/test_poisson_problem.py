import unittest
from src.problem.poisson_problem import PoissonProblem


class TestPoissonProblem(unittest.TestCase):

    def test_solve(self):
        problem = PoissonProblem(10,
                                 10,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="dirichlet",
                                 use_sparse=False)  # Ensure use_sparse is False for DirectSolver
        u, X, Y = problem.solve()
        self.assertEqual(u.shape, (8, 8))

    def test_solve_parallel(self):
        problem = PoissonProblem(10,
                                 10,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="dirichlet")
        u, X, Y = problem.solve_parallel()
        self.assertEqual(u.shape, (8, 8))

    def test_solve_multithreaded(self):
        problem = PoissonProblem(10,
                                 10,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="dirichlet")
        u, X, Y = problem.solve_multithreaded()
        self.assertEqual(u.shape, (8, 8))

    def test_poisson_problem_dirichlet(self):
        N = 10
        problem = PoissonProblem(Nx=N,
                                 Ny=N,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="dirichlet",
                                 use_sparse=False)  # Ensure use_sparse is False for DirectSolver
        u, X, Y = problem.solve()
        self.assertEqual(u.shape, (N - 2, N - 2))
        # Additional assertions can verify boundary conditions if accessible

    def test_poisson_problem_neumann(self):
        N = 10
        problem = PoissonProblem(Nx=N,
                                 Ny=N,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="neumann",
                                 use_sparse=False)  # Ensure use_sparse is False for DirectSolver
        u, X, Y = problem.solve()
        self.assertEqual(u.shape, (N - 2, N - 2))
        # Additional assertions can verify boundary conditions if accessible

    def test_poisson_problem_non_uniform_grid(self):
        N = 10
        problem = PoissonProblem(
            Nx=N,
            Ny=N,
            grid_type="non_uniform",  # Now supported
            solver_type="direct",
            boundary_condition_type="dirichlet",
            use_sparse=False)  # Ensure use_sparse is False for DirectSolver
        u, X, Y = problem.solve()
        self.assertEqual(u.shape, (N - 2, N - 2))
        # Additional assertions can verify boundary conditions if accessible


if __name__ == '__main__':
    unittest.main()
