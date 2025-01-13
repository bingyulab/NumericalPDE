from src.tools.singleton import SingletonMeta
from src.tools.preconditioner import JacobiPreconditioner, IncompleteCholeskyPreconditioner
from src.solver.direct_solver import DirectSolver, FastPoissonSolver 
from src.solver.decomposition_solver import LUdecompositionSolver, CholeskySolver, LowRankApproximationSolver
from src.solver.iterative_solver import ConjugateGradientSolver, JacobiSolver, GaussSeidelSolver, SORSolver
from src.solver.multigrid_solver import MultigridSolver
from src.solver.graph_solver import GraphSolver
from src.solver.other_solver import PriorityQueueSolver
from src.solver.pinn_solver import SolverPINN  # Import the PINN Solver


class SolverFactory(metaclass=SingletonMeta):

    @staticmethod
    def create_solver(solver_type,
                      preconditioner_type=None,
                      tol=1e-8,
                      max_iter=1000,
                      **kwargs):  # Added kwargs to accept additional arguments
        preconditioner = None
        if preconditioner_type == "jacobi":
            preconditioner = JacobiPreconditioner()
        elif preconditioner_type == "incomplete_cholesky":
            preconditioner = IncompleteCholeskyPreconditioner()

        if solver_type == "direct":
            return DirectSolver()
        elif solver_type == "fst":
            return FastPoissonSolver()
        elif solver_type == "lu":
            return LUdecompositionSolver()
        elif solver_type == "cholesky":
            return CholeskySolver()
        elif solver_type == "low_rank":
            return LowRankApproximationSolver()
        elif solver_type == "jacobi":
            return JacobiSolver(tol=tol,
                                max_iter=max_iter,
                                preconditioner=preconditioner)
        elif solver_type == "conjugate_gradient":
            return ConjugateGradientSolver(tol=tol,
                                           max_iter=max_iter,
                                           preconditioner=preconditioner)
        elif solver_type == "multigrid":
            return MultigridSolver(tol=tol, max_iter=max_iter)
        elif solver_type == "graph":
            return GraphSolver()
        elif solver_type == "priority_queue":
            return PriorityQueueSolver(tol=tol, max_iter=max_iter)
        elif solver_type == "gauss_seidel":
            return GaussSeidelSolver(tol=tol, max_iter=max_iter, preconditioner=preconditioner)
        elif solver_type == "sor":
            omega = kwargs.get('omega', 1.5)
            return SORSolver(omega=omega, tol=tol, max_iter=max_iter, preconditioner=preconditioner)
        elif solver_type == "pinn":
            return SolverPINN(**kwargs)  # Pass additional keyword arguments to SolverPINN
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @staticmethod
    def load_config(config_path="config.yaml"):
        import yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        solver_config = config.get('solver', {})
        return solver_config
