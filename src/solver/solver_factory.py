from src.tools.singleton import SingletonMeta
from src.solver.direct_solver import DirectSolver, FastPoissonSolver 
from src.solver.decomposition_solver import LUdecompositionSolver, CholeskySolver, LowRankApproximationSolver
from src.solver.iterative_solver import ConjugateGradientSolver, JacobiSolver, GaussSeidelSolver, SORSolver, CppJacobiSolver
from src.solver.graph_solver import GraphSolver
from src.solver.other_solver import PriorityQueueSolver


class SolverFactory(metaclass=SingletonMeta):

    @staticmethod
    def create_solver(solver_type,
                      tol=1e-8,
                      max_iter=2000,
                      **kwargs):  
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
            return JacobiSolver(tol=tol, max_iter=max_iter, **kwargs)
        elif solver_type == "conjugate_gradient":
            return ConjugateGradientSolver(tol=tol, max_iter=max_iter, **kwargs)
        elif solver_type == "graph":
            return GraphSolver()
        elif solver_type == "priority_queue":
            return PriorityQueueSolver(tol=tol, max_iter=max_iter, **kwargs)
        elif solver_type == "gauss_seidel":
            return GaussSeidelSolver(tol=tol, max_iter=max_iter, **kwargs)
        elif solver_type == "sor":
            omega = kwargs.get('omega', 1.5)
            return SORSolver(omega=omega, tol=tol, max_iter=max_iter, **kwargs)
        elif solver_type == "jacobi_cpp":
            return CppJacobiSolver(**kwargs)  # Existing C++ solver
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    
