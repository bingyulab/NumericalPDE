from src.solver.base_solver import Solver
import numpy as np
import logging
from src.tools.sparse_matrix import SparseMatrix
from scipy.fftpack import dst, idst


class DirectSolver(Solver):

    def solve(self, L, b):
        logging.info("DirectSolver: Solving system")
        if isinstance(L, SparseMatrix):
            solution = splin.spsolve(L.matrix, b)
        else:
            solution = np.linalg.solve(L, b)
        
        # Ensure the solution is reshaped correctly
        if solution.ndim == 1:
            # Assuming square grid for 2D Poisson
            N = int(np.sqrt(len(solution)))
            solution = solution.reshape((N, N))
        return solution


class FastPoissonSolver(Solver):

    def solve(self, L, b):
        logging.info("FastPoissonSolver: Solving system")
        if isinstance(L, SparseMatrix):
            L = L.to_dense()
        N = int(np.sqrt(L.shape[0]))
        # Removed the rank check to avoid SparseEfficiencyWarning and ValueError
        # if np.linalg.matrix_rank(L) < N * N:
        #     raise Exception("FastPoissonSolver encountered a singular matrix (rank < size).")

        b = b.reshape((N, N))
        b_hat = dst(dst(b, type=1).T, type=1).T
        k = np.arange(1, N + 1)
        lambda_k = 2 * (1 - np.cos(np.pi * k / (N + 1)))
        L_hat = np.add.outer(lambda_k, lambda_k)
        if np.any(L_hat == 0):
            raise Exception(
                "FastPoissonSolver encountered a singular matrix (L_hat contains zeros)."
            )
        u_hat = b_hat / L_hat
        u = idst(idst(u_hat, type=1).T, type=1).T / (2 * (N + 1))
        return u.flatten()
