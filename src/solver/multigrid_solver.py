import logging
import numpy as np
from src.solver.base_solver import Solver  # Changed import to only import Solver
from src.tools.sparse_matrix import SparseMatrix  # Added import for SparseMatrix


class MultigridSolver(Solver):
    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, L, b):
        logging.info("MultigridSolver: Solving system")
        if isinstance(L, SparseMatrix):
            L = L.to_dense()

        def restrict(x):
            # Handle odd-length arrays by truncating to the minimum length
            x_even = x[::2]
            x_odd = x[1::2]
            min_len = min(len(x_even), len(x_odd))
            return (x_even[:min_len] + x_odd[:min_len]) / 2

        def prolong(e, target_size):
            # Handle odd-length target sizes by padding e as necessary
            e_fine = np.zeros(target_size)
            len_even = len(e_fine[::2])
            len_odd = len(e_fine[1::2])
            
            # Pad e with zeros if it's shorter than the required length
            e_padded_even = np.pad(e, (0, max(0, len_even - len(e))), 'constant')
            e_padded_odd = np.pad(e, (0, max(0, len_odd - len(e))), 'constant')
            
            e_fine[::2] += e_padded_even[:len_even]
            e_fine[1::2] += e_padded_odd[:len_odd]
            return e_fine

        def v_cycle(L, b, x):
            if L.shape[0] <= 2:
                return np.linalg.solve(L, b)
            from src.solver.iterative_solver import JacobiSolver  # Keep local import
            solver = JacobiSolver(tol=self.tol, max_iter=10)
            x = solver.solve(L, b)
            r = b - np.dot(L, x)
            r2 = restrict(r)
            e2 = np.zeros_like(r2)
            
            # Correct size calculation for coarser grid
            size = len(r2)
            L2 = self.restrict_matrix(L, size)
            
            e2 = v_cycle(L2, r2, e2)
            prolong_e2 = prolong(e2, len(x))
            if len(prolong_e2) != len(x):
                raise ValueError(f"Prolongated error has shape {prolong_e2.shape}, expected {x.shape}")
            x += prolong_e2
            solver = JacobiSolver(tol=self.tol, max_iter=10)
            x = solver.solve(L, b)
            return x

        x = np.zeros_like(b)
        for _ in range(self.max_iter):
            x = v_cycle(L, b, x)
            if np.linalg.norm(b - np.dot(L, x)) < self.tol:
                break
        return x

    def restrict_matrix(self, L, size):
        # Correctly restrict the matrix to the coarser grid size
        # Assuming a grid reduction by factor of 2 in each dimension for 2D problems
        return L[:size, :size]
