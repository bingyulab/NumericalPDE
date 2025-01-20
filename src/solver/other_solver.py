import numpy as np
from scipy.sparse.linalg import spsolve
from src.tools.sparse_matrix import SparseMatrix
from queue import PriorityQueue
import logging
from src.solver.base_solver import Solver  


class DirectSolver(Solver):

    def solve(self, L, b):
        logging.info("DirectSolver: Solving system")
        if isinstance(L, SparseMatrix):
            return spsolve(L.matrix, b)
        return np.linalg.solve(L, b)


class PriorityQueueSolver(Solver):

    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter        
        self.iterations = 0

    def solve(self, L, b):
        logging.info("PriorityQueueSolver: Solving system")
        N = L.shape[0]
        x = np.zeros(N)
        residuals = PriorityQueue()
        for i in range(N):
            residuals.put((np.abs(b[i]), i))

        for _ in range(self.max_iter):
            if residuals.empty():
                break
            _, i = residuals.get()
            # Handle i = 0 by assigning sum1 = 0 and convert sums to scalars
            sum1 = np.dot(L[i, :i].toarray(), x[:i]).item() if isinstance(
                L, SparseMatrix) and i > 0 else (
                    np.dot(L[i, :i], x[:i]) if i > 0 else 0)
            sum2 = np.dot(L[i,
                            i + 1:].toarray(), x[i + 1:]).item() if isinstance(
                                L, SparseMatrix) else np.dot(
                                    L[i, i + 1:], x[i + 1:])
            # Check if L[i, i] is zero to prevent division by zero
            if L[i, i] == 0:
                logging.error(
                    f"PriorityQueueSolver: Zero diagonal element at index {i}")
                raise ZeroDivisionError(
                    f"Matrix L has zero diagonal at index {i}")
            x_new = (b[i] - sum1 - sum2) / L[i, i]
            if np.abs(x_new - x[i]) > self.tol:
                residuals.put((np.abs(x_new - x[i]), i))                
                self.iterations += 1
            x[i] = x_new
        return x
