from src.solver.base_solver import Solver  # Changed import
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import isspmatrix
from src.tools.singleton import SingletonMeta
from src.tools.sparse_matrix import SparseMatrix
import logging


class IterativeSolver(Solver, ABC, metaclass=SingletonMeta):

    def __init__(self, tol=1e-8, max_iter=1000, preconditioner=None):
        self.tol = tol
        self.max_iter = max_iter
        self.preconditioner = preconditioner

    @abstractmethod
    def solve(self, L, b):
        pass


class JacobiSolver(IterativeSolver):

    def __init__(self, tol=1e-8, max_iter=1000, preconditioner=None):
        super().__init__(tol, max_iter, preconditioner)

    def solve(self, L, b):
        if self.preconditioner:
            b = self.preconditioner.apply(L, b)
        x = np.zeros_like(b)
        if isspmatrix(L.matrix if isinstance(L, SparseMatrix) else L):
            L_matrix = L.matrix.toarray() if isinstance(L, SparseMatrix) else L
        else:
            L_matrix = L
        if isinstance(L, SparseMatrix):
            D = L.diagonal(
            )  # Use method from SparseMatrix to get the diagonal
            L_dense = L.to_dense()
            R = L_dense - np.diagflat(D)  # Convert to dense before subtraction
        else:
            D = np.diag(L_matrix)
            R = L_matrix - np.diagflat(D)
        for _ in range(self.max_iter):
            # Modify dot product to handle SparseMatrix
            if isinstance(L, SparseMatrix):
                dot_R_x = L.dot(x) - D * x  # Equivalent to np.dot(R, x)
            else:
                dot_R_x = np.dot(R, x)
            x_new = (b - dot_R_x) / D
            if np.linalg.norm(x_new - x, ord=np.inf) < self.tol:
                break
            x = x_new
        return x


class ConjugateGradientSolver(IterativeSolver):

    def __init__(self, tol=1e-8, max_iter=1000, preconditioner=None):
        super().__init__(tol, max_iter, preconditioner)

    def solve(self, L, b):
        if self.preconditioner:
            b = self.preconditioner.apply(L, b)
        x = np.zeros_like(b, dtype=float)  # Initialize x as float
        if isspmatrix(L.matrix if isinstance(L, SparseMatrix) else L):
            L_matrix = L.matrix.toarray() if isinstance(L, SparseMatrix) else L
        else:
            L_matrix = L

        # Modify residual computation to handle SparseMatrix
        if isinstance(L, SparseMatrix):
            r = b - L.dot(x)
        else:
            r = b - np.dot(L, x)

        p = r
        rsold = np.dot(r.T, r)
        iteration = 0  # Initialize iteration counter
        for _ in range(self.max_iter):
            # Modify dot product to handle SparseMatrix
            if isinstance(L, SparseMatrix):
                Ap = L.dot(p)
            else:
                Ap = np.dot(L, p)
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            iteration += 1  # Increment iteration counter
            logging.debug(
                f"CG Iteration {iteration}: Residual = {np.sqrt(rsnew)}")
            if np.sqrt(rsnew) < self.tol:
                logging.info(
                    f"ConjugateGradientSolver converged after {iteration} iterations."
                )
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        else:
            logging.warning(
                f"ConjugateGradientSolver did not converge within max_iter={self.max_iter}."
            )
            raise Exception(
                f"ConjugateGradientSolver did not converge within max_iter={self.max_iter}."
            )
        logging.debug(
            f"ConjugateGradientSolver completed {iteration} iterations.")
        return x


class GaussSeidelSolver(IterativeSolver):
    def solve(self, L, b):
        logging.info("GaussSeidelSolver: Solving system")
        # Convert matrix to dense if sparse
        if isinstance(L, SparseMatrix):
            L_matrix = L.to_dense()
        else:
            L_matrix = L

        n = len(b)
        x = np.zeros_like(b, dtype=float)
        D = np.diag(L_matrix)
        # Off-diagonal part
        M = L_matrix - np.diagflat(D)
        
        residuals = []
        for _ in range(self.max_iter):
            for i in range(n):
                sigma = 0
                for j in range(n):
                    if j != i:
                        sigma += L_matrix[i, j] * x[j]
                x[i] = (b[i] - sigma) / L_matrix[i, i]
            # Save residual for plotting
            r = np.linalg.norm(b - L_matrix.dot(x), ord=2)
            residuals.append(r)
            if r < self.tol:
                break
        return x

class SORSolver(IterativeSolver):
    def __init__(self, omega=1.5, tol=1e-8, max_iter=1000, preconditioner=None):
        super().__init__(tol, max_iter, preconditioner)
        self.omega = omega

    def solve(self, L, b):
        logging.info("SORSolver: Solving system with ω=%f" % self.omega)
        if isinstance(L, SparseMatrix):
            L_matrix = L.to_dense()
        else:
            L_matrix = L

        n = len(b)
        x = np.zeros_like(b, dtype=float)
        # Diagonal
        D = np.diag(L_matrix)

        residuals = []
        for _ in range(self.max_iter):
            for i in range(n):
                sigma = 0
                for j in range(n):
                    if j != i:
                        sigma += L_matrix[i, j] * x[j]
                x[i] += self.omega * (b[i] - sigma - L_matrix[i,i]*x[i]) / L_matrix[i,i]
            # Track residual
            r = np.linalg.norm(b - L_matrix.dot(x), ord=2)
            residuals.append(r)
            if r < self.tol:
                break
        return x
