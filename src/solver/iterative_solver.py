from src.solver.base_solver import Solver  # Changed import
from abc import ABC, abstractmethod
import numpy as np
import logging
import scipy.sparse as sp
from cpp.build import poisson_solvers

class IterativeSolver(Solver, ABC):

    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = float(tol)  # Ensure tol is a float
        self.max_iter = max_iter
        self.iterations = 0  # Initialize iterations

    @abstractmethod
    def solve(self, L, b):
        pass


class JacobiSolver(IterativeSolver):

    def solve(self, L, b):
        logging.info("JacobiSolver: Solving system")
        if sp.issparse(L):
            L = L.toarray()
        logging.debug(f"L: {L.shape}, {sp.issparse(L)}")
        
        D = np.diag(L)
        R = L - np.diagflat(D)
        x = np.zeros_like(b, dtype=float)

        for i in range(self.max_iter):
            x_new = (b - np.dot(R, x)) / D
            if np.linalg.norm(x_new - x, ord=np.inf) < self.tol:
                self.iterations = i + 1
                return x_new
            x = x_new
        else:
            self.iterations = self.max_iter

        return x


class ConjugateGradientSolver(IterativeSolver):

    def solve(self, L, b):
        if sp.issparse(L):
            L = L.toarray()

        x = np.zeros_like(b, dtype=float)
        r = b - L.dot(x)
        p = r.copy()
        rsold = np.dot(r, r)

        for i in range(self.max_iter):
            Ap = L.dot(p)
            alpha = rsold / np.dot(p, Ap)  # Now both are scalars
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < self.tol:
                self.iterations = i + 1
                return x
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        else:
            self.iterations = self.max_iter
        return x


class GaussSeidelSolver(IterativeSolver):

    def solve(self, L, b):
        logging.info("GaussSeidelSolver: Solving system")
        if sp.issparse(L):
            L = L.toarray()

        n = len(b)
        x = np.zeros_like(b, dtype=float)
        D = np.diag(L)
        M = L - np.diagflat(D)

        for j in range(self.max_iter):
            x_new = x.copy()
            for i in range(n):
                sigma = np.dot(M[i, :], x_new)
                x_new[i] = (b[i] - sigma) / D[i]
            if np.linalg.norm(x_new - x, ord=np.inf) < self.tol:
                self.iterations = j + 1
                return x_new
            x = x_new
        else:
            self.iterations = self.max_iter
        return x


class SORSolver(IterativeSolver):
    def __init__(self, omega=1.5, tol=1e-8, max_iter=1000):
        super().__init__(tol, max_iter)
        self.omega = omega

    def solve(self, L, b):
        logging.info("SORSolver: Solving system")
        if sp.issparse(L):
            L = L.toarray()

        n = len(b)
        x = np.zeros_like(b, dtype=float)
        D = np.diag(L)
        M = L - np.diagflat(D)

        for j in range(self.max_iter):
            x_new = x.copy()
            for i in range(n):
                sigma = np.dot(M[i, :], x_new)
                x_new[i] = (1 - self.omega) * x[i] + (self.omega / D[i]) * (b[i] - sigma)
            if np.linalg.norm(x_new - x, ord=np.inf) < self.tol:
                self.iterations = j + 1
                return x_new
            x = x_new

        else:
            self.iterations = self.max_iter
        return x


class CppJacobiSolver(Solver):

    def solve(self, L, b):
        logging.info("CppJacobiSolver: Solving system")
        if sp.issparse(L):
            L = L.toarray()
        
        logging.info(f"L: {L.shape}")
        # Call the C++ solver and unpack solution and iterations
        solution, iterations = poisson_solvers.JacobiSolver().solve(L, b)
        # Convert the solution to a NumPy array
        self.iterations = iterations  # Store the number of iterations
        return np.array(solution)
