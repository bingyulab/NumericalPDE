import numpy as np
from scipy.sparse.linalg import splu
from scipy.linalg import cholesky, solve_triangular, svd, lu_factor, lu_solve
from src.tools.sparse_matrix import SparseMatrix
import logging
from src.solver.base_solver import Solver  # Changed import


class LUdecompositionSolver(Solver):

    def solve(self, L, b):
        logging.info("LUdecompositionSolver: Solving system")
        if isinstance(L, SparseMatrix):
            # Use splu for sparse matrices
            lu = splu(L.matrix)
            return lu.solve(b)
        else:
            lu, piv = lu_factor(L)
            return lu_solve((lu, piv), b)


class CholeskySolver(Solver):

    def solve(self, L, b):
        logging.info("CholeskySolver: Solving system")
        if isinstance(L, SparseMatrix):
            L = L.to_dense()  # Ensure L is a NumPy array

        # Check for symmetry
        if not np.allclose(L, L.T, atol=1e-8):
            logging.error("CholeskySolver: Matrix L is not symmetric.")
            raise ValueError(
                "Matrix L must be symmetric for Cholesky decomposition.")

        # Check for positive definiteness by verifying all eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(L)
        min_eig = np.min(eigenvalues)
        if min_eig <= 0:
            logging.error(
                f"CholeskySolver: Matrix L is not positive definite. Minimum eigenvalue: {min_eig}"
            )
            raise np.linalg.LinAlgError(
                f"Matrix L is not positive definite. Minimum eigenvalue: {min_eig}"
            )

        try:
            L_chol = cholesky(L, lower=True)
            y = solve_triangular(L_chol, b, lower=True)
            x = solve_triangular(L_chol.T, y, lower=False)
            return x
        except np.linalg.LinAlgError as e:
            logging.error(
                f"CholeskySolver: Cholesky decomposition failed: {e}")
            raise


class LowRankApproximationSolver(Solver):

    def solve(self, L, b, rank=10):
        logging.info("LowRankApproximationSolver: Solving system")
        if isinstance(L, SparseMatrix):
            L = L.to_dense()  # Convert SparseMatrix to dense NumPy array

        U, s, Vt = svd(L)
        S = np.diag(s[:rank])
        L_approx = U[:, :rank] @ S @ Vt[:rank, :]
        return np.linalg.solve(L_approx, b)
