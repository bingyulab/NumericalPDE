import numpy as np
import logging
from scipy.linalg import lu_factor, lu_solve, cholesky, solve_triangular  
from src.solver.base_solver import Solver
import scipy.sparse as sp

class LUdecompositionSolver(Solver):
    
    def solve(self, A, b):
        logging.info("LUdecompositionSolver: Solving system")
        if sp.issparse(A):
            # Use splu for sparse matrices
            lu = sp.linalg.splu(A)
            return lu.solve(b)
        else:
            lu, piv = lu_factor(A)
            return lu_solve((lu, piv), b)


class CholeskySolver(Solver):
    
    
    def solve(self, L, b):
        logging.info("CholeskySolver: Solving system")
        if sp.issparse(L):
            L = L.toarray()  # Ensure L is a NumPy array

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
    def __init__(self, rank=10):
        self.rank = rank
        
    def solve(self, A, b):
        logging.info("LowRankApproximationSolver: Solving system")
        U, S, Vt = sp.linalg.svds(A, k=self.rank)
        # Step 1: Compute U^T b
        U_T_b = U.T @ b  # Shape: (k,)
        # Step 2: Multiply by S^-1 (element-wise)
        S_inv_U_T_b = (1.0 / S) * U_T_b  # Shape: (k,)
        # Step 3: Multiply by V to get the solution vector
        u_int = Vt.T @ S_inv_U_T_b  # Shape: (m,)
        return u_int
