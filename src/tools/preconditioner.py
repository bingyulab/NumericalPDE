import numpy as np
from scipy.sparse import isspmatrix

class Preconditioner:
    def apply(self, L, b):
        raise NotImplementedError("Preconditioner must implement the apply method")

class JacobiPreconditioner(Preconditioner):
    def apply(self, L, b):
        if isspmatrix(L):
            L = L.tocsr()
            diag_L = L.diagonal()
        elif isinstance(L, np.ndarray):
            diag_L = np.diag(L)
        else:
            diag_L = L.diagonal()
        D_inv = np.diag(1 / diag_L)
        return D_inv @ b

class IncompleteCholeskyPreconditioner(Preconditioner):
    def apply(self, L, b):
        if isspmatrix(L):
            L = L.tocsr()
            L = L.toarray()  # Convert to dense if necessary
        L = np.tril(L)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
