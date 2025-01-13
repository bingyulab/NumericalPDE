from abc import ABC, abstractmethod

class BoundaryCondition(ABC):
    @abstractmethod
    def apply(self, F, X=None, Y=None):
        pass

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self, g=None):
        self.g = g  # g is the boundary condition function

    def apply(self, F, X=None, Y=None):
        """Apply Dirichlet boundary conditions using the provided boundary function g"""
        f_bc = F.copy()
        if self.g is not None and X is not None and Y is not None:
            f_bc[0, :] = self.g(X[0, :], Y[0, :])    # Bottom boundary
            f_bc[-1, :] = self.g(X[-1, :], Y[-1, :])  # Top boundary
            f_bc[:, 0] = self.g(X[:, 0], Y[:, 0])    # Left boundary
            f_bc[:, -1] = self.g(X[:, -1], Y[:, -1])  # Right boundary
        else:
            f_bc[0, :] = 0  # Bottom boundary
            f_bc[-1, :] = 0  # Top boundary
            f_bc[:, 0] = 0  # Left boundary
            f_bc[:, -1] = 0  # Right boundary
        return f_bc

class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self, boundary_value=0):
        self.boundary_value = boundary_value

    def apply(self, f):
        f_bc = f.copy()
        f_bc[0, :] = f_bc[1, :] + self.boundary_value
        f_bc[-1, :] = f_bc[-2, :] + self.boundary_value
        f_bc[:, 0] = f_bc[:, 1] + self.boundary_value
        f_bc[:, -1] = f_bc[:, -2] + self.boundary_value
        return f_bc
