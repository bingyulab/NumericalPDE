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
            f_bc[0, :] = self.g(X[0, :], Y[0, :])  # Bottom boundary
            f_bc[-1, :] = self.g(X[-1, :], Y[-1, :])  # Top boundary
            f_bc[:, 0] = self.g(X[:, 0], Y[:, 0])  # Left boundary
            f_bc[:, -1] = self.g(X[:, -1], Y[:, -1])  # Right boundary
        else:
            f_bc[0, :] = 0  # Bottom boundary
            f_bc[-1, :] = 0  # Top boundary
            f_bc[:, 0] = 0  # Left boundary
            f_bc[:, -1] = 0  # Right boundary
        return f_bc


class NeumannBoundaryCondition(BoundaryCondition):

    def __init__(self, boundary_value=0, h=1.0):
        self.boundary_value = boundary_value  # Derivative value at the boundary
        self.h = h  # Grid spacing

    def apply(self, F, X=None, Y=None):
        """Apply Neumann boundary conditions using the finite difference approximation for the derivative"""
        f_bc = F.copy()
        if X is not None and Y is not None:
            # Apply Neumann boundary condition using finite difference approximation
            # Bottom boundary (y=0): forward difference approximation for u'(0)
            f_bc[0, :] = f_bc[
                1, :] + self.boundary_value * self.h  # Bottom boundary
            # Top boundary (y=N-1): backward difference approximation for u'(N-1)
            f_bc[-1, :] = f_bc[
                -2, :] - self.boundary_value * self.h  # Top boundary
            # Left boundary (x=0): forward difference approximation for u'(0)
            f_bc[:,
                 0] = f_bc[:,
                           1] + self.boundary_value * self.h  # Left boundary
            # Right boundary (x=N-1): backward difference approximation for u'(N-1)
            f_bc[:,
                 -1] = f_bc[:,
                            -2] - self.boundary_value * self.h  # Right boundary
        else:
            # Homogeneous Neumann boundary conditions (derivative = 0)
            f_bc[0, :] = f_bc[1, :]  # Bottom boundary
            f_bc[-1, :] = f_bc[-2, :]  # Top boundary
            f_bc[:, 0] = f_bc[:, 1]  # Left boundary
            f_bc[:, -1] = f_bc[:, -2]  # Right boundary
        return f_bc
