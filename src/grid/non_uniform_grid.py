import numpy as np
from src.grid.base_grid import BaseGrid


class NonUniformGrid(BaseGrid):

    def __init__(self, Nx, Ny):
        super().__init__(Nx, Ny)  # Ensure proper initialization

    def generate_grid(self):
        # Example non-uniform spacing using a quadratic distribution
        x = np.linspace(0, 1, self.Nx)**2
        y = np.linspace(0, 1, self.Ny)**2
        X, Y = np.meshgrid(x, y)
        return X, Y
