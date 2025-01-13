from abc import ABC, abstractmethod
import numpy as np

class Grid(ABC):
    @abstractmethod
    def generate_grid(self):
        pass

class UniformGrid(Grid):
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

    def generate_grid(self):
        x = np.linspace(0, 1, self.Nx)
        y = np.linspace(0, 1, self.Ny)
        X, Y = np.meshgrid(x, y)
        return X, Y
