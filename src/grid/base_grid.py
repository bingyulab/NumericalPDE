import abc

class BaseGrid(abc.ABC):
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

    @abc.abstractmethod
    def generate_grid(self):
        """
        Generate the grid points.

        Returns:
            X (np.ndarray): Grid points in the x-direction.
            Y (np.ndarray): Grid points in the y-direction.
        """
        pass