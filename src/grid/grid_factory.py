from src.grid.uniform_grid import UniformGrid
from src.grid.non_uniform_grid import NonUniformGrid  # Added import


class GridFactory:

    @staticmethod
    def create_grid(grid_type, Nx, Ny):
        if grid_type == "uniform":
            return UniformGrid(Nx, Ny)
        elif grid_type == "non_uniform":
            return NonUniformGrid(Nx, Ny)  # Added support
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
