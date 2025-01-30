import numpy as np
from src.solver.solver_factory import SolverFactory
from src.grid.grid_factory import GridFactory
from src.boundary.boundary_condition_factory import BoundaryConditionFactory
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
import scipy.sparse as sp
from src.tools.utils import load_config


class PoissonProblem:

    def __init__(
        self,
        Nx,
        Ny,
        f,
        u_ex,  # Add u_ex parameter
        grid_type,
        solver_type,
        boundary_condition_type,
        use_sparse=True,
        alpha=0.0,  # New alpha parameter
        **solver_kwargs):  # Accept additional solver-specific arguments
        # Nx, Ny = number of intervals, so total grid points = Nx+1, Ny+1
        self.Nx, self.Ny = Nx, Ny
        self.f = f
        self.u_ex = u_ex  # Store u_ex
        self.solver_type = solver_type  # Store solver_type
        # Create grid with Nx+1 and Ny+1 points
        self.grid = GridFactory.create_grid(grid_type, Nx + 1,
                                            Ny + 1)  # Include boundary points
        solver_config = load_config()
        self.solver = SolverFactory.create_solver(
            solver_type=solver_type,
            tol=float(solver_config.get('tolerance',
                                        1e-8)),  # Ensure tol is float
            max_iter=solver_config.get('max_iterations', 10000),
            **solver_kwargs)  # Pass additional arguments

        # Pass the boundary function 'g' (self.u_ex) to the factory
        self.boundary_condition = BoundaryConditionFactory.create_boundary_condition(
            boundary_condition_type,
            g=self.u_ex  # Pass u_ex as the boundary function
        )

        self.use_sparse = use_sparse
        self.alpha = alpha

    def discretize_poisson(self, h):
        """Construct the sparse matrix for the 2D Poisson equation with Dirichlet BCs using Kronecker products"""
        nx = self.Nx - 1  # Number of interior points in x direction
        ny = self.Ny - 1  # Number of interior points in y direction

        # 1D Tridiagonal matrix T
        diagonals = [
            -1 * np.ones(nx - 1),  # Lower diagonal
            2 * np.ones(nx),  # Main diagonal
            -1 * np.ones(nx - 1)  # Upper diagonal
        ]
        offsets = [-1, 0, 1]
        
        T = sp.diags(diagonals, offsets, shape=(nx, nx)) / (h * h)

        # Identity matrix
        I = sp.identity(ny)

        # 2D Laplacian using Kronecker product
        A = sp.kron(I, T) + sp.kron(T, I)
        # Add alpha * I to the operator
        A += self.alpha * sp.eye((self.Nx - 1) * (self.Ny - 1))
        logging.debug(f"A:{sp.isspmatrix(A)}")
        if self.use_sparse:
            A = sp.csr_matrix(A)
        else:
            A = A.toarray()
        return A

    def solve(self, mode='sequential'):
        logging.info("PoissonProblem: Solving problem")
        h = 1.0 / self.Nx
        A = self.discretize_poisson(h)

        X, Y = self.grid.generate_grid()
        F = self.f(X, Y)
        F = self.boundary_condition.apply(F)
        b = F[1:-1, 1:-1].flatten()
        logging.info("PoissonProblem: Solving linear system")

        if mode == 'sequential':
            u_int = self.solver.solve(A, b)
        elif mode == 'parallel':            
            if self.use_sparse:
                A = A.toarray()

            with mp.Pool(mp.cpu_count()) as pool:
                u_int = pool.apply(self.solver.solve, args=(A, b))
        elif mode == 'multithreaded':            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self.solver.solve, A, b)
                u_int = future.result()
        else:
            logging.error(f"Unknown solving mode: {mode}")
            raise ValueError(f"Unknown solving mode: {mode}")

        # Capture iterations if available
        if hasattr(self.solver, 'iterations'):
            self.iterations = self.solver.iterations
        else:
            self.iterations = 0

        return u_int.reshape((self.Nx - 1, self.Ny - 1))
