import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as splin
from src.solver.solver_factory import SolverFactory
from src.grid.grid_factory import GridFactory
from src.boundary.boundary_condition_factory import BoundaryConditionFactory
from src.tools.sparse_matrix import SparseMatrix
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
import scipy.sparse as sp


class PoissonProblem:

    def __init__(self,
                 Nx,
                 Ny,
                 f,
                 u_ex,  # Add u_ex parameter
                 grid_type,
                 solver_type,
                 boundary_condition_type,
                 use_sparse=True,
                 alpha=0.0,  # New alpha parameter
                 preconditioner_type=None,  # Add preconditioner_type parameter
                 **solver_kwargs):  # Accept additional solver-specific arguments
        # Nx, Ny = number of intervals, so total grid points = Nx+1, Ny+1
        self.Nx, self.Ny = Nx, Ny  
        self.f = f
        self.u_ex = u_ex  # Store u_ex
        self.solver_type = solver_type  # Store solver_type
        # Create grid with Nx+1 and Ny+1 points
        self.grid = GridFactory.create_grid(grid_type, Nx+1, Ny+1)  # Include boundary points
        solver_config = SolverFactory.load_config()
        self.solver = SolverFactory.create_solver(
            solver_type=solver_type,
            preconditioner_type=preconditioner_type,  # Pass preconditioner_type directly
            tol=solver_config.get('tolerance', 1e-8),
            max_iter=solver_config.get('max_iterations', 1000),
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
            2 * np.ones(nx),       # Main diagonal
            -1 * np.ones(nx - 1)   # Upper diagonal
        ]
        offsets = [-1, 0, 1]
        T = sp.diags(diagonals, offsets, shape=(nx, nx), format='csr') / (h * h)

        # Identity matrix
        I = sp.identity(ny, format='csr')

        # 2D Laplacian using Kronecker product
        A = sp.kron(I, T) + sp.kron(T, I)
        # Add alpha * I to the operator
        A += self.alpha * sp.eye((self.Nx - 1) * (self.Ny - 1), format='csr')

        return A

    def solve(self):
        logging.info("PoissonProblem: Solving problem")
        h = 1.0 / self.Nx
        A = self.discretize_poisson(h)

        X, Y = self.grid.generate_grid()
        F_full = self.f(X, Y)
        F_bc = self.boundary_condition.apply(F_full, X, Y)

        b_full = -F_full  # PDE form: -Δu = f ⇒ Δu = -f
        b = b_full[1:-1, 1:-1].copy()

        # Incorporate homogeneous Dirichlet boundary conditions
        # Since u_boundary = 0, the subtraction has no effect, but structured for non-zero BCs
        b[0, :]    -= F_bc[0, 1:-1]    / (h * h)  # Bottom boundary
        b[-1, :]   -= F_bc[-1, 1:-1]   / (h * h)  # Top boundary
        b[:, 0]    -= F_bc[1:-1, 0]    / (h * h)  # Left boundary
        b[:, -1]   -= F_bc[1:-1, -1]   / (h * h)  # Right boundary

        logging.debug(f"Matrix A shape: {A.shape}, Vector b shape: {b.shape}")
        logging.info("PoissonProblem: Solving linear system")

        # Convert or solve depending on use_sparse
        if self.use_sparse:
            # Use sparse solver
            u_int = splin.spsolve(A, b.flatten())
        else:
            # Convert matrix to dense for scipy.linalg.solve
            A_dense = A.toarray()
            u_int = lin.solve(A_dense, b.flatten())

        return u_int.reshape((self.Nx - 1, self.Ny - 1))

    def solve_parallel(self):
        logging.info("PoissonProblem: Solving problem in parallel")
        X, Y = self.grid.generate_grid()
        Nx, Ny = self.Nx, self.Ny
        h = 1 / (Nx - 1)
        A = self.discretize_poisson(h)  # Use discretize_poisson instead of construct_matrix
        F = self.f(X, Y)
        F = self.boundary_condition.apply(F)
        b = F[1:-1, 1:-1].flatten()  # Extract internal points only

        if self.use_sparse:
            A = SparseMatrix.from_dense(A)

        if self.solver_type.lower() == "pinn":
            # PINN solver might not support multiprocessing in the same way
            u = self.solver.solve(self.f, self.boundary_condition)
        else:
            with mp.Pool(mp.cpu_count()) as pool:
                u = pool.apply(self.solver.solve, args=(A, b))

        u = u.reshape((Nx - 1, Ny - 1))
        return u, X, Y

    def solve_multithreaded(self):
        logging.info("PoissonProblem: Solving problem with multithreading")
        X, Y = self.grid.generate_grid()
        Nx, Ny = self.Nx, self.Ny
        h = 1 / (Nx - 1)
        A = self.discretize_poisson(h)  # Use discretize_poisson instead of construct_matrix
        F = self.f(X, Y)
        F = self.boundary_condition.apply(F)
        b = F[1:-1, 1:-1].flatten()  # Extract internal points only

        if self.use_sparse:
            A = SparseMatrix.from_dense(A)

        if self.solver_type.lower() == "pinn":
            # PINN solver might not support multithreading in the same way
            u = self.solver.solve(self.f, self.boundary_condition)
        else:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self.solver.solve, A, b)
                u = future.result()

        u = u.reshape((Nx - 1, Ny - 1))
        return u, X, Y

