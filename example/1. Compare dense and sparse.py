import sys
import os
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import time
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem
import matplotlib as mpl
from src.tools.utils import get_logger, load_config

config = load_config()
logging = get_logger()


def convergence_test(u_ex, poisson_solver, n, use_sparse=False):
    """Test convergence for n total points (including boundaries) and measure solving time"""
    # Generate grid for internal points
    h = 1.0 / (n - 1)
    nx = n - 2  # Number of interior points
    xi = np.linspace(h, 1 - h, nx)  # Interior points only
    yi = np.linspace(h, 1 - h, nx)  # Interior points only
    x, y = np.meshgrid(xi, yi)

    # Compute exact solution on internal points
    U_ex = u_ex(x, y)

    # Solve numerical solution and measure time
    start_time = time.time()
    U = poisson_solver.solve()
    end_time = time.time()
    solve_time = end_time - start_time

    # Compute error (L-infinity norm)
    err = np.max(np.abs(U_ex - U))

    # Assuming PoissonProblem has an attribute 'iterations' after solve()
    iterations = getattr(poisson_solver, 'iterations', np.nan)

    # After solving, capture iterations from the solver if available
    if hasattr(poisson_solver.solver, 'iterations'):
        poisson_solver.iterations = poisson_solver.solver.iterations
    else:
        poisson_solver.iterations = np.nan

    logging.info(
        f"Solver: {poisson_solver.solver_type}, Mode: {'Sparse' if use_sparse else 'Dense'}, N = {n}, h = {h:.6f}, Error = {err:.6e}, Solve Time = {solve_time:.6f} seconds, Iterations = {iterations}"
    )

    return {
        'U': U,
        'error': err,
        'time': solve_time,
        'iterations': iterations,  # Added iterations
        'x': x,
        'y': y,
        'N': n
    }


if __name__ == "__main__":
    ns = [10, 20, 40, 80]
    all_data_dense = []
    all_data_sparse = []

    # First: Compare Dense and Sparse for Direct Solver
    for n in ns:
        # Dense Solver
        logging.info(f"Starting Dense Solver for N={n}")
        poisson_solver_dense = PoissonProblem(
            Nx=n,
            Ny=n,
            f=f,
            u_ex=u_ex,
            grid_type="uniform",
            solver_type="direct",  # Direct solver for dense matrices
            boundary_condition_type="dirichlet",
            boundary_condition='dirichlet',  # Specify boundary condition type
            use_sparse=False)
        data_dense = convergence_test(u_ex,
                                      poisson_solver_dense,
                                      n + 1,
                                      use_sparse=False)
        data_dense[
            'solver'] = 'direct_dense'  # Add solver key for dense solver
        all_data_dense.append(data_dense)

        logging.info(f"Completed Dense Solver for N={n}")

        # Sparse Solver
        logging.info(f"Starting Sparse Solver for N={n}")
        poisson_solver_sparse = PoissonProblem(
            Nx=n,
            Ny=n,
            f=f,
            u_ex=u_ex,
            grid_type="uniform",
            solver_type="direct",
            boundary_condition_type="dirichlet",
            boundary_condition='dirichlet',  # Specify boundary condition type
            use_sparse=True)  # Enable sparse solver
        data_sparse = convergence_test(u_ex,
                                       poisson_solver_sparse,
                                       n + 1,
                                       use_sparse=True)
        data_sparse['solver'] = 'direct_sparse'
        all_data_sparse.append(data_sparse)

        logging.info(f"Completed Sparse Solver for N={n}")

    # Collect all data for plotting solutions
    combined_data = all_data_dense + all_data_sparse

    # **Combine Plot Solutions into Two Big Plots**
    # Separate dense and sparse data
    dense_data = [data for data in combined_data if 'dense' in data['solver']]
    sparse_data = [
        data for data in combined_data if 'sparse' in data['solver']
    ]

    # Plot all Dense Solutions and Exact Solution in one figure using enumeration for subplot indexing
    plt.figure(figsize=(12, 10))
    for idx, data in enumerate(dense_data, 1):
        ax = plt.subplot(2, 2, idx, projection='3d')
        # Plot Numerical Solution
        surf = ax.plot_surface(data['x'],
                               data['y'],
                               data['U'],
                               rstride=1,
                               cstride=1,
                               cmap=plt.cm.viridis,
                               alpha=0.7,
                               label='Numerical Solution')
        # Compute exact solution on the same grid
        u_exact = u_ex(data['x'], data['y'])
        # Plot Exact Solution as a wireframe
        exact_wire = ax.plot_wireframe(data['x'],
                                       data['y'],
                                       u_exact,
                                       color='red',
                                       linewidth=1.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u(x,y)$')
        ax.set_title(f"{data['solver'].capitalize()} Solver - N={data['N']}")
        # Create a custom legend with specific colors
        from matplotlib.patches import Patch
        viridis_color = mpl.cm.viridis(0.5)  # Middle color of viridis
        exact_color = 'red'  # Solid color for exact solution
        legend_elements = [
            Patch(facecolor=viridis_color,
                  edgecolor=viridis_color,
                  label='Numerical Solution'),
            Patch(facecolor=exact_color,
                  edgecolor=exact_color,
                  label='Exact Solution')
        ]
        ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig('doc/combined_dense_solutions.png')
    plt.show()

    # Plot all Sparse Solutions and Exact Solution in one figure using enumeration for subplot indexing
    plt.figure(figsize=(12, 10))
    for idx, data in enumerate(sparse_data, 1):
        ax = plt.subplot(2, 2, idx, projection='3d')
        # Plot Numerical Solution
        surf = ax.plot_surface(data['x'],
                               data['y'],
                               data['U'],
                               rstride=1,
                               cstride=1,
                               cmap=plt.cm.viridis,
                               alpha=0.7,
                               label='Numerical Solution')
        # Compute exact solution on the same grid
        u_exact = u_ex(data['x'], data['y'])
        # Plot Exact Solution as a wireframe
        exact_wire = ax.plot_wireframe(data['x'],
                                       data['y'],
                                       u_exact,
                                       color='red',
                                       linewidth=1.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u(x,y)$')
        ax.set_title(f"{data['solver'].capitalize()} Solver - N={data['N']}")
        # Create a custom legend with specific colors
        from matplotlib.patches import Patch
        viridis_color = mpl.cm.viridis(0.5)  # Middle color of viridis
        exact_color = 'red'  # Solid color for exact solution
        legend_elements = [
            Patch(facecolor=viridis_color,
                  edgecolor=viridis_color,
                  label='Numerical Solution'),
            Patch(facecolor=exact_color,
                  edgecolor=exact_color,
                  label='Exact Solution')
        ]
        ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig('doc/combined_sparse_solutions.png')
    plt.show()

    # **Compare Computation Time for Dense and Sparse Solvers**
    plt.figure(figsize=(12, 6))
    plt.plot([data['N'] for data in all_data_dense],
             [data['time'] for data in all_data_dense],
             marker='o',
             label='Direct Dense Solver')
    plt.plot([data['N'] for data in all_data_sparse],
             [data['time'] for data in all_data_sparse],
             marker='o',
             label='Direct Sparse Solver')
    plt.title('Computation Time: Dense vs Sparse Direct Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('doc/dense_vs_sparse_computation_time.png')
    plt.show()
