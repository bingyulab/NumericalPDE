import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import pandas as pd
import time
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem

# Configure logging to output to both the terminal and a log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('poisson_problem.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

config = yaml.load(open('./config.yaml'), Loader=yaml.FullLoader)


def plot2D(X, Y, Z, title=""):
    # Define a new figure with given size and resolution
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,  # Sampling rates for the x and y input data
        cmap=plt.cm.viridis)  # Use the new fancy colormap
    # Set initial view angle
    ax.view_init(30, 225)
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()


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
    logging.info(
        f"Solver: {poisson_solver.solver_type}, Mode: {'Sparse' if use_sparse else 'Dense'}, N = {n}, h = {h:.6f}, Error = {err:.6e}, Solve Time = {solve_time:.6f} seconds"
    )

    return {
        'U': U,
        'error': err,
        'time': solve_time,
        'x': x,  
        'y': y,  
        'N': n    
    }


def compare_solvers(solver_types, preconditioner_types, n_list):
    """
    Compare errors and computation times of different solvers for grid sizes in n_list.
    """
    results = []
    for solver_name in solver_types:
        for preconditioner_name in preconditioner_types:
            for n in n_list:
                logging.info(f"Initializing Solver: {solver_name}, Preconditioner: {preconditioner_name}, Mode: {'Sparse' if 'sparse' in solver_name else 'Dense'}, N={n}")
                poisson_solver = PoissonProblem(
                    Nx=n,
                    Ny=n,
                    f=f,
                    u_ex=u_ex,
                    grid_type="uniform",
                    solver_type=solver_name,
                    boundary_condition_type="dirichlet",
                    use_sparse=solver_name != "direct",  # Example condition to set mode
                    preconditioner_type=preconditioner_name  # Pass preconditioner type
                )
                try:
                    start_time = time.time()
                    data = convergence_test(u_ex,
                                            poisson_solver,
                                            n + 1,
                                            use_sparse=poisson_solver.use_sparse)
                    elapsed = time.time() - start_time
                    err = data['error']
                    logging.info(f"Completed Solver: {solver_name}, Preconditioner: {preconditioner_name}, Mode: {'Sparse' if poisson_solver.use_sparse else 'Dense'}, N={n}")
                except Exception as e:
                    logging.error(f"Solver {solver_name} with N={n} failed: {e}")
                    err = np.nan
                    elapsed = np.nan

                results.append({
                    'solver': solver_name,
                    'preconditioner': preconditioner_name,
                    'N': n,
                    'error': err,
                    'time': elapsed
                })

    # Create a DataFrame to summarize
    df = pd.DataFrame(results)
    logging.info("\nComparison of Different Solvers:")
    logging.info(df)
    df.to_csv('solver_comparison_results.csv', index=False)

    return df  # Return DataFrame for further plotting


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
            solver_type="direct",
            boundary_condition_type="dirichlet",
            use_sparse=False)
        data_dense = convergence_test(u_ex,
                                      poisson_solver_dense,
                                      n + 1,
                                      use_sparse=False)
        data_dense['solver'] = 'direct_dense'  # Add solver key for dense solver
        all_data_dense.append(data_dense)
        
        # Visualize the solution
        plot2D(
            data_dense['x'],
            data_dense['y'],
            data_dense['U'],
            title=f'Numerical Solution (N={n})'
        )
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
            use_sparse=True)  # Enable sparse solver
        data_sparse = convergence_test(u_ex,
                                       poisson_solver_sparse,
                                       n + 1,
                                       use_sparse=True)
        data_sparse['solver'] = 'direct_sparse'  # Changed from 'direct' to 'direct_sparse'
        all_data_sparse.append(data_sparse)
        
        # Visualize the sparse solution
        plot2D(
            data_sparse['x'],
            data_sparse['y'],
            data_sparse['U'],
            title=f'Numerical Solution with Sparse Solver (N={n})'
        )
        logging.info(f"Completed Sparse Solver for N={n}")

    # Plot convergence for Dense vs Sparse Direct Solver
    if len(all_data_dense) >= 1 and len(all_data_sparse) >= 1:
        # Combine dense and sparse data into a single DataFrame
        combined_data = all_data_dense + all_data_sparse
        df_combined = pd.DataFrame(combined_data)
        
        plt.figure(figsize=(10, 7))
        markers = {'direct_dense': 'o-', 'direct_sparse': 's-'}
        for solver, group in df_combined.groupby('solver'):
            plt.loglog(
                1.0 / group['N'],
                group['error'],
                markers.get(solver, '^-'),
                label=f'{solver.capitalize()} Solver Error'
            )
        plt.loglog(
            np.array([1.0 / n for n in ns]),
            1.0 / np.array(ns)**2 * df_combined[df_combined['solver'] == 'direct_dense']['error'].iloc[0],
            '--',
            label='O(h^2)'
        )
        plt.xlabel('Step size h')
        plt.ylabel('L-infinity Error')
        plt.title('Convergence Plot of Dense vs Sparse Direct Solver')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig('convergence_plot_dense_vs_sparse.png')  # Save the plot
        plt.show()

        plt.figure(figsize=(10, 7))
        for solver, group in df_combined.groupby('solver'):
            plt.loglog(
                group['N'],
                group['time'],
                markers.get(solver, '^-'),
                label=f'{solver.capitalize()} Solver Time'
            )
        plt.xlabel('Number of Intervals N')
        plt.ylabel('Computation Time (s)')
        plt.title('Computation Time Comparison of Dense vs Sparse Direct Solver')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig('computation_time_plot_dense_vs_sparse.png')  # Save the plot
        plt.show()

    # Second: Compare All Methods
    solver_types_to_compare = [
        "direct", "fst", "jacobi", "gauss_seidel", "sor", "lu", "cholesky",
        "low_rank", "conjugate_gradient", "multigrid", "graph",
        "priority_queue", "pinn"
    ]
    df_all_solvers = compare_solvers(solver_types_to_compare, ["none"], [10, 20, 40])

    # Plotting Errors of All Solvers
    plt.figure(figsize=(10, 7))
    for solver, group in df_all_solvers.groupby('solver'):
        plt.loglog(
            1.0 / group['N'],
            group['error'],
            marker='o',
            label=f'{solver.capitalize()} Solver Error'
        )
    plt.loglog(
        np.array([1.0 / n for n in [10, 20, 40]]),
        1.0 / np.array([10, 20, 40])**2 * df_all_solvers[df_all_solvers['solver'] == 'direct']['error'].iloc[0],
        '--',
        label='O(h^2)'
    )
    plt.xlabel('Step size h')
    plt.ylabel('L-infinity Error')
    plt.title('Convergence Plot of All Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('convergence_plot_all_methods.png')  # Save the plot
    plt.show()

    # Plotting Computation Times of All Solvers
    plt.figure(figsize=(10, 7))
    for solver, group in df_all_solvers.groupby('solver'):
        plt.loglog(
            group['N'],
            group['time'],
            marker='s',
            label=f'{solver.capitalize()} Solver Time'
        )
    plt.xlabel('Number of Intervals N')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison of All Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('computation_time_plot_all_methods.png')  # Save the plot
    plt.show()

    # Third: Compare Best Model with Various Preconditioner Types
    best_solver_type = "conjugate_gradient"  # Example best solver type
    preconditioner_types_to_compare = ["none", "jacobi", "incomplete_cholesky"]
    df_best_solver = compare_solvers([best_solver_type], preconditioner_types_to_compare, [10, 20, 40])

    # Plotting Errors of Best Solver with Various Preconditioners
    plt.figure(figsize=(10, 7))
    for solver, group in df_best_solver.groupby('preconditioner'):
        plt.loglog(
            1.0 / group['N'],
            group['error'],
            marker='o',
            label=f'{best_solver_type.capitalize()} Solver with {solver.capitalize()} Preconditioner Error'
        )
    plt.loglog(
        np.array([1.0 / n for n in [10, 20, 40]]),
        1.0 / np.array([10, 20, 40])**2 * df_best_solver[df_best_solver['preconditioner'] == 'none']['error'].iloc[0],
        '--',
        label='O(h^2)'
    )
    plt.xlabel('Step size h')
    plt.ylabel('L-infinity Error')
    plt.title(f'Convergence Plot of {best_solver_type.capitalize()} Solver with Various Preconditioners')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'convergence_plot_{best_solver_type}_preconditioners.png')  # Save the plot
    plt.show()

    # Plotting Computation Times of Best Solver with Various Preconditioners
    plt.figure(figsize=(10, 7))
    for solver, group in df_best_solver.groupby('preconditioner'):
        plt.loglog(
            group['N'],
            group['time'],
            marker='s',
            label=f'{best_solver_type.capitalize()} Solver with {solver.capitalize()} Preconditioner Time'
        )
    plt.xlabel('Number of Intervals N')
    plt.ylabel('Computation Time (s)')
    plt.title(f'Computation Time Comparison of {best_solver_type.capitalize()} Solver with Various Preconditioners')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'computation_time_plot_{best_solver_type}_preconditioners.png')  # Save the plot
    plt.show()
