import sys
import os
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem
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


def compare_solvers(solver_types, n_list):
    """
    Compare errors and computation times of different solvers for grid sizes in n_list.
    """
    results = []
    for solver_name in solver_types:
        for n in n_list:
            logging.info(
                f"Initializing Solver: {solver_name}, Mode: {'Sparse' if solver_name in ["lu", "low_rank", "direct"] else 'Dense'}, N={n}"
            )
            poisson_solver = PoissonProblem(
                Nx=n,
                Ny=n,
                f=f,
                u_ex=u_ex,
                grid_type="uniform",
                solver_type=solver_name,
                boundary_condition_type="dirichlet",
                use_sparse=solver_name in ["lu", "low_rank"]  
            )
            
            data = convergence_test(u_ex,
                                    poisson_solver,
                                    n + 1,
                                    use_sparse=poisson_solver.use_sparse)
            err = data['error']
            iterations = data['iterations']  # Retrieve iterations
            logging.info(f"Completed Solver: {solver_name}, N={n}")

            results.append({
                'solver': solver_name,
                'N': n,
                'error': err,
                'time': data['time'],
                'iterations': iterations  # Include iterations
            })

    df = pd.DataFrame(results)
    logging.info("\nComparison of Different Solvers:")
    logging.info(df)

    return df


if __name__ == "__main__":
    ns = [10, 20, 40, 80]
    # Compare all solvers and store results in df_all_solvers
    solver_types_to_compare = [
        "gauss_seidel", "jacobi", "sor", "conjugate_gradient"
    ]
    df_all_solvers = compare_solvers(solver_types_to_compare, ns)

    # **Compare Speed and Error for All Methods**
    plt.figure(figsize=(14, 6))

    # Subplot for Computation Time of All Solvers
    plt.subplot(1, 2, 1)
    for solver, group in df_all_solvers.groupby('solver'):
        plt.plot(group['N'],
                 group['time'],
                 marker='o',
                 label=f'{solver.capitalize()} Solver')
    plt.xlabel('Number of Intervals N')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time of All Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Subplot for Error of All Solvers
    plt.subplot(1, 2, 2)
    for solver, group in df_all_solvers.groupby('solver'):
        plt.plot(group['N'],
                 group['error'],
                 marker='s',
                 label=f'{solver.capitalize()} Solver')
    plt.xlabel('Number of Intervals N')
    plt.ylabel('L-infinity Error')
    plt.title('Error of All Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(
        'doc/speed_error_comparison_all_solvers.png')  # Save the comparison plot
    plt.show()

    # **Plot Number of Iterations for Iterative Solvers**
    iterative_solvers = ["jacobi", "gauss_seidel", "sor", "conjugate_gradient"]
    iterative_solvers_df = df_all_solvers[df_all_solvers['solver'].isin(
        iterative_solvers)]

    # Pivot the filtered DataFrame for a clearer bar chart
    pivot_df = iterative_solvers_df.pivot(index='solver',
                                          columns='N',
                                          values='iterations')
    pivot_df.plot(kind='bar')
    plt.xlabel('Solver')
    plt.ylabel('Number of Iterations')
    plt.title('Number of Iterations for Iterative Solvers')
    plt.legend(title='Number of Intervals N')
    plt.tight_layout()
    plt.savefig('doc/iterations_bar_chart.png')  # Save the iterations bar chart
    plt.show()