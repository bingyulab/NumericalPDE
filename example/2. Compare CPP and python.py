import sys
import os
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import pandas as pd
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem
import timeit
from src.tools.utils import get_logger, load_config

config = load_config()
logging = get_logger()


def benchmark_solver(poisson_solver, description):
    """
    Benchmark the given Poisson solver and log the execution time.
    """
    start_time = timeit.default_timer()
    U = poisson_solver.solve()
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    logging.info(
        f"{description} took {elapsed_time:.6f} seconds with {poisson_solver.iterations} iterations."
    )
    return elapsed_time


if __name__ == "__main__":
    ns = [10, 20, 40, 80]
    # Benchmarking code
    solver_types = ["jacobi_cpp", "jacobi"]  # Updated to include Python solver
    results = []

    for solver_type in solver_types:
        for n in ns:
            logging.info(f"Starting {solver_type} solver for N={n}")
            poisson_solver = PoissonProblem(
                Nx=n,
                Ny=n,
                f=f,
                u_ex=u_ex,
                grid_type="uniform",
                solver_type=solver_type,
                boundary_condition_type="dirichlet",
                use_sparse=False  # Assuming dense for comparison
            )
            elapsed = benchmark_solver(poisson_solver, f"{solver_type} Solver")
            results.append({
                'solver': solver_type,
                'N': n,
                'time': elapsed,
                'iterations': poisson_solver.iterations
            })
            logging.info(f"Completed {solver_type} solver for N={n}")

    # Convert results to DataFrame for analysis
    df_benchmark = pd.DataFrame(results)
    logging.info("\nBenchmark Results:")
    logging.info(df_benchmark)

    # Plot Benchmark Results
    plt.figure(figsize=(10, 6))
    for solver in solver_types:
        subset = df_benchmark[df_benchmark['solver'] == solver]
        plt.plot(subset['N'], subset['time'], marker='o', label=solver)
    plt.xlabel('Number of Intervals N')
    plt.ylabel('Computation Time (s)')
    plt.title('Benchmark: C++ JacobiSolver vs. PythonJacobiSolver')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('doc/jacobi_benchmark_comparison.png')
    plt.show()
    
    
    # Sample data
    plt.figure(figsize=(8,6))
    plt.loglog(h, error, marker='o', linestyle='-')
    plt.xlabel('Step size h')
    plt.ylabel('Error (L-Infinity Norm)')
    plt.title('Log-Log Convergence Plot')
    plt.grid(True, which="both", ls="--")    
    plt.savefig('doc/Log-Log Convergence Plot.png')
    plt.show()
