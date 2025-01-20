import sys
import os
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import logging
import time
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem
from src.tools.utils import get_logger, load_config

config = load_config()
logging = get_logger()

if __name__ == "__main__":
    # Plot Comparison of Best Method Using Multithreaded and Parallel
    # Assuming 'conjugate_gradient' is the best-performing solver
    best_solver = "conjugate_gradient"
    n_best = 60

    logging.info(
        f"Comparing multithreaded and parallel solutions for solver: {best_solver}, N={n_best}"
    )
    poisson_solver = PoissonProblem(Nx=n_best,
                                    Ny=n_best,
                                    f=f,
                                    u_ex=u_ex,
                                    grid_type="uniform",
                                    solver_type=best_solver,
                                    boundary_condition_type="dirichlet",
                                    use_sparse=True)

    # Solve using sequential
    logging.info("Starting parallel solver")
    plt.tight_layout()
    start_time_sequential = time.time()
    U_sequential = poisson_solver.solve(mode='sequential')
    end_time_sequential = time.time()
    time_sequential = end_time_sequential - start_time_sequential
    logging.info(
        f"Sequential solver completed in {time_sequential:.6f} seconds")

    # Solve using parallel
    logging.info("Starting parallel solver")
    plt.tight_layout()
    start_time_parallel = time.time()
    U_parallel = poisson_solver.solve(mode='parallel')
    end_time_parallel = time.time()
    time_parallel = end_time_parallel - start_time_parallel
    logging.info(f"Parallel solver completed in {time_parallel:.6f} seconds")

    # Solve using multithreaded
    logging.info("Starting multithreaded solver")
    start_time_multithreaded = time.time()
    U_multithreaded = poisson_solver.solve(mode='multithreaded')
    end_time_multithreaded = time.time()
    time_multithreaded = end_time_multithreaded - start_time_multithreaded
    logging.info(
        f"Multithreaded solver completed in {time_multithreaded:.6f} seconds")

    # **Plot Comparison of Speed: Parallel, Multithreaded, and Original**
    methods = [best_solver, 'Parallel', 'Multithreaded']

    times = [time_sequential, time_parallel, time_multithreaded]
    logging.info(f"Times: {times}")
    plt.bar(methods, times, color=['blue', 'green', 'red'])
    plt.xlabel('Method')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison: Original vs Parallel vs Multithreaded')
    plt.tight_layout()
    plt.savefig('doc/parallel_multithreaded_speed_comparison.png') 
    plt.show()
