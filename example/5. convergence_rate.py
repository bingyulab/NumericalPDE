import sys
import os
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.boundary.function import u_ex, f
from src.problem.poisson_problem import PoissonProblem
import matplotlib.pyplot as plt


def generate_grid(Nx, Ny):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def finite_difference(N, h):
    #  [0 − 1 1], [−1 1 0], or [−1 0 1] for the first derivatives
    #  [1 − 2 1] for the second derivatives
    e = np.ones(N)
    D = (np.diag(-2 * e) + np.diag(e[:-1], k=-1) + np.diag(e[:-1], k=1))
    D /= h**2
    return D


def discretize_poisson(Nx, Ny, h):
    Dx = finite_difference(Nx - 1, h)
    Dy = finite_difference(Ny - 1, h)
    Ix = np.eye(Nx - 1)
    Iy = np.eye(Ny - 1)
    L = np.kron(Iy, Dx) + np.kron(Dy, Ix)
    return L


def possion_solver(Nx, Ny, f):
    h = 1 / (Nx - 1)
    L = discretize_poisson(Nx, Ny, h)
    b = f.flatten()
    u = np.linalg.solve(L, b)
    return u.reshape(Nx - 1, Ny - 1)


def apply_boundary_conditions(f, boundary_value=0):
    # Create a new array with boundary conditions applied
    f_bc = f.copy()
    f_bc[0, :] = boundary_value
    f_bc[-1, :] = boundary_value
    f_bc[:, 0] = boundary_value
    f_bc[:, -1] = boundary_value
    return f_bc


def validate_solution(Nx, Ny):
    X, Y = generate_grid(Nx + 1, Ny + 1)  # Include boundary points
    u_exact = u_ex(X, Y)
    f_val = f(X, Y)
    f_bc = apply_boundary_conditions(f_val, boundary_value=0)
    problem = PoissonProblem(
        Nx=Nx,
        Ny=Ny,
        f=f,
        u_ex=u_ex,
        grid_type="uniform",
        solver_type="direct",
        boundary_condition_type="dirichlet",
        boundary_condition='dirichlet')  # Specify boundary condition type
    u_numerical = problem.solve()
    error = np.max(np.abs(u_exact[1:-1, 1:-1] - u_numerical))
    return error


def validate_homogeneous_solution(Nx, Ny):
    """Validate the solver with homogeneous Dirichlet boundary conditions (all zeros)."""
    X, Y = generate_grid(Nx + 1, Ny + 1)  # Include boundary points
    u_exact = np.zeros_like(X)  # Exact solution is zero everywhere
    f_val = f(X, Y)  # Source term remains the same
    f_bc = apply_boundary_conditions(
        f_val, boundary_value=0)  # Homogeneous Dirichlet BCs
    problem = PoissonProblem(
        Nx=Nx,
        Ny=Ny,
        f=f,
        u_ex=lambda x, y: 0 * x,  # Exact solution is zero
        grid_type="uniform",
        solver_type="direct",
        boundary_condition_type="dirichlet",
        boundary_condition='dirichlet')  # Specify boundary condition type
    u_numerical = problem.solve()
    error = np.max(np.abs(u_exact[1:-1, 1:-1] - u_numerical))
    return error


def convergence_plot():
    errors_non_homogeneous = []
    hs_non_homogeneous = []
    errors_homogeneous = []
    hs_homogeneous = []

    for N in [10, 20, 40, 80]:
        h = 1 / N
        # Non-Homogeneous Dirichlet BCs
        error_non_homog = validate_solution(N, N)
        errors_non_homogeneous.append(error_non_homog)
        hs_non_homogeneous.append(h)

        # Homogeneous Dirichlet BCs
        error_homog = validate_homogeneous_solution(N, N)
        errors_homogeneous.append(error_homog)
        hs_homogeneous.append(h)

    print("Step sizes (h):", hs_non_homogeneous)
    print("Errors (Non-Homogeneous Dirichlet):", errors_non_homogeneous)
    print("Errors (Homogeneous Dirichlet):", errors_homogeneous)

    plt.loglog(hs_non_homogeneous,
               errors_non_homogeneous,
               marker='o',
               label='Non-Homogeneous Dirichlet')
    # plt.loglog(hs_homogeneous, errors_homogeneous, marker='s', label='Homogeneous Dirichlet')
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('Convergence Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('doc/convergence_plots.png')
    plt.show()


def converge_plot():
    errors = []
    hs = []
    for N in [10, 20, 40, 80]:
        h = 1 / (N - 1)
        problem = PoissonProblem(Nx=N,
                                 Ny=N,
                                 f=f,
                                 u_ex=u_ex,
                                 grid_type="uniform",
                                 solver_type="direct",
                                 boundary_condition_type="dirichlet")
        X, Y = generate_grid(N - 1, N - 1)
        u_exact = u_ex(X, Y)

        # f_bc = apply_boundary_conditions(f)
        u_numerical = problem.solve()
        error = np.max(np.abs(u_exact - u_numerical))
        errors.append(error)
        hs.append(h)
    print(hs, errors)
    print([np.log(x) for x in hs], [np.log(x) for x in errors])
    plt.loglog(hs, errors, marker='o')
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('Convergence plot')
    plt.savefig('doc/converge_plots.png')
    plt.show()


if __name__ == "__main__":
    converge_plot()
    convergence_plot()
