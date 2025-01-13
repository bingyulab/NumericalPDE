import numpy as np
from src.problem.poisson_problem import PoissonProblem

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
    X, Y = generate_grid(Nx - 1, Ny - 1)
    u_exact = np.sin(np.pi * X)**2 * np.sin(np.pi * Y)**2
    f = 2 * np.pi**2 * (np.sin(np.pi * X)**2 * np.sin(np.pi * Y)**2)
    f_bc = apply_boundary_conditions(f)
    u_numerical = possion_solver(Nx, Ny, f_bc)
    error = np.max(np.abs(u_exact - u_numerical))
    return error

def convergence_plot():
    import matplotlib.pyplot as plt
    errors = []
    hs = []
    for N in [10, 20, 40, 80]:
        h = 1 / (N - 1)
        problem = PoissonProblem(N, N, grid_type="uniform", solver_type="direct", boundary_condition_type="dirichlet")
        u_numerical, X, Y = problem.solve()
        u_exact = np.sin(np.pi * X)**2 * np.sin(np.pi * Y)**2
        error = np.max(np.abs(u_exact - u_numerical))
        errors.append(error)
        hs.append(h)
    plt.loglog(hs, errors, marker='o')
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('Convergence plot')
    plt.show()

if __name__ == "__main__":
    convergence_plot()

