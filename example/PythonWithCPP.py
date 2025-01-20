import sys
import os
import numpy as np
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cpp.build import poisson_solvers

def solve_with_jacobi(A, b, tol=1e-8, max_iter=1000):
    solver = poisson_solvers.JacobiSolver(tol, max_iter)
    return solver.solve(A, b)

def test_jacobi_solver():
    # Example system Ax = b
    A = np.array([[4, 1, 0, 0],
                  [1, 4, 1, 0],
                  [0, 1, 4, 1],
                  [0, 0, 1, 3]], dtype=float)
    b = np.array([15, 10, 10, 10], dtype=float)
    
    # Expected solution using NumPy
    x_expected = np.linalg.solve(A, b)
    
    # Call the C++ Jacobi solver
    x_computed = solve_with_jacobi(A, b)
    
    # Convert to numpy array for comparison
    x_computed = np.array(x_computed)
    
    # Compute the difference
    difference = np.linalg.norm(x_expected - x_computed, ord=np.inf)
    
    print("Expected Solution:", x_expected)
    print("Computed Solution:", x_computed)
    print("Difference (Infinity Norm):", difference)
    
    assert difference < 1e-5, "Jacobi Solver test failed!"
    print("Jacobi Solver test passed!")

if __name__ == "__main__":
    test_jacobi_solver()


