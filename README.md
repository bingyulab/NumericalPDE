## Installation

Clone the repository and install the package using `pip`:

```bash
git clone https://github.com/username/Project-of-2D-Poisson-problem.git
cd Project-of-2D-Poisson-problem
pip install .
```

Alternatively, install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the package:

```python
from poisson_solver import PoissonProblem, SolverFactory

# Define problem parameters
N = 50
solver_type = 'conjugate_gradient'
preconditioner = 'jacobi'

# Initialize and solve the problem
problem = PoissonProblem(N, N, grid_type="uniform", solver_type=solver_type, boundary_condition_type="dirichlet", use_sparse=True)
u_numerical, X, Y = problem.solve()

# Analyze results
error = problem.compute_error(u_numerical)
print(f"Maximum error: {error}")
```

## Running Tests

Execute all unit tests using:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! Please open issues and submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.