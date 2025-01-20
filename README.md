# Project of Numerical PDE

## Installation

### Prerequisites

Ensure the following tools and libraries are installed on your system:

1. **CMake**: Used for managing the build process.
    - **Installation**:
        - **macOS**: `brew install cmake`
        - **Ubuntu**: `sudo apt-get install cmake`
        - **Windows**: Download from [CMake Official Website](https://cmake.org/download/).

2. **Eigen**: A C++ template library for linear algebra.
    - **Installation**:
        - **macOS**: `brew install eigen`
        - **Ubuntu**: `sudo apt-get install libeigen3-dev`
        - **Windows**: Download from [Eigen Official Website](https://gitlab.com/libeigen/eigen).

3. **PyBind11**: Seamlessly binds C++ and Python.
    - **Installation**:
        - **Using pip**: `pip install pybind11`
        - **From Source**: Clone the repository from [PyBind11 GitHub](https://github.com/pybind/pybind11) and follow the build instructions.

4. **Make**: A build automation tool.
    - **Installation**:
        - **macOS**: Included with Xcode Command Line Tools.
        - **Ubuntu**: `sudo apt-get install build-essential`
        - **Windows**: Install via [MinGW](http://www.mingw.org/) or use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install).

### Python Dependencies

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Compiling the C++ Code

1. **Navigate to the Project Directory**:

    ```bash
    cd /Users/leonjiang/Documents/Projects/Project-of-2D-Poisson-problem/
    ```

2. **Build the Project Using Makefile**:

    ```bash
    make
    ```

    This will compile the C++ code and build the PyBind11 module.

3. **Running the Examples**:

    ```bash
    python example/pinn.py
    python example/1.compare_dense_and_sparse.py
    # ... other example scripts ...
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