import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from typing import NamedTuple, Callable
import numpy.typing as npt

# Define data structures
Mesh = NamedTuple(
    "Mesh", (("geometry", npt.NDArray[np.float64]), ("topology", npt.NDArray[np.int32]))
)

FunctionSpace = NamedTuple(
    "FunctionSpace",
    (
        ("mesh", Mesh),
        ("dofmap", npt.NDArray[np.int32]),
        ("size", int),
    ),
)

# Define PDE parameter and exact solution
c = 1.0  # example constant
def f_source(x: float) -> float:
    """Right-hand side f(x) = c^2 * sin(c x)."""
    return c**2 * np.sin(c * x)

def u_exact_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Exact solution u(x) = sin(c x)."""
    return np.sin(c * x)

# Quadrature rule: 2-point Gauss on [0,1]
quadrature_points = np.array([0.5 - 1/(2*np.sqrt(3)), 0.5 + 1/(2*np.sqrt(3))])
quadrature_weights = np.array([1.0, 1.0])

def create_unit_interval_mesh(num_cells: int) -> Mesh:
    """
    Generate a 1D uniform mesh on the unit interval [0,1].

    Args:
        num_cells: Number of cells (elements).

    Returns:
        Mesh: A NamedTuple with 'geometry' (node coordinates) and 'topology' (cell-node indices).
    """
    # Node coordinates (N+1 nodes)
    geometry = np.linspace(0.0, 1.0, num_cells + 1).reshape(-1, 1)
    # Each cell has two nodes: i and i+1
    topology = np.vstack([
        np.arange(num_cells, dtype=np.int32),
        np.arange(1, num_cells + 1, dtype=np.int32)
    ]).T
    return Mesh(geometry=geometry, topology=topology)

def assemble_stiffness(
    fs: FunctionSpace, cell_stiffness_fn: Callable[[float, float], np.ndarray]
) -> scipy.sparse.lil_matrix:
    """
    Assemble the global stiffness matrix.

    Args:
        fs: FunctionSpace containing mesh and dofmap.
        cell_stiffness_fn: Function computing local stiffness for a cell [a,b].

    Returns:
        K: Global stiffness matrix in LIL format.
    """
    K = scipy.sparse.lil_matrix((fs.size, fs.size))
    for cell in range(fs.mesh.topology.shape[0]):
        dofs = fs.dofmap[cell]
        coords = fs.mesh.geometry[dofs].flatten()
        k_local = cell_stiffness_fn(coords[0], coords[1])
        K[np.ix_(dofs, dofs)] += k_local
    return K

def assemble_load(
    fs: FunctionSpace, cell_load_fn: Callable[[float, float], np.ndarray]
) -> np.ndarray:
    """
    Assemble the global load vector.

    Args:
        fs: FunctionSpace containing mesh and dofmap.
        cell_load_fn: Function computing local load for a cell [a,b].

    Returns:
        f: Global load vector.
    """
    f = np.zeros(fs.size)
    for cell in range(fs.mesh.topology.shape[0]):
        dofs = fs.dofmap[cell]
        a, b = fs.mesh.geometry[dofs].flatten()
        f_local = cell_load_fn(a, b)
        f[dofs] += f_local
    return f

def apply_boundary_conditions(
    dofs: npt.NDArray[np.int32], 
    K: scipy.sparse.lil_matrix, 
    f: npt.NDArray[np.float64],
    value: float = 0.0
):
    """
    Apply homogeneous Dirichlet BCs at specified DOFs.

    Args:
        dofs: Indices where u = value is enforced.
        K: Global stiffness matrix (modified in-place).
        f: Global load vector (modified in-place).
        value: Prescribed value (default 0).
    """
    n = K.shape[0]
    for dof in dofs:
        # Zero row and column
        K.rows[dof] = []
        K.data[dof] = []
        for i in range(n):
            row = K.rows[i]
            if dof in row:
                idx = row.index(dof)
                row.pop(idx)
                K.data[i].pop(idx)
        # Set diagonal = 1 and RHS = value
        K[dof, dof] = 1.0
        f[dof] = value

def cell_stiffness_quadrature(a: float, b: float) -> np.ndarray:
    """
    Compute local stiffness matrix on [a,b] via quadrature.

    Args:
        a, b: Endpoints of the cell.

    Returns:
        A: 2x2 local stiffness matrix.
    """
    h = b - a
    dphi_hat = np.array([-1.0, 1.0])
    A = np.zeros((2, 2))
    for w_q, xhat in zip(quadrature_weights, quadrature_points):
        for i in range(2):
            for j in range(2):
                A[i, j] += w_q * (dphi_hat[i] * dphi_hat[j] / h)
    return A

def cell_load(a: float, b: float) -> np.ndarray:
    """
    Compute local load vector on [a,b] via quadrature.

    Args:
        a, b: Endpoints of the cell.

    Returns:
        f_cell: 2-entry local load vector.
    """
    h = b - a
    f_cell = np.zeros(2)
    for w_q, xhat in zip(quadrature_weights, quadrature_points):
        x = a + h * xhat
        phi_vals = np.array([1 - xhat, xhat])
        f_val = f_source(x)
        f_cell += w_q * f_val * phi_vals
    return f_cell * h

def solve(num_cells: int, return_error: bool = False):
    """
    Solve -u'' = f on [0,1] with u(0)=u(1)=0 using P1 FEM.

    Args:
        num_cells: Number of uniform cells.
        return_error: If True, also compute H1 error.

    Returns:
        x: Node coordinates.
        u_h: FEM solution.
        error (optional): H1 seminorm error ||u - u_h||_H1^2.
    """
    mesh = create_unit_interval_mesh(num_cells)
    fs = FunctionSpace(mesh=mesh, dofmap=mesh.topology.copy(), size=mesh.geometry.shape[0])

    K = assemble_stiffness(fs, cell_stiffness_quadrature)
    f_vec = assemble_load(fs, cell_load)

    boundary_dofs = np.array([0, fs.size-1], dtype=np.int32)
    apply_boundary_conditions(boundary_dofs, K, f_vec)

    u_h = scipy.sparse.linalg.spsolve(K.tocsr(), f_vec)

    if return_error:
        x = mesh.geometry.flatten()
        u_ex = u_exact_func(x)
        e = u_ex - u_h
        error = e @ (K.tocsr().dot(e))
        return x, u_h, error

    return mesh.geometry.flatten(), u_h

# Example: plot solutions on refined meshes
plt.figure()
for N in [4, 8, 16, 32, 64]:
    x, u = solve(N)
    plt.plot(x, u, label=f"{N} cells")
plt.legend()
plt.title("FEM solutions on refined meshes")
plt.xlabel("x"); plt.ylabel("u")
plt.show()

# Compute and plot error vs h
hs = []
errors = []
for N in [4, 8, 16, 32, 64, 128]:
    h = 1.0 / N
    _, _, err = solve(N, return_error=True)
    hs.append(h)
    errors.append(err)

plt.figure()
plt.loglog(hs, errors, "o-", label=r"$\|I_hu - u_h\|_{H^1}^2$")
plt.xlabel("h"); plt.ylabel("Error")
plt.grid(True, which="both")
plt.legend()
plt.show()

