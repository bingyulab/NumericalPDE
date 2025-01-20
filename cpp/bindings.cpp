#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Added to enable Eigen type conversions
#include "jacobi_solver.h"  
#include "base_solver.h" 

namespace py = pybind11;

PYBIND11_MODULE(poisson_solvers, m) {
    // Binding for Solver (abstract class)
    py::class_<Solver, std::shared_ptr<Solver>>(m, "Solver")
        .def("solve", &Solver::solve);  // Bind the pure virtual method

    // Binding for JacobiSolver with inheritance
    py::class_<JacobiSolver, Solver, std::shared_ptr<JacobiSolver>>(m, "JacobiSolver")
        .def(py::init<double, int>(), py::arg("tol") = 1e-8, py::arg("max_iter") = 10000)
        .def("solve", 
             // Overloaded solve methods returning pair
             static_cast<std::pair<std::vector<double>, int> (JacobiSolver::*)(const Eigen::SparseMatrix<double>&, const std::vector<double>&)>(&JacobiSolver::solve),
             "Solve the system using a sparse matrix and return (solution, iterations)")
        .def("solve", 
             static_cast<std::pair<std::vector<double>, int> (JacobiSolver::*)(const Eigen::MatrixXd&, const std::vector<double>&)>(&JacobiSolver::solve),
             "Solve the system using a dense matrix and return (solution, iterations)");

    // ... Bind other solvers if necessary ...
}
