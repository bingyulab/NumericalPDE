#include "jacobi_solver.h"
#include <iostream>
#include <cmath>

JacobiSolver::JacobiSolver(double tol, int max_iter)
    : tolerance(tol), max_iterations(max_iter) {}

// Implementation of the solve method for dense matrices
std::pair<std::vector<double>, int> JacobiSolver::solve(const Eigen::MatrixXd& A, const std::vector<double>& b) {
    int n = A.rows();
    std::vector<double> x(n, 0.0);
    std::vector<double> x_new(n, 0.0);
    int iter;
    
    for (iter = 0; iter < max_iterations; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            double diag = A(i, i);
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += A(i, j) * x[j];
                }
            }
            if (diag == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << std::endl;
                return {x_new, iter};
            }
            x_new[i] = (b[i] - sigma) / diag;
        }

        // Compute the infinity norm of (x_new - x)
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff = std::max(diff, std::abs(x_new[i] - x[i]));
        }

        // Check for convergence
        if (diff < tolerance) {
            return {x_new, iter + 1};
        }

        x = x_new;
    }

    return {x_new, iter};
}

// Implementation of the solve method for sparse matrices
std::pair<std::vector<double>, int> JacobiSolver::solve(const Eigen::SparseMatrix<double>& A, const std::vector<double>& b) {
    int n = A.rows();
    std::vector<double> x(n, 0.0);
    std::vector<double> x_new(n, 0.0);
    int iter;

    // Convert SparseMatrix to RowMajor for efficient row access
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_row = A;

    for (iter = 0; iter < max_iterations; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            double diag = 0.0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A_row, i); it; ++it) {
                if (it.col() == i) {
                    diag = it.value();
                } else {
                    sigma += it.value() * x[it.col()];
                }
            }
            if (diag == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << std::endl;
                return {x_new, iter};
            }
            x_new[i] = (b[i] - sigma) / diag;
        }

        // Compute the infinity norm of (x_new - x)
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff = std::max(diff, std::abs(x_new[i] - x[i]));
        }

        // Check for convergence
        if (diff < tolerance) {
            return {x_new, iter + 1};
        }

        x = x_new;
    }

    return {x_new, iter};
}

