#ifndef JACOBI_SOLVER_H
#define JACOBI_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector> 
#include "base_solver.h" 

class JacobiSolver : public Solver {
public:
    JacobiSolver(double tol = 1e-8, int max_iter = 1000);

    // Override the solve method to return a pair of solution and iterations
    std::pair<std::vector<double>, int> solve(const Eigen::SparseMatrix<double>& A, const std::vector<double>& b) override;

    // New solve method for dense matrices
    std::pair<std::vector<double>, int> solve(const Eigen::MatrixXd& A, const std::vector<double>& b);

private:
    double tolerance;
    int max_iterations;
};

#endif // JACOBI_SOLVER_H