#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include <Eigen/Sparse>
#include <vector>

// Abstract base class for solvers
class Solver {
public:
    virtual ~Solver() = default;

    // Ensure the solve method returns a std::pair
    virtual std::pair<std::vector<double>, int> solve(const Eigen::SparseMatrix<double>& A, const std::vector<double>& b) = 0;
};
#endif // BASE_SOLVER_H