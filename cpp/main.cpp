// based on https://github.com/tgolubev/Poisson_eqn_solvers.git
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include "jacobi_solver.h" // Include the JacobiSolver header

typedef Eigen::Triplet<double> Trp; // Triplet class for sparse matrix construction

int main()
{
    // Start clock timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Define problem size
    const int num_cell = 10; // Example value, adjust as needed
    const int N = num_cell; // Number of cells in one dimension
    const int num_elements = N * N;

    // Initialize matrices and vectors
    Eigen::VectorXd x_eigen(num_elements);
    Eigen::VectorXd b_eigen(num_elements);
    Eigen::SparseMatrix<double> A(num_elements, num_elements);
    Eigen::MatrixXd epsilon = Eigen::MatrixXd::Ones(num_cell + 1, num_cell + 1); // Example initialization
    Eigen::MatrixXd netcharge = Eigen::MatrixXd::Zero(num_cell + 1, num_cell + 1); // Example initialization

    // Initialize diagonals and RHS
    std::vector<double> main_diag(num_elements + 1, 0.0);
    std::vector<double> upper_diag(num_elements, 0.0);
    std::vector<double> lower_diag(num_elements, 0.0);
    std::vector<double> far_lower_diag(num_elements - N + 1, 0.0);
    std::vector<double> far_upper_diag(num_elements - N + 1, 0.0);
    std::vector<double> V(num_elements + 1, 0.0); // Electric potential
    std::vector<double> rhs(num_elements + 1, 0.0);

    // Define boundary conditions and initial conditions
    double Va = 1.0;
    double V_bottomBC = 0.0;
    double Vt = 1.0; // Define Vt as needed
    double V_topBC = Va / Vt;

    // Initial conditions
    double diff = (V_topBC - V_bottomBC) / num_cell;
    int index = 0;
    for (int j = 1; j <= N; j++) { // Corresponds to y-coordinate
        index++;
        V[index] = diff * j;
        for (int i = 2; i <= N; i++) { // Elements along the x-direction
            index++;
            V[index] = V[index - 1];
        }
    }

    // Side boundary conditions (insulating BCs)
    std::vector<double> V_leftBC(N + 1, 0.0);
    std::vector<double> V_rightBC(N + 1, 0.0);

    for (int i = 1; i <= N; i++) {
        V_leftBC[i] = V[(i - 1) * N + 1];
        V_rightBC[i] = V[i * N];
    }

    // Define functions to set up diagonals (implement these functions as needed)
    // ...existing code...
    // Example placeholders for diagonal setup functions
    auto set_main_Vdiag = [&](const Eigen::MatrixXd& epsilon, std::vector<double>& diag) -> std::vector<double> {
        // Implement your logic to set the main diagonal
        // Example:
        for (int i = 1; i <= num_elements; i++) {
            diag[i] = 4.0; // Example value
        }
        return diag;
    };
    
    auto set_upper_Vdiag = [&](const Eigen::MatrixXd& epsilon, std::vector<double>& diag) -> std::vector<double> {
        // Implement your logic to set the upper diagonal
        // Example:
        for (int i = 1; i < num_elements; i++) {
            diag[i] = -1.0; // Example value
        }
        return diag;
    };
    
    auto set_lower_Vdiag = [&](const Eigen::MatrixXd& epsilon, std::vector<double>& diag) -> std::vector<double> {
        // Implement your logic to set the lower diagonal
        // Example:
        for (int i = 1; i < num_elements; i++) {
            diag[i] = -1.0; // Example value
        }
        return diag;
    };
    
    auto set_far_upper_Vdiag = [&](const Eigen::MatrixXd& epsilon, std::vector<double>& diag) -> std::vector<double> {
        // Implement your logic to set the far upper diagonal
        // Example:
        for (int i = 1; i < num_elements - N + 1; i++) {
            diag[i] = -1.0; // Example value
        }
        return diag;
    };
    
    auto set_far_lower_Vdiag = [&](const Eigen::MatrixXd& epsilon, std::vector<double>& diag) -> std::vector<double> {
        // Implement your logic to set the far lower diagonal
        // Example:
        for (int i = 1; i < num_elements - N + 1; i++) {
            diag[i] = -1.0; // Example value
        }
        return diag;
    };

    // Setup diagonals
    main_diag = set_main_Vdiag(epsilon, main_diag);
    upper_diag = set_upper_Vdiag(epsilon, upper_diag);
    lower_diag = set_lower_Vdiag(epsilon, lower_diag);
    far_upper_diag = set_far_upper_Vdiag(epsilon, far_upper_diag);
    far_lower_diag = set_far_lower_Vdiag(epsilon, far_lower_diag);

    // Setup RHS of Poisson equation
    int index2 = 0;
    for (int j = 1; j <= N; j++) {
        if (j == 1) {
            for (int i = 1; i <= N; i++) {
                index2++;
                if (i == 1) {
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * (V_leftBC[1] + V_bottomBC);
                }
                else if (i == N)
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * (V_rightBC[1] + V_bottomBC);
                else
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * V_bottomBC;
            }
        }
        else if (j == N) {
            for (int i = 1; i <= N; i++) {
                index2++;
                if (i == 1)
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * (V_rightBC[N] + V_topBC);
                else if (i == N)
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * (V_rightBC[N] + V_topBC);
                else
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * V_topBC;
            }
        }
        else {
            for (int i = 1; i <= N; i++) {
                index2++;
                if (i == 1)
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * V_leftBC[j];
                else if (i == N)
                    rhs[index2] = netcharge(i, j) + epsilon(i, j) * V_rightBC[j];
                else
                    rhs[index2] = netcharge(i, j);
            }
        }
    }

    // Setup the triplet list for sparse matrix
    std::vector<Trp> triplet_list;
    triplet_list.reserve(5 * num_elements); // Approximate size

    for (int i = 1; i <= num_elements; i++) {
        b_eigen(i - 1) = rhs[i];
        triplet_list.emplace_back(Trp(i - 1, i - 1, main_diag[i]));
    }
    for (int i = 1; i < upper_diag.size(); i++) {
        triplet_list.emplace_back(Trp(i - 1, i, upper_diag[i]));
    }
    for (int i = 1; i < lower_diag.size(); i++) {
        triplet_list.emplace_back(Trp(i, i - 1, lower_diag[i]));
    }
    for (int i = 1; i < far_upper_diag.size(); i++) {
        triplet_list.emplace_back(Trp(i - 1, i - 1 + N, far_upper_diag[i]));
        triplet_list.emplace_back(Trp(i - 1 + N, i - 1, far_lower_diag[i]));
    }

    A.setFromTriplets(triplet_list.begin(), triplet_list.end()); // Construct sparse matrix A

    // Using Sparse Cholesky
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> SCholesky;
    SCholesky.analyzePattern(A);
    SCholesky.factorize(A);
    x_eigen = SCholesky.solve(b_eigen);

    // Using Jacobi Solver
    JacobiSolver jacobi_solver(1e-8, 1000);
    std::vector<double> b_vector(b_eigen.data(), b_eigen.data() + b_eigen.size());
    // Capture both solution and iterations from solve(...)
    std::pair<std::vector<double>, int> jacobi_result = jacobi_solver.solve(A, b_vector);
    std::vector<double> x_jacobi = jacobi_result.first;
    int jacobi_iterations = jacobi_result.second;

    // Convert std::vector<double> to Eigen::VectorXd for comparison
    Eigen::VectorXd x_jacobi_eigen = Eigen::VectorXd::Map(x_jacobi.data(), x_jacobi.size());

    // Compute error compared to the Cholesky solver solution
    Eigen::VectorXd error = x_jacobi_eigen - x_eigen;
    double error_norm = error.norm();

    std::cout << "Jacobi Solver:" << std::endl;
    std::cout << "#iterations: " << jacobi_iterations << std::endl;
    std::cout << "Solution Error (Cholesky - Jacobi): " << error_norm << std::endl;

    // End clock timer
    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    std::cout << "1 Va CPU time = " << time.count() << " seconds" << std::endl;

    return 0;
}