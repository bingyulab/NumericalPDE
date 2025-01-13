# Solver Performance Statistics

## Overview

This document provides a comprehensive statistical analysis of the various solvers implemented in the `poisson_solver` package. It highlights the performance metrics, accuracy, and best-use scenarios for each solver to assist users in selecting the most appropriate method for their specific needs.

## Solvers Overview

The package includes the following solvers:

- **Direct Solvers**:
  - `DirectSolver`
  - `FastPoissonSolver`
  - `LUdecompositionSolver`
  - `CholeskySolver`
  - `LowRankApproximationSolver`
  
- **Iterative Solvers**:
  - `JacobiSolver`
  - `ConjugateGradientSolver`
  
- **Advanced Solvers**:
  - `MultigridSolver`
  - `GraphSolver`
  - `PriorityQueueSolver`

## Performance Metrics

The solvers have been evaluated based on the following criteria:

- **Execution Time**: Time taken to solve the system.
- **Accuracy**: The difference between the numerical solution and the exact solution.
- **Scalability**: Performance as the problem size increases.
- **Memory Usage**: Amount of memory consumed during execution.

### Execution Time

![Execution Time Comparison](./plots/execution_time.png)

*Figure 1: Execution Time of Various Solvers*

### Accuracy

![Accuracy Comparison](./plots/accuracy.png)

*Figure 2: Accuracy of Various Solvers*

### Scalability

![Scalability Comparison](./plots/scalability.png)

*Figure 3: Scalability of Various Solvers*

### Memory Usage

![Memory Usage Comparison](./plots/memory_usage.png)

*Figure 4: Memory Usage of Various Solvers*

## Solver Analysis

### Direct Solvers

#### DirectSolver

- **Execution Time**: Fast for small to medium-sized systems.
- **Accuracy**: High accuracy as it computes the exact solution.
- **Scalability**: Not suitable for very large systems due to computational complexity.
- **Best Use Case**: Ideal for problems where exact solutions are required and the system size is manageable.

#### FastPoissonSolver

- **Execution Time**: Optimized for speed using FFT-based methods.
- **Accuracy**: Slightly less accurate than standard direct solvers due to numerical approximations.
- **Scalability**: Suitable for larger systems but limited by FFT performance.
- **Best Use Case**: Best for large, structured grid problems where speed is critical.

#### LUdecompositionSolver

- **Execution Time**: Moderate, efficient for multiple right-hand sides.
- **Accuracy**: High accuracy with precise LU decomposition.
- **Scalability**: Better than DirectSolver for certain matrix structures.
- **Best Use Case**: Useful when solving multiple systems with the same coefficient matrix.

#### CholeskySolver

- **Execution Time**: Efficient for symmetric positive-definite matrices.
- **Accuracy**: Highly accurate for applicable matrix types.
- **Scalability**: Similar to LUdecompositionSolver with better performance on specific matrices.
- **Best Use Case**: Optimal for symmetric positive-definite systems commonly found in Poisson problems.

#### LowRankApproximationSolver

- **Execution Time**: Fast for low-rank matrices but time increases with rank.
- **Accuracy**: Dependent on the chosen rank; higher ranks yield better accuracy.
- **Scalability**: Scales well for matrices with low-rank properties.
- **Best Use Case**: Effective for large systems where the matrix can be approximated with low rank, reducing computational load.

### Iterative Solvers

#### JacobiSolver

- **Execution Time**: Slower convergence compared to advanced iterative methods.
- **Accuracy**: High accuracy with sufficient iterations.
- **Scalability**: Scales well with system size but may require many iterations.
- **Best Use Case**: Suitable for educational purposes and simple problems where implementation simplicity is desired.

#### ConjugateGradientSolver

- **Execution Time**: Faster convergence than JacobiSolver, especially for symmetric positive-definite matrices.
- **Accuracy**: High accuracy with fewer iterations.
- **Scalability**: Efficient for large, sparse systems.
- **Best Use Case**: Ideal for large-scale problems with symmetric positive-definite matrices, common in numerical simulations.

### Advanced Solvers

#### MultigridSolver

- **Execution Time**: Extremely efficient with optimal scalability.
- **Accuracy**: High accuracy leveraging multiple grid resolutions.
- **Scalability**: Excellent scalability, handling very large systems effectively.
- **Best Use Case**: Best suited for very large, structured grid problems where computational resources are a constraint.

#### GraphSolver

- **Execution Time**: Variable, dependent on graph complexity.
- **Accuracy**: Accurate for systems that can be represented as graphs.
- **Scalability**: Limited by graph size and connectivity.
- **Best Use Case**: Applicable to problems that naturally map to graph structures, such as network simulations.

#### PriorityQueueSolver

- **Execution Time**: Efficient for systems where prioritizing certain variables improves convergence.
- **Accuracy**: High accuracy with strategic variable updates.
- **Scalability**: Scales well with problem size, especially when coupled with effective ordering strategies.
- **Best Use Case**: Suitable for sparse systems where selective updating can lead to faster convergence.

## Comparative Summary

| Solver                 | Execution Time | Accuracy | Scalability | Memory Usage | Best Use Case                                         |
|------------------------|-----------------|----------|-------------|--------------|-------------------------------------------------------|
| DirectSolver           | Fast            | High     | Medium      | Moderate     | Small to medium-sized exact solutions                 |
| FastPoissonSolver      | Very Fast       | Medium   | Large       | Moderate     | Large structured grids requiring speed                |
| LUdecompositionSolver  | Moderate        | High     | Medium      | High         | Multiple systems with the same coefficients           |
| CholeskySolver         | Fast            | High     | Medium      | Moderate     | Symmetric positive-definite systems                   |
| LowRankApproximation   | Fast            | Variable | Large       | Low          | Large low-rank approximable systems                   |
| JacobiSolver           | Slow            | High     | Large       | Low          | Educational purposes and simple systems               |
| ConjugateGradientSolver| Fast            | High     | Very Large  | Low          | Large symmetric positive-definite systems             |
| MultigridSolver        | Extremely Fast  | High     | Very Large  | Moderate     | Very large structured grid problems                    |
| GraphSolver            | Variable        | High     | Limited     | Moderate     | Graph-structured problems                              |
| PriorityQueueSolver    | Efficient       | High     | Large       | Low          | Sparse systems with effective variable prioritization |

## Recommendations

- **For Small to Medium Systems**: Use `DirectSolver` or `CholeskySolver` for exact solutions with high accuracy.
- **For Large Sparse Systems**: Opt for `ConjugateGradientSolver` or `MultigridSolver` to leverage scalability and efficiency.
- **When Speed is Crucial**: `FastPoissonSolver` provides rapid solutions for large, structured grids.
- **For Systems with Multiple Right-Hand Sides**: `LUdecompositionSolver` is ideal due to its efficiency in handling multiple systems.
- **When Memory is Limited**: `LowRankApproximationSolver` offers a balance between speed and memory usage for low-rank matrices.
- **Educational and Simple Implementations**: `JacobiSolver` serves well for learning and straightforward problems.
- **Graph-Structured Problems**: Utilize `GraphSolver` for systems that can be represented and solved as graphs.
- **Selective Variable Updating**: `PriorityQueueSolver` enhances convergence in sparse systems through prioritized updates.

## Conclusion

Selecting the appropriate solver depends on the specific requirements of your problem, including system size, sparsity, matrix properties, and desired accuracy. This statistical analysis serves as a guide to help you make informed decisions, ensuring optimal performance and efficiency in your numerical computations.

## Future Work

- **Expand Statistical Analysis**: Incorporate more metrics such as convergence rates and parallel performance.
- **Automate Reporting**: Develop scripts to generate updated statistical reports based on new test results.
- **User Feedback Integration**: Collect and include user experiences to refine solver recommendations.
