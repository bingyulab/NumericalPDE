cmake_minimum_required(VERSION 3.14)
project(PoissonSolvers)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find Pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add source files
set(SOURCES
    main.cpp
    jacobi_solver.cpp
    # ... other solver cpp files ...
)

# Add executable
add_executable(main_exec ${SOURCES})

# Link libraries if necessary
target_link_libraries(main_exec Eigen3::Eigen)

# Create Pybind11 module
pybind11_add_module(poisson_solvers bindings.cpp jacobi_solver.cpp)
target_link_libraries(poisson_solvers PRIVATE Eigen3::Eigen pybind11::module)