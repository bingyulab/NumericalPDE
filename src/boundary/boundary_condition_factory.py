from src.boundary.boundary_condition import DirichletBoundaryCondition, NeumannBoundaryCondition

class BoundaryConditionFactory:
    @staticmethod
    def create_boundary_condition(condition_type, g=None):
        if condition_type.lower() == "dirichlet":
            return DirichletBoundaryCondition(g=g)  # Pass g to the boundary condition
        elif condition_type.lower() == "neumann":
            return NeumannBoundaryCondition()
        else:
            raise ValueError(f"Unknown boundary condition type: {condition_type}")
