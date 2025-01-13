import pickle
import torch
from torch import nn
from src.solver.base_solver import Solver


class SolverPINN(Solver):
    """
    Physics-Informed Neural Network (PINN) Solver for the Poisson Equation.
    """

    def __init__(self, device='cpu', precision=torch.float32, verbose=False,
                 use_weights=True, compile_model=True, lambdas_pde=None, seed=0):
        # Removed arguments from super().__init__ since the base Solver class does not accept any
        super().__init__()  # Call to base class constructor without arguments

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def build_model(self, input_dim, output_dim, hidden_layers, neurons):
        layers = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, output_dim))
        model = nn.Sequential(*layers)
        return model

    def evaluate_network_pde(self, model, x, y):
        x.requires_grad = True
        y.requires_grad = True
        u = model(torch.cat([x, y], dim=1))
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        pde_residual = u_xx + u_yy
        return pde_residual

    def evaluate_network_bc(self, model, x_bc, y_bc, u_bc):
        u_pred = model(torch.cat([x_bc, y_bc], dim=1))
        bc_residual = u_pred - u_bc
        return bc_residual

    def load_precomputed_data(self, precomputed_path):
        with open(precomputed_path, 'rb') as f:
            precomputed_data = pickle.load(f)
        return precomputed_data

    def precompute_LHS_RHS(self, model, x, y, x_bc, y_bc, u_bc, lambdas_pde):
        pde_residual = self.evaluate_network_pde(model, x, y)
        bc_residual = self.evaluate_network_bc(model, x_bc, y_bc, u_bc)
        LHS = lambdas_pde[0] * pde_residual + lambdas_pde[1] * bc_residual
        RHS = torch.zeros_like(LHS)
        return LHS, RHS

    def save_precomputed_data(self, precomputed_path, precomputed_data):
        with open(precomputed_path, 'wb') as f:
            pickle.dump(precomputed_data, f)

    def precompute(self, model, x, y, x_bc, y_bc, u_bc, lambdas_pde, precomputed_path):
        LHS, RHS = self.precompute_LHS_RHS(model, x, y, x_bc, y_bc, u_bc, lambdas_pde)
        precomputed_data = {'LHS': LHS, 'RHS': RHS}
        self.save_precomputed_data(precomputed_path, precomputed_data)

    def solve(self, L, b):
        # Load data
        data_path = 'data.pkl'
        data = self.load_data(data_path)
        x, y, x_bc, y_bc, u_bc = data['x'], data['y'], data['x_bc'], data['y_bc'], data['u_bc']

        # Build model
        input_dim = 2
        output_dim = 1
        hidden_layers = 4
        neurons = 20
        model = self.build_model(input_dim, output_dim, hidden_layers, neurons)

        # Precompute LHS and RHS
        lambdas_pde = [1.0, 1.0]
        precomputed_path = 'precomputed_data.pkl'
        self.precompute(model, x, y, x_bc, y_bc, u_bc, lambdas_pde, precomputed_path)

        # Load precomputed data
        precomputed_data = self.load_precomputed_data(precomputed_path)
        LHS, RHS = precomputed_data['LHS'], precomputed_data['RHS']

        # Solve the system
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = torch.mean((LHS - RHS) ** 2)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Return the solution
        u_pred = model(torch.cat([x, y], dim=1))
        return u_pred.detach().numpy()