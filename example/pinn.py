import sys
import os

# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from src.boundary.function import f, u_ex
from torch.utils.data import Dataset, DataLoader


# Generate random points within the domain
def generate_random_points(num_points):
    x_rand = np.random.uniform(0, 1, (num_points, 1))
    y_rand = np.random.uniform(0, 1, (num_points, 1))
    return x_rand, y_rand


# Define a custom Dataset for generating random points
class PoissonDataset(Dataset):
    def __init__(self, initial_points=2000, additional_points=200):
        self.initial_points = initial_points
        self.additional_points = additional_points
        self.x_data, self.y_data = self.generate_initial_points()

    def generate_initial_points(self):
        x_rand, y_rand = generate_random_points(self.initial_points)
        return torch.tensor(x_rand, dtype=torch.float32), torch.tensor(y_rand, dtype=torch.float32)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def add_additional_points(self):
        x_rand, y_rand = generate_random_points(self.additional_points)
        new_x = torch.tensor(x_rand, dtype=torch.float32)
        new_y = torch.tensor(y_rand, dtype=torch.float32)
        self.x_data = torch.cat([self.x_data, new_x], dim=0)
        self.y_data = torch.cat([self.y_data, new_y], dim=0)


# Define the Physics-Informed Neural Network (PINN) model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Define the dense layers
        self.dense1 = nn.Linear(2, 40)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(40, 40)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(40, 1)
        self.tanh = nn.Tanh()

    # Define the forward pass
    def forward(self, inputs):
        x = self.tanh(self.dense1(inputs))
        x = self.dropout1(x)
        x = self.tanh(self.dense2(x))
        x = self.dropout2(x)
        return self.dense3(x)

    # Calculate the Laplacian of the predicted solution
    def laplacian(self, u, x, y):
        grads = grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
        u_x = grads[0]
        u_y = grads[1]
        grads2 = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)
        u_xx = grads2[0]
        grads2 = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)
        u_yy = grads2[0]
        laplacian_u = u_xx + u_yy
        return laplacian_u


# Define the loss function
def loss_function(model, x, y):
    # Concatenate x and y along the feature dimension
    inputs = torch.cat([x, y], dim=1)
    predicted_solution = model(inputs)
    poisson_residual = model.laplacian(predicted_solution, x, y) - f(x, y)
    boundary_residual = predicted_solution - u_ex(x, y)
    mse_pde_residual = torch.mean(poisson_residual**2)
    mse_boundary_residual = torch.mean(boundary_residual**2)
    loss_val = mse_pde_residual + mse_boundary_residual
    return loss_val


# Set random seed for reproducibility
np.random.seed(42)
# Set the number of training points
num_points = 2000

# Initialize the dataset and dataloader
dataset = PoissonDataset(initial_points=num_points)
batch_size = 32  # Define an appropriate batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cpu')

model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


loss_history = []
max_accuracy = 0
epochs = 2001
max_norms = []
l2_norms = []


for epoch in range(epochs):
    # Add additional random points every 100 epochs
    if epoch % 100 == 0 and epoch != 0:
        dataset.add_additional_points()

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Enable gradient tracking for the current batch
        batch_x.requires_grad = True
        batch_y.requires_grad = True
        
        optimizer.zero_grad()
        loss_value = loss_function(model, batch_x, batch_y)
        loss_value.backward()

        # Calculate and store the maximum norm of the gradients
        max_grad_norm = max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)
        max_norms.append(max_grad_norm.item())

        # Calculate and store the L2 norm of the gradients
        l2_grad_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in model.parameters() if p.grad is not None))
        l2_norms.append(l2_grad_norm.item())

        optimizer.step()

        loss_history.append(loss_value.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item()}, Max Norm: {max_grad_norm.item()}, L2 Norm: {l2_grad_norm.item()}")

# Generate test points for visualization
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
x_test, y_test = np.meshgrid(x_test, y_test)
x_test = x_test.flatten().reshape(-1, 1)
y_test = y_test.flatten().reshape(-1, 1)
inputs_test = torch.tensor(np.concatenate([x_test, y_test], axis=1), dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predicted_solution = model(inputs_test).cpu().numpy().reshape(100, 100)


plt.contourf(x_test.reshape(100, 100), y_test.reshape(100, 100), predicted_solution, cmap='viridis')
plt.colorbar(label='Predicted')
plt.title('PINN Solution 2D chart to Poisson\'s Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_test.reshape(100, 100), y_test.reshape(100, 100), predicted_solution, cmap='viridis')
ax1.set_title('PINN Solution 3D chart to Poisson\'s Equation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Predicted Solution')




plt.figure(figsize=(8, 5))
plt.plot(max_norms, label='Max Gradient Norm')
plt.plot(l2_norms, label='L2 Gradient Norm')
plt.title('Maximum and L2 Gradient Norm During Training')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Loss changing')
plt.title('Change Loss Function During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

