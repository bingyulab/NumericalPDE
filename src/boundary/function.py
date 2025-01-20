import numpy as np
import torch


def u_ex(x, y):
    """Exact solution u(x,y) = sin²(πx) sin²(πy)"""
    if torch.is_tensor(x):
        return torch.sin(np.pi * x)**2 * torch.sin(np.pi * y)**2
    else:
        return np.sin(np.pi * x)**2 * np.sin(np.pi * y)**2


def f(x, y):
    """Right-hand side f = -Δu = -(uxx + uyy)
    For u = sin²(πx)sin²(πy), we have:
    Δu = 2π² [cos(2πx) sin²(πy) + cos(2πy) sin²(πx)]
    Hence, f = -Δu
    """
    if torch.is_tensor(x):
        return -2 * np.pi**2 * (
            torch.cos(2 * np.pi * x) * torch.sin(np.pi * y)**2
            + torch.cos(2 * np.pi * y) * torch.sin(np.pi * x)**2
        )
    else:
        return -2 * np.pi**2 * (
            np.cos(2 * np.pi * x) * np.sin(np.pi * y)**2
            + np.cos(2 * np.pi * y) * np.sin(np.pi * x)**2
        )
