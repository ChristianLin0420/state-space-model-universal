from typing import Optional, Tuple
import math
import torch
import numpy as np
from scipy.special import legendre

def hippo_initializer(size: int, method: str = "legendre") -> torch.Tensor:
    """Initialize HiPPO matrix using different methods.
    
    Args:
        size: Size of the square matrix
        method: Initialization method ("legendre" or "fourier")
        
    Returns:
        HiPPO matrix as torch tensor
    """
    if method == "legendre":
        return _hippo_legendre(size)
    elif method == "fourier":
        return _hippo_fourier(size)
    else:
        raise ValueError(f"Unknown HiPPO initialization method: {method}")

def _hippo_legendre(size: int) -> torch.Tensor:
    """Generate HiPPO-LegS matrix.
    
    Args:
        size: Size of the square matrix
        
    Returns:
        HiPPO-LegS matrix
    """
    # Get coefficients of Legendre polynomials
    Q = np.zeros((size, size))
    for k in range(size):
        poly = legendre(k)
        Q[k] = poly.c[::-1]
        Q[k] = Q[k] / np.sqrt(2 * k + 1)
    
    # Construct HiPPO-LegS matrix
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i < j:
                A[i, j] = (2 * j + 1) ** 0.5
            else:
                A[i, j] = -(2 * i + 1) ** 0.5
                
    A = Q @ A @ Q.T
    return torch.from_numpy(A).float()

def _hippo_fourier(size: int) -> torch.Tensor:
    """Generate HiPPO-FouS matrix.
    
    Args:
        size: Size of the square matrix
        
    Returns:
        HiPPO-FouS matrix
    """
    # Generate frequencies
    freqs = torch.arange(size//2).float()
    freqs = torch.cat([freqs, -freqs[:(size-size//2)]])
    
    # Construct block diagonal matrix
    A = torch.zeros(size, size)
    for i in range(size):
        A[i, i] = 2 * math.pi * freqs[i]
        
    return A 