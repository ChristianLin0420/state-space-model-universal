from typing import Optional, Literal, Tuple
import math
import torch
import numpy as np
from scipy.special import legendre

class HiPPOInit:
    """HiPPO (High-Order Polynomial Projection Operators) initialization.
    
    Implements measure families and corresponding HiPPO operators:
    - LegT: Translated Legendre (uniform on sliding window)
    - LagT: Translated Laguerre (exponential decay)
    - LegS: Scaled Legendre (uniform on [0,t] with timescale robustness)
    
    Paper: HiPPO: Recurrent Memory with Optimal Polynomial Projections
    """
    
    @staticmethod
    def make_LegT_matrix(N: int, theta: float = 1.0) -> torch.Tensor:
        """Generate HiPPO-LegT matrix using translated Legendre measure.
        
        LegT assigns uniform weight to the most recent history [t-θ, t].
        
        Args:
            N: Size of the square matrix
            theta: Length of sliding window (default: 1.0)
            
        Returns:
            torch.Tensor: HiPPO-LegT matrix of shape (N, N)
            
        Reference:
            Theorem 1, Equation (1) from the paper
        """
        # Initialize A matrix
        A = torch.zeros((N, N))
        
        # Compute entries according to Theorem 1
        # A_{nk} = 1/θ * {
        #   (-1)^{n-k}(2n+1)  if n ≥ k
        #   2n+1               if n ≤ k
        # }
        # Create indices matrices
        n_idx = torch.arange(N).unsqueeze(1)  # Shape: (N, 1)
        k_idx = torch.arange(N).unsqueeze(0)  # Shape: (1, N)
        
        # Create mask for n >= k condition
        mask = (n_idx >= k_idx)
        
        # Compute (2n+1) term
        multiplier = (2 * n_idx + 1)
        
        # Compute (-1)^(n-k) term for n >= k entries
        power_term = (-1.0) ** (n_idx - k_idx)
        
        # Combine terms using mask
        A = torch.where(mask, power_term * multiplier, multiplier)
        
        # Scale by theta
        A = A / theta
        
        return A
        
    @staticmethod
    def make_LagT_matrix(N: int) -> torch.Tensor:
        """Generate HiPPO-LagT matrix using translated Laguerre measure.
        
        LagT uses exponentially-decaying measure, assigning more importance
        to recent history.
        
        Args:
            N: Size of the square matrix
            
        Returns:
            torch.Tensor: HiPPO-LagT matrix of shape (N, N)
            
        Reference:
            Theorem 1, Equation (2) from the paper
        """
        # Initialize A matrix
        A = torch.zeros((N, N))
        
        # Compute entries according to Theorem 1
        # A_{nk} = {
        #   1  if n ≥ k
        #   0  if n < k
        # }
        # Create indices matrices
        n_idx = torch.arange(N).unsqueeze(1)  # Shape: (N, 1)
        k_idx = torch.arange(N).unsqueeze(0)  # Shape: (1, N)
        
        # Create mask for n >= k condition and convert to float
        A = (n_idx >= k_idx).float()
                    
        return A
        
    @staticmethod
    def make_LegT_B_vector(N: int, theta: float = 1.0) -> torch.Tensor:
        """Generate B vector for HiPPO-LegT.
        
        Args:
            N: Size of vector
            theta: Length of sliding window
            
        Returns:
            torch.Tensor: B vector of shape (N,)
            
        Reference:
            Theorem 1, Equation (1) from the paper
        """
        # B_n = 1/θ * (2n+1)(-1)^n
        n = torch.arange(N)
        B = (2*n + 1) * ((-1.0) ** n) / theta
        return B
        
    @staticmethod
    def make_LagT_B_vector(N: int) -> torch.Tensor:
        """Generate B vector for HiPPO-LagT.
        
        Args:
            N: Size of vector
            
        Returns:
            torch.Tensor: B vector of shape (N,)
            
        Reference:
            Theorem 1, Equation (2) from the paper
        """
        # B_n = 1 for all n
        return torch.ones(N)
        
    @staticmethod
    def make_LegS_matrix(N: int) -> torch.Tensor:
        """Generate HiPPO-LegS matrix using scaled Legendre measure.
        
        LegS assigns uniform weight to all history [0,t] with timescale robustness.
        
        Args:
            N: Size of the square matrix
            
        Returns:
            torch.Tensor: HiPPO-LegS matrix of shape (N, N)
            
        Reference:
            Theorem 2, Equation (3) from the paper
        """
        # Create indices matrices
        n_idx = torch.arange(N).unsqueeze(1)  # Shape: (N, 1)
        k_idx = torch.arange(N).unsqueeze(0)  # Shape: (1, N)
        
        # Compute matrix entries according to Theorem 2:
        # A_{nk} = {
        #   (2n+1)^{1/2}(2k+1)^{1/2}  if n > k
        #   n+1                        if n = k
        #   0                          if n < k
        # }
        
        # Create masks for different conditions
        greater_mask = (n_idx > k_idx)
        equal_mask = (n_idx == k_idx)
        
        # Compute terms
        greater_term = torch.sqrt((2*n_idx + 1) * (2*k_idx + 1))
        equal_term = n_idx + 1
        
        # Combine terms using masks
        A = torch.zeros((N, N))
        A = torch.where(greater_mask, greater_term, A)
        A = torch.where(equal_mask, equal_term, A)
        
        return A
        
    @staticmethod
    def make_LegS_B_vector(N: int) -> torch.Tensor:
        """Generate B vector for HiPPO-LegS.
        
        Args:
            N: Size of vector
            
        Returns:
            torch.Tensor: B vector of shape (N,)
            
        Reference:
            Theorem 2 from the paper
        """
        # B_n = (2n+1)^{1/2}
        n = torch.arange(N)
        B = torch.sqrt(2*n + 1)
        return B
        
    @staticmethod
    def discretize_LegS(A: torch.Tensor, B: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize HiPPO-LegS continuous dynamics using Euler method.
        
        Implements the discrete-time dynamics from Theorem 2, Equation (4):
            c_{k+1} = (1 - A/k)c_k + (1/k)Bf_k
        
        Args:
            A: Continuous-time state matrix
            B: Continuous-time input vector
            step: Current time step k
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Discretized (A_d, B_d) matrices
        """
        # Implement discrete dynamics from Equation (4)
        k = step + 1  # Add 1 to avoid division by zero at step 0
        A_d = torch.eye(A.shape[0]) - (A / k)
        B_d = B / k
        return A_d, B_d

def hippo_initializer(
    size: int,
    method: Literal["legT", "lagT", "legS"] = "legS",
    theta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize HiPPO matrices according to specified method.
    
    Args:
        size: Size of the square matrix
        method: Initialization method ("legT", "lagT", or "legS")
        theta: Length of sliding window for LegT (ignored for LagT and LegS)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (A, B) matrices where
            - A: State transition matrix of shape (size, size)
            - B: Input projection vector of shape (size,)
            
    Raises:
        ValueError: If method is not recognized
    """
    if method.lower() == "legt":
        A = HiPPOInit.make_LegT_matrix(size, theta)
        B = HiPPOInit.make_LegT_B_vector(size, theta)
    elif method.lower() == "lagt":
        A = HiPPOInit.make_LagT_matrix(size)
        B = HiPPOInit.make_LagT_B_vector(size)
    elif method.lower() == "legs":
        A = HiPPOInit.make_LegS_matrix(size)
        B = HiPPOInit.make_LegS_B_vector(size)
    else:
        raise ValueError(f"Unknown HiPPO method: {method}. Choose from ['legT', 'lagT', 'legS']")
        
    return A, B 