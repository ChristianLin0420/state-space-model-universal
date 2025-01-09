from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel

class S4DLayer(StateSpaceModel):
    """Diagonal State Space Model (S4D) implementation.
    
    This implementation follows Section 4 of the paper, which discusses three key initializations:
    1. S4D-Lin: Linear initialization with inverse scaling
    2. S4D-Inv: Inverse initialization
    3. S4D-LegS: Approximation to S4-LegS
    
    Paper: "On the Parameterization and Initialization of Diagonal State Space Models"
    
    Args:
        d_model (int): Model dimension (H in paper notation)
        d_state (int): State dimension (N in paper notation)
        dropout (float, optional): Dropout rate. Defaults to 0.0
        dt_min (float, optional): Minimum step size for discretization. Defaults to 0.001
        dt_max (float, optional): Maximum step size for discretization. Defaults to 0.1
        init_method (str, optional): Initialization method for diagonal state matrix.
            One of ["inv", "lin", "legs"]:
            - "inv": S4D-Inv with inverse scaling (Equation 8)
            - "lin": S4D-Lin with linear scaling (Equation 9)
            - "legs": S4D-LegS approximation
            Defaults to "inv"
        real_transform (bool, optional): Whether to apply real transformation
            to state matrix (-exp(x) + iy). Defaults to True
    
    Attributes:
        N (int): State dimension (from paper notation)
        H (int): Model dimension (from paper notation)
        A (Parameter): Diagonal state matrix (H, N)
        B (Parameter): Input projection (H,)
        C (Parameter): Output projection (H,)
        D (Parameter): Direct feedthrough term (H,)
        log_dt (Parameter): Log step sizes (H,)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init_method: Literal["inv", "lin", "legs"] = "inv",
        real_transform: bool = True,
    ):
        """Initialize S4D layer."""
        super().__init__(d_model, d_state, dropout)
        
        # Save arguments
        self.N = d_state  # Following paper notation
        self.H = d_model
        self.init_method = init_method
        self.real_transform = real_transform
        
        # Initialize dt
        log_dt = torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Initialize diagonal state matrix A according to chosen method
        A = self._init_A()
        self.A = nn.Parameter(A)
        
        # Initialize B, C with proper scaling
        self.B = nn.Parameter(torch.randn(self.H, dtype=torch.complex64) / math.sqrt(self.N))
        self.C = nn.Parameter(torch.randn(self.H, dtype=torch.complex64) / math.sqrt(self.N))
        
        # Optional direct term
        self.D = nn.Parameter(torch.zeros(self.H))
        
        self.dropout = nn.Dropout(dropout)
        
    def _init_A(self) -> torch.Tensor:
        """Initialize diagonal state matrix A according to chosen method.
        
        Following Section 4 of the paper:
        - S4D-Inv: A_n = -1/2 + iN(N/(2n+1) - 1)
        - S4D-Lin: A_n = -1/2 + iπn
        - S4D-LegS: Approximation to S4-LegS with c ≈ 0.5226
        
        The initialization is critical for the model's performance:
        1. S4D-Inv provides inverse scaling for better long-range dependencies
        2. S4D-Lin offers simpler linear scaling
        3. S4D-LegS approximates the full S4-LegS behavior
        
        Returns:
            torch.Tensor: Complex diagonal state matrix of shape (H, N)
        """
        N = self.N
        n = torch.arange(N, dtype=torch.float32)
        
        if self.init_method == "inv":
            # S4D-Inv initialization (Equation 8)
            A_real = -0.5 * torch.ones(N)
            A_imag = N * (N/(2*n + 1) - 1)
            
        elif self.init_method == "lin":
            # S4D-Lin initialization (Equation 9)
            A_real = -0.5 * torch.ones(N)
            A_imag = math.pi * n
            
        else:  # legs
            # S4D-LegS initialization
            # Note: c ≈ 0.5226 is a constant found empirically
            c = 0.5226
            A_real = -0.5 * torch.ones(N)
            A_imag = torch.sqrt(torch.arange(N, dtype=torch.float32) + 0.5)
            A_imag = c * N * A_imag
            
        # Combine real and imaginary parts
        A = torch.complex(A_real, A_imag)
        
        if self.real_transform:
            # Apply real transform: -exp(x) + iy
            A = -torch.exp(A_real) + 1j * A_imag
            
        # Repeat for each feature dimension
        A = A.unsqueeze(0).repeat(self.H, 1)
        
        return A
        
    def forward(
        self,
        u: torch.Tensor,  # (B, L, H)
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using diagonal state matrices.
        
        Implements the SSM computation using the diagonal structure:
        H(s) = C(sI - A)^{-1}B
        
        The computation is performed efficiently in the frequency domain:
        1. Convert input to frequency domain using FFT
        2. Apply transfer function H(s)
        3. Convert back to time domain using IFFT
        
        Args:
            u: Input tensor of shape (batch_size, seq_len, d_model)
            state: Not used in diagonal variant (kept for interface consistency)
            
        Returns:
            tuple:
                - output: Tensor of shape (batch_size, seq_len, d_model)
                - None: No state is returned in diagonal variant
        """
        B, L, H = u.shape
        
        # Get discretized step size
        dt = torch.exp(self.log_dt)  # (H,)
        
        # Discretize A
        # z = exp(dt * A)
        z = torch.exp(dt.unsqueeze(-1) * self.A)  # (H, N)
        
        # Convert to frequency domain
        k = torch.arange(L//2 + 1, device=u.device)
        omega = torch.exp(-2j * math.pi * k / L)  # (L//2 + 1,)
        
        # Prepare parameters for broadcast
        z = z.unsqueeze(-1)        # (H, N, 1)
        B = self.B.unsqueeze(-1)   # (H, 1)
        C = self.C.unsqueeze(-1)   # (H, 1)
        omega = omega.unsqueeze(0)  # (1, L//2 + 1)
        
        # Compute transfer function
        h = (C * B) / (omega - z)  # (H, L//2 + 1)
        
        # Apply SSM in frequency domain
        U_f = torch.fft.rfft(u, dim=-2)
        Y_f = U_f * rearrange(h, 'h f -> f h')
        y = torch.fft.irfft(Y_f, n=L, dim=-2)
        
        # Add direct term
        y = y + u * self.D
        
        # Apply dropout
        y = self.dropout(y)
        
        return y, None
        
    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step state update for autoregressive generation.
        
        Implements the state space recurrence for a single time step:
            x' = Ax + Bu
            y = Cx + Du
        
        This is useful for autoregressive generation where we need
        to process one token at a time.
        
        Args:
            u: Input tensor of shape (batch_size, d_model)
            state: Previous state tensor of shape (batch_size, d_state)
            
        Returns:
            tuple:
                - output: Tensor of shape (batch_size, d_model)
                - next_state: Updated state tensor of shape (batch_size, d_state)
        """
        dt = torch.exp(self.log_dt)
        z = torch.exp(dt.unsqueeze(-1) * self.A)
        next_state = state * z + u.unsqueeze(-1) * self.B.unsqueeze(-1)
        y = torch.sum(next_state * self.C.unsqueeze(-1).conj(), dim=-1)
        y = y + u * self.D
        return y, next_state 