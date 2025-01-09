from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel

class MambaLayer(StateSpaceModel):
    """Selective State Space layer using Mamba's scan mechanism.
    
    This implements the core selective scan operation from the Mamba paper,
    which can be used as a building block in various architectures.
    
    Paper: https://arxiv.org/abs/2312.00752
    
    Args:
        d_model (int): Input/output dimension
        d_state (int): State dimension
        expand_factor (int): Expansion factor for inner dimension (default: 2)
        dropout (float): Dropout rate (default: 0.1)
        init_method (str): Initialization method for diagonal state matrix ["inv", "lin", "legs"] (default: "inv")
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand_factor: int = 2,
        dropout: float = 0.1,
        init_method: Literal["inv", "lin", "legs"] = "inv"
    ):
        super().__init__(d_model, d_state, dropout)
        self.d_inner = expand_factor * d_model
        self.init_method = init_method
        
        # Initialize diagonal state matrix A
        A = self._init_A()
        self.A = nn.Parameter(A)
        
        # Input-dependent parameter networks
        self.B_network = nn.Linear(d_model, self.d_inner * d_state)
        self.C_network = nn.Linear(d_model, self.d_inner * d_state)
        self.D_network = nn.Linear(d_model, self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _init_A(self) -> torch.Tensor:
        """Initialize diagonal state matrix A according to chosen method.
        
        Following Section 4 of the S4D paper:
        - S4D-Inv: A_n = -1/2 + iN(N/(2n+1) - 1)
        - S4D-Lin: A_n = -1/2 + iπn
        - S4D-LegS: Approximation with c ≈ 0.5226
        
        Returns:
            torch.Tensor: Complex diagonal state matrix of shape (d_inner, d_state, d_state)
        """
        N = self.d_state
        n = torch.arange(N, dtype=torch.float32)
        
        # Initialize real and imaginary parts based on method
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
            c = 0.5226
            A_real = -0.5 * torch.ones(N)
            A_imag = torch.sqrt(n + 0.5)
            A_imag = c * N * A_imag
            
        # Combine real and imaginary parts
        A = torch.complex(A_real, A_imag)
        
        # Apply real transform: -exp(x) + iy
        A = -torch.exp(A_real) + 1j * A_imag
        
        # Create diagonal matrix and repeat for each feature dimension
        A = torch.diag_embed(A)  # (N, N)
        A = A.unsqueeze(0).repeat(self.d_inner, 1, 1)  # (d_inner, N, N)
        
        return A
        
    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∆(x) for time steps.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            Delta tensor (B, L, D_inner)
        """
        # Project and ensure positivity
        delta = self.D_network(x)  # (B, L, D_inner)
        return F.softplus(delta)
        
    def _discretize(
        self,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize continuous-time matrices.
        
        Args:
            delta: Time step tensor (B, L, D_inner)
            A: State matrix (D_inner, N, N)
            B: Input matrix (B, L, D_inner, N)
            
        Returns:
            Tuple of discretized matrices (A_bar, B_bar)
        """
        # Discretize A: Ā = exp(∆ ⊗ A)
        A_bar = torch.matrix_exp(
            A.unsqueeze(0) * delta.unsqueeze(-1).unsqueeze(-1)
        )
        
        # Discretize B: B̄ = ∆ ⊗ B
        B_bar = delta.unsqueeze(-1) * B
        
        return A_bar, B_bar
        
    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """Perform selective scan operation.
        
        Args:
            x: Input tensor (B, L, D)
            delta: Time steps (B, L, D_inner)
            A_bar: Discretized state matrix (B, L, D_inner, N, N)
            B_bar: Discretized input matrix (B, L, D_inner, N)
            C: Output matrix (B, L, D_inner, N)
            
        Returns:
            Output tensor (B, L, D_inner)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        state = torch.zeros(
            batch_size, self.d_inner, self.d_state,
            device=x.device, dtype=x.dtype
        )
        
        outputs = []
        for t in range(seq_len):
            # Update state: h_t = Āh_{t-1} + B̄x_t
            state = torch.bmm(
                A_bar[:, t].view(-1, self.d_state, self.d_state),
                state.view(-1, self.d_state, 1)
            ).view(batch_size, self.d_inner, self.d_state)
            
            state = state + B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Compute output: y_t = Ch_t
            y = torch.sum(state * C[:, t], dim=-1)
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass implementing selective scan.
        
        Args:
            x: Input tensor (B, L, D)
            state: Previous state (optional)
            
        Returns:
            tuple: (output, next_state)
        """
        B, L, D = x.shape
        
        # Compute input-dependent parameters
        B_param = self.B_network(x).view(B, L, self.d_inner, -1)
        C_param = self.C_network(x).view(B, L, self.d_inner, -1)
        
        # Compute ∆
        delta = self._compute_delta(x)
        
        # Discretize SSM
        A_bar, B_bar = self._discretize(delta, self.A, B_param)
        
        # Selective scan
        y = self._selective_scan(x, delta, A_bar, B_bar, C_param)
        
        # Project back to model dimension
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y, None
        
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state tensor
        """
        return torch.zeros(batch_size, self.d_state, device=self.A.device) 