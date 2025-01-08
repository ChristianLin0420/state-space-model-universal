from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel
from .hippo import hippo_initializer

class S4Layer(StateSpaceModel):
    """Structured State Space Sequence (S4) model implementation.
    
    Paper: https://arxiv.org/abs/2111.00396
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        hippo_method: str = "legendre"
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        self.bidirectional = bidirectional
        
        # Initialize discretization parameters
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.dt = nn.Parameter(torch.exp(log_dt))
        
        # Initialize SSM parameters with HiPPO matrix
        A = hippo_initializer(d_state, method=hippo_method)
        self.A = nn.Parameter(A.unsqueeze(0).repeat(d_model, 1, 1))
        
        # Initialize B and C matrices
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        
        # Output projection
        self.D = nn.Parameter(torch.zeros(d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def discretize(self, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize continuous-time system using ZOH discretization."""
        # Compute matrix exponential
        dA = torch.matrix_exp(self.A * dt.unsqueeze(-1).unsqueeze(-1))
        dB = torch.einsum('bst,bs->bt', 
                         torch.matrix_exp(self.A * dt.unsqueeze(-1).unsqueeze(-1)), 
                         self.B)
        return dA, dB
        
    def forward(
        self, 
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of S4.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = self.init_state(batch_size)
            
        # Discretize SSM
        dA, dB = self.discretize(self.dt)
        
        # Compute state evolution
        u = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # State space recursion
        next_state = state
        outputs = []
        for t in range(seq_len):
            y = torch.einsum('bd,bd->b', self.C, next_state) + self.D * u[:, :, t]
            outputs.append(y)
            next_state = torch.einsum('bds,bd->bd', dA, next_state) + dB * u[:, :, t].unsqueeze(-1)
            
        output = torch.stack(outputs, dim=1)
        output = self.dropout(output)
        
        return output, next_state
        
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state tensor of shape (batch_size, d_state)
        """
        return torch.zeros(batch_size, self.d_state, device=self.A.device) 