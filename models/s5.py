from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel
from .hippo import hippo_initializer

class S5Layer(StateSpaceModel):
    """Simplified State Space Sequence (S5) model implementation.
    
    Paper: https://arxiv.org/abs/2208.04933
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init_scale: float = 1.0,
        hippo_method: str = "legendre"
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        
        # Initialize time step parameters
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.dt = nn.Parameter(torch.exp(log_dt))
        
        # Initialize diagonal state matrix with HiPPO
        hippo_matrix = hippo_initializer(d_state, method=hippo_method)
        self.Lambda = nn.Parameter(
            init_scale * torch.diag(hippo_matrix).unsqueeze(0).repeat(d_model, 1)
        )
        
        # Initialize input and output projections
        self.B = nn.Parameter(
            init_scale * torch.randn(d_model, d_state) / math.sqrt(d_state)
        )
        self.C = nn.Parameter(
            init_scale * torch.randn(d_model, d_state) / math.sqrt(d_state)
        )
        
        # Direct feed-forward term
        self.D = nn.Parameter(torch.zeros(d_model))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of S5.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = self.init_state(batch_size)
            
        # Compute diagonal state transition
        # exp(-dt * Lambda) for discrete-time state transition
        dA = torch.exp(-self.dt.unsqueeze(-1) * torch.exp(self.Lambda))  # (d_model, d_state)
        
        # Prepare input
        u = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # Initialize output storage
        outputs = []
        next_state = state
        
        # Selective scan - simplified state space operation
        for t in range(seq_len):
            # Output projection
            y = torch.einsum('bd,md->bm', next_state, self.C)
            
            # Add direct pathway
            y = y + self.D * u[:, :, t]
            outputs.append(y)
            
            # State update
            next_state = dA * next_state + self.B * u[:, :, t].unsqueeze(-1)
            
        # Stack outputs and apply dropout
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
        return torch.zeros(batch_size, self.d_state, device=self.Lambda.device) 