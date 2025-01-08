from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .base import StateSpaceModel
from .hippo import hippo_initializer

class MambaLayer(StateSpaceModel):
    """Mamba model implementation.
    
    Paper: https://arxiv.org/abs/2312.00752
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        expansion_factor: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        
        self.d_inner = expansion_factor * d_model
        self.conv_kernel = conv_kernel
        
        # Input projection and expansion
        self.in_proj = nn.Linear(d_model, self.d_inner * 3)  # for x, delta, gamma
        
        # Convolutional layer for local context
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner
        )
        
        # Initialize SSM parameters with HiPPO
        hippo_matrix = hippo_initializer(d_state, method="legendre")
        self.A = nn.Parameter(
            hippo_matrix.unsqueeze(0).repeat(self.d_inner, 1, 1)
        )
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner) / math.sqrt(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Mamba.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and split input
        x_proj = self.in_proj(x)  # (batch_size, seq_len, 3 * d_inner)
        x_proj = rearrange(x_proj, 'b l (n d) -> n b l d', n=3)
        x_in, delta, gamma = x_proj  # Each of shape (batch_size, seq_len, d_inner)
        
        # Apply convolution for local context
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv(x_conv)[..., :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size)
        
        # Prepare outputs
        outputs = []
        next_state = state
        
        # Selective scan
        for t in range(seq_len):
            # Current input and modulation
            xt = x_conv[:, t]
            dt = torch.sigmoid(delta[:, t])  # Input-dependent step size
            gt = torch.sigmoid(gamma[:, t])  # Input-dependent gate
            
            # State space computation
            y = torch.einsum('bd,md->bm', next_state, self.C)
            y = y + self.D * xt
            y = gt * y + (1 - gt) * xt  # Gated update
            
            outputs.append(y)
            
            # Update state with input-dependent dynamics
            dA = torch.matrix_exp(self.A * dt.unsqueeze(-1).unsqueeze(-1))
            next_state = torch.einsum('bmd,bd->bm', dA, next_state) + \
                        torch.einsum('bd,md->bm', xt.unsqueeze(-1), self.B)
        
        # Stack outputs and project back
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_inner)
        output = self.out_proj(output)  # (batch_size, seq_len, d_model)
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