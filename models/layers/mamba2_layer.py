from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..base import StateSpaceModel
from ..utils.mlp import LayerNorm, MLP

class Mamba2Layer(StateSpaceModel):
    """Mamba2 model implementation with improved architecture.
    
    Enhancements over Mamba:
    - Parallel state computation
    - Improved gating mechanism
    - Layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        expansion_factor: int = 2,
        conv_kernel: int = 4,
        selective_scan: bool = True,
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        
        self.d_inner = expansion_factor * d_model
        self.conv_kernel = conv_kernel
        self.selective_scan = selective_scan
        
        # Normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(self.d_inner)
        
        # Input projection and expansion
        self.in_proj = nn.Linear(d_model, self.d_inner * 4)  # x, delta, gamma, beta
        
        # Convolutional layer for local context
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state, d_state) / math.sqrt(d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner) / math.sqrt(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Feed-forward network
        self.mlp = MLP(d_model, expansion_factor=4, dropout=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform selective scan operation.
        
        Args:
            x: Input tensor
            delta: Time step modulation
            gamma: Output modulation
            beta: State modulation
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = self.init_state(batch_size)
            
        # Prepare outputs
        outputs = []
        next_state = state
        
        for t in range(seq_len):
            # Current timestep tensors
            xt = x[:, t]
            dt = torch.sigmoid(delta[:, t])
            gt = torch.sigmoid(gamma[:, t])
            bt = torch.sigmoid(beta[:, t])
            
            # State space computation
            y = torch.einsum('bd,md->bm', next_state, self.C)
            y = y + self.D * xt
            
            # Gated update with state modulation
            y = gt * y + (1 - gt) * xt
            outputs.append(y)
            
            # Update state with input and state modulation
            dA = torch.matrix_exp(self.A * dt.unsqueeze(-1).unsqueeze(-1))
            next_state = bt * (
                torch.einsum('bmd,bd->bm', dA, next_state) + 
                torch.einsum('bd,md->bm', xt.unsqueeze(-1), self.B)
            ) + (1 - bt) * next_state
            
        return torch.stack(outputs, dim=1), next_state
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Mamba2.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        # Input normalization
        x = self.norm1(x)
        
        # Project and split input
        x_proj = self.in_proj(x)
        x_proj = rearrange(x_proj, 'b l (n d) -> n b l d', n=4)
        x_in, delta, gamma, beta = x_proj
        
        # Apply convolution for local context
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv(x_conv)[..., :x.size(1)]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.norm2(x_conv)
        
        # Apply selective scan
        output, next_state = self._selective_scan(x_conv, delta, gamma, beta, state)
        
        # Project back to original dimension
        output = self.out_proj(output)
        output = self.dropout(output)
        
        # Residual connection and MLP
        output = x + output
        output = output + self.mlp(output)
        
        return output, next_state
        
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state tensor
        """
        return torch.zeros(batch_size, self.d_state, device=self.A.device) 