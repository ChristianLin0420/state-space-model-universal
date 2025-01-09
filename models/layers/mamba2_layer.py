"""Mamba2 (State Space Duality) Layer Implementation.

This module implements the Mamba2 layer based on the State Space Duality (SSD) framework.
The implementation follows the block matrix decomposition algorithm with four steps:
1. Intra-chunk outputs (diagonal blocks)
2. Chunk states (right term of low-rank factorization)
3. Pass states (inter-chunk SSM recurrence)
4. Output states (left term of low-rank factorization)

Reference:
    https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..utils.mlp import MLP

def segsum(x: torch.Tensor) -> torch.Tensor:
    """Compute segment sums in log space for numerical stability.
    
    This computes sums over segments [i:j] without using subtraction,
    which is important for numerical stability of the SSM.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Segment sums in log space
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

class Mamba2Layer(nn.Module):
    """Mamba2 layer implementing the State Space Duality (SSD) mechanism.
    
    The layer uses a block matrix decomposition algorithm that combines:
    1. Hardware-efficient matrix multiplications for most operations
    2. Linear-time SSM processing for cross-chunk connections
    
    Args:
        d_model (int): Model dimension
        d_state (int): State dimension (N)
        d_head (int): Head dimension (P), analogous to attention head size
        expand_factor (int, optional): Expansion factor for input projection. Defaults to 2.
        dt_rank (int, optional): Rank for delta (timestep) projection. Defaults to 8.
        dt_min (float, optional): Minimum delta value. Defaults to 0.001.
        dt_max (float, optional): Maximum delta value. Defaults to 0.1.
        dt_init (str, optional): Delta initialization method ['random', 'uniform']. Defaults to 'random'.
        dt_scale (float, optional): Scale factor for delta initialization. Defaults to 1.0.
        block_size (int, optional): Size of blocks for chunked processing. Defaults to 64.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_head: int = 64,
        expand_factor: int = 2,
        dt_rank: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0,
        block_size: int = 64,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.block_size = block_size
        
        # Input projection
        self.d_inner = int(expand_factor * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=bias)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(1))  # Scalar A (shared across time)
        self.B = nn.Parameter(torch.randn(d_state))
        self.C = nn.Parameter(torch.randn(d_state))
        
        # Delta (timestep) parameters
        self.dt_proj = nn.Linear(self.d_inner, dt_rank, bias=False)
        self.dt_scale = dt_scale
        self.register_buffer('dt_init_bound', torch.tensor([dt_min, dt_max]))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights(dt_init)
    
    def _init_weights(self, dt_init: str):
        """Initialize layer weights.
        
        Args:
            dt_init (str): Delta initialization method ['random', 'uniform']
        """
        # Initialize A to be stable (negative real part)
        with torch.no_grad():
            self.A_log.data.uniform_(-2, -1)
        
        # Initialize B and C using Gaussian
        nn.init.normal_(self.B, std=0.1)
        nn.init.normal_(self.C, std=0.1)
        
        # Initialize delta projection
        if dt_init == 'random':
            nn.init.normal_(self.dt_proj.weight, std=0.1)
        else:  # uniform
            nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
    
    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent timesteps (delta).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D)
            
        Returns:
            torch.Tensor: Delta tensor of shape (B, L)
        """
        dt = self.dt_proj(x)  # (B, L, R)
        dt = torch.sigmoid(dt) * self.dt_scale  # Scale to [0, dt_scale]
        dt = dt * (self.dt_init_bound[1] - self.dt_init_bound[0]) + self.dt_init_bound[0]
        return dt.mean(dim=-1)  # Average across rank dimension
    
    def _ssd_forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using the SSD block matrix decomposition algorithm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, H, P)
            delta (torch.Tensor): Timestep tensor of shape (B, L, H)
            initial_state (Optional[torch.Tensor]): Initial state. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and final state
        """
        # Ensure sequence length is divisible by block size
        B, L, H, P = x.shape
        assert L % self.block_size == 0, f"Sequence length must be divisible by block_size {self.block_size}"
        
        # Rearrange into blocks/chunks
        x = rearrange(x, 'b (c l) h p -> b c l h p', l=self.block_size)
        delta = rearrange(delta, 'b (c l) h -> b h c l', l=self.block_size)
        
        # Compute cumulative sums for A
        A = torch.exp(self.A_log)
        A_cumsum = torch.cumsum(delta, dim=-1)
        
        # Step 1: Compute intra-chunk outputs (diagonal blocks)
        L = torch.exp(segsum(delta))  # Lower triangular matrix
        Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp',
                            self.C.expand_as(x),
                            self.B.expand_as(x),
                            L,
                            x)
        
        # Step 2: Compute chunk states (right term)
        decay_states = torch.exp((A_cumsum[..., -1:] - A_cumsum))
        states = torch.einsum('bclhn,bhcl,bclhp->bchpn',
                           self.B.expand_as(x),
                           decay_states,
                           x)
        
        # Step 3: Compute inter-chunk SSM recurrence
        if initial_state is None:
            initial_state = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_state, states], dim=1)
        decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
        new_states = torch.einsum('bhzc,bchpn->bzhpn',
                               decay_chunk,
                               states)
        states, final_state = new_states[:, :-1], new_states[:, -1]
        
        # Step 4: Compute state -> output conversion
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp',
                          self.C.expand_as(x),
                          states,
                          state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, 'b c l h p -> b (c l) h p')
        
        return Y, final_state
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using the SSD algorithm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D)
            state (Optional[torch.Tensor]): Initial state. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and final state
        """
        # Project input
        x = self.in_proj(x)
        
        # Reshape for multi-head processing
        B, L, D = x.shape
        x = rearrange(x, 'b l (h p) -> b l h p', h=self.d_head)
        
        # Compute input-dependent timesteps
        delta = self._compute_delta(x)
        
        # Apply SSD algorithm
        output, final_state = self._ssd_forward(x, delta, state)
        
        # Reshape and project output
        output = rearrange(output, 'b l h p -> b l (h p)')
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, final_state 