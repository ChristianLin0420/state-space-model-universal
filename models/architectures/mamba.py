from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.base import StateSpaceModel
from ..utils.mlp import LayerNorm
from ..utils.conv import PositionalEmbedding, ConvBlock

class MambaBlock(nn.Module):
    """Mamba block combining SSM with parallel gating.
    
    Structure (from paper):
    1. Parallel paths:
       - Main path: Conv -> SiLU -> SSM
       - Gate path: SiLU
    2. Combine with multiplication
    """
    
    def __init__(
        self,
        ssm_layer: nn.Module,
        d_model: int,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer normalization
        self.norm = LayerNorm(d_model)
        d_inner = expand_factor * d_model
        
        # Main path
        self.proj_in_1 = nn.Linear(d_model, d_inner)
        self.conv = ConvBlock(
            d_model=d_inner,
            kernel_size=d_conv,
            activation=nn.SiLU()
        )
        self.ssm = ssm_layer
        
        # Gate path
        self.proj_in_2 = nn.Linear(d_model, d_inner)
        
        # Output projection
        self.proj_out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Mamba block.
        
        Args:
            x: Input tensor (B, L, D)
            state: Optional state for SSM
            
        Returns:
            tuple: (output, next_state)
        """
        # Input normalization
        residual = x
        x = self.norm(x)
        
        # Main path: projection -> conv -> ssm
        x1 = self.proj_in_1(x)
        x1 = self.conv(x1)
        x1, next_state = self.ssm(x1, state)
        
        # Gate path: projection -> silu
        x2 = F.silu(self.proj_in_2(x))
        
        # Combine paths
        x = x1 * x2
        
        # Output projection
        x = self.proj_out(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, next_state

class MambaModel(nn.Module):
    """Mamba architecture with flexible SSM layer choice."""
    
    def __init__(
        self,
        ssm_layer_class: Type[StateSpaceModel],
        d_model: int,
        d_state: int,
        num_layers: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        d_conv: int = 4,
        expand_factor: int = 2,
        layer_kwargs: dict = None,
    ):
        super().__init__()
        layer_kwargs = layer_kwargs or {}
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                ssm_layer=ssm_layer_class(
                    d_model=d_model * expand_factor,  # SSM operates on expanded dimension
                    d_state=d_state,
                    dropout=dropout,
                    **layer_kwargs
                ),
                d_model=d_model,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through Mamba model.
        
        Args:
            x: Input tensor (B, L, D)
            states: Optional list of states for each layer
            
        Returns:
            tuple: (output, list_of_states)
        """
        # Add positional embeddings
        x = self.pos_emb(x)
        
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.layers)
        
        # Process through layers
        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)
            
        # Final normalization
        x = self.norm(x)
        
        return x, new_states 