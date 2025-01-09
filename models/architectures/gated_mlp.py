from typing import Optional, Tuple, Type
import torch
import torch.nn as nn

from ..layers.base import StateSpaceModel
from ..utils.mlp import LayerNorm, MLP
from ..utils.conv import PositionalEmbedding

class GatedMLPBlock(nn.Module):
    """Gated MLP block with optional SSM path.
    
    Structure (from paper):
    1. Linear projection
    2. Parallel paths:
       - Main path: Activation
       - Gate path: Activation
    3. Optional SSM in main path
    4. Combine with multiplication
    """
    
    def __init__(
        self,
        ssm_layer: Optional[nn.Module],
        d_model: int,
        expand_factor: int = 4,
        dropout: float = 0.1,
        activation: nn.Module = nn.SiLU()
    ):
        super().__init__()
        
        # Layer normalization
        self.norm = LayerNorm(d_model)
        d_inner = expand_factor * d_model
        
        # Main path
        self.proj_in_1 = nn.Linear(d_model, d_inner)
        self.ssm = ssm_layer  # Optional SSM layer
        
        # Gate path
        self.proj_in_2 = nn.Linear(d_model, d_inner)
        
        # Shared components
        self.activation = activation
        self.proj_out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Gated MLP block.
        
        Args:
            x: Input tensor (B, L, D)
            state: Optional state for SSM
            
        Returns:
            tuple: (output, next_state)
        """
        # Input normalization
        residual = x
        x = self.norm(x)
        
        # Main path
        x1 = self.activation(self.proj_in_1(x))
        if self.ssm is not None:
            x1, next_state = self.ssm(x1, state)
        else:
            next_state = None
            
        # Gate path
        x2 = self.activation(self.proj_in_2(x))
        
        # Combine paths
        x = x1 * x2
        
        # Output projection
        x = self.proj_out(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, next_state

class GatedMLPModel(nn.Module):
    """Gated MLP architecture with optional SSM layers."""
    
    def __init__(
        self,
        ssm_layer_class: Optional[Type[StateSpaceModel]],
        d_model: int,
        d_state: int,
        num_layers: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        expand_factor: int = 4,
        layer_kwargs: dict = None,
    ):
        super().__init__()
        layer_kwargs = layer_kwargs or {}
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Stack of Gated MLP blocks
        self.layers = nn.ModuleList([
            GatedMLPBlock(
                ssm_layer=ssm_layer_class(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    **layer_kwargs
                ) if ssm_layer_class else None,
                d_model=d_model,
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
        """Forward pass through Gated MLP model.
        
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