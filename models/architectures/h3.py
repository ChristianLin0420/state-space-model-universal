from typing import Optional, Tuple, Type
import torch
import torch.nn as nn

from ..layers.base import StateSpaceModel
from ..utils.mlp import LayerNorm
from ..utils.conv import PositionalEmbedding, ConvBlock

class H3Block(nn.Module):
    """H3 block combining SSM with convolution and gating.
    
    Structure (from paper):
    1. Linear projection
    2. Multiplicative gate
    3. SSM
    4. Multiplicative gate
    5. Conv
    6. Linear projections
    """
    
    def __init__(
        self,
        ssm_layer: nn.Module,
        d_model: int,
        d_conv: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer normalization
        self.norm = LayerNorm(d_model)
        
        # Input projection
        self.proj_in = nn.Linear(d_model, d_model)
        
        # SSM layer
        self.ssm = ssm_layer
        
        # Convolution
        self.conv = ConvBlock(
            d_model=d_model,
            kernel_size=d_conv,
            activation=nn.Identity()
        )
        
        # Output projections
        self.proj_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through H3 block.
        
        Args:
            x: Input tensor (B, L, D)
            state: Optional state for SSM
            
        Returns:
            tuple: (output, next_state)
        """
        # Input normalization and projection
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        
        # First multiplicative gate
        gate1 = torch.sigmoid(x)
        x = x * gate1
        
        # SSM
        x, next_state = self.ssm(x, state)
        
        # Second multiplicative gate
        gate2 = torch.sigmoid(x)
        x = x * gate2
        
        # Convolution
        x = self.conv(x)
        
        # Output projection
        x = self.proj_out(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, next_state

class H3Model(nn.Module):
    """H3 architecture with flexible SSM layer choice."""
    
    def __init__(
        self,
        ssm_layer_class: Type[StateSpaceModel],
        d_model: int,
        d_state: int,
        num_layers: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        d_conv: int = 4,
        layer_kwargs: dict = None,
    ):
        super().__init__()
        layer_kwargs = layer_kwargs or {}
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Stack of H3 blocks
        self.layers = nn.ModuleList([
            H3Block(
                ssm_layer=ssm_layer_class(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    **layer_kwargs
                ),
                d_model=d_model,
                d_conv=d_conv,
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
        """Forward pass through H3 model.
        
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