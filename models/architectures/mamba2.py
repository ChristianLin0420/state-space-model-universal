"""Mamba2 Architecture Implementation.

This module implements two variants of the Mamba2 architecture:
1. Sequential Mamba Block: SSM parameters are produced as a function of the SSM input X
2. Parallel Mamba Block: SSM parameters are produced at the beginning of the block

Reference:
    Mamba2 paper Figure 6: Sequential vs Parallel architecture variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

from ..utils.mlp import LayerNorm
from ..utils.conv import ConvBlock, PositionalEmbedding
from ..layers.base import StateSpaceModel


class SequentialMambaBlock(nn.Module):
    """Sequential variant of Mamba2 block.
    
    In this variant, SSM parameters A, B, C are produced as a function of the SSM input X.
    Features:
    - Sequential linear projections
    - SSM with input-dependent parameters
    - Convolution layer
    - Gating mechanism
    
    Args:
        ssm_layer (nn.Module): SSM layer instance
        d_model (int): Model dimension
        d_conv (int, optional): Convolution dimension. Defaults to 4.
        expand_factor (int, optional): Expansion factor. Defaults to 2.
        conv_kernel_size (int, optional): Convolution kernel size. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    
    def __init__(
        self,
        ssm_layer: nn.Module,
        d_model: int,
        d_conv: int = 4,
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = expand_factor * d_model
        
        # Layer normalization
        self.norm = LayerNorm(d_model)
        
        # Input projection
        self.proj_in = nn.Linear(d_model, d_inner)
        
        # SSM layer
        self.ssm = ssm_layer
        
        # Convolution
        self.conv = ConvBlock(
            d_model=d_inner,
            d_conv=d_conv,
            kernel_size=conv_kernel_size,
            activation=nn.SiLU()
        )
        
        # Output projection
        self.proj_out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Sequential Mamba Block.
        
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
        
        # SSM processing
        x_ssm, new_state = self.ssm(x, state)
        
        # Convolution branch
        x_conv = self.conv(x)
        
        # Combine branches with gating
        x = x_ssm * F.sigmoid(x_conv)
        
        # Output projection
        x = self.proj_out(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, new_state


class ParallelMambaBlock(nn.Module):
    """Parallel variant of Mamba2 block.
    
    In this variant, SSM parameters A, B, C are produced at the beginning of the block.
    Features:
    - Normalization layer before SSM
    - SSM with fixed parameters
    - Convolution layer
    - Gating mechanism
    
    Args:
        ssm_layer (nn.Module): SSM layer instance
        d_model (int): Model dimension
        d_conv (int, optional): Convolution dimension. Defaults to 4.
        expand_factor (int, optional): Expansion factor. Defaults to 2.
        conv_kernel_size (int, optional): Convolution kernel size. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    
    def __init__(
        self,
        ssm_layer: nn.Module,
        d_model: int,
        d_conv: int = 4,
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = expand_factor * d_model
        
        # Layer normalization
        self.norm = LayerNorm(d_model)
        
        # Input projection
        self.proj_in = nn.Linear(d_model, d_inner)
        
        # SSM layer
        self.ssm = ssm_layer
        
        # Convolution
        self.conv = ConvBlock(
            d_model=d_inner,
            d_conv=d_conv,
            kernel_size=conv_kernel_size,
            activation=nn.SiLU()
        )
        
        # Output projection
        self.proj_out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Parallel Mamba Block.
        
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
        
        # SSM processing
        x_ssm, new_state = self.ssm(x, state)
        
        # Convolution branch
        x_conv = self.conv(x)
        
        # Combine branches with gating
        x = x_ssm * F.sigmoid(x_conv)
        
        # Output projection
        x = self.proj_out(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, new_state


class Mamba2Model(nn.Module):
    """Mamba2 architecture with flexible SSM layer choice.
    
    Args:
        ssm_layer_class (Type[StateSpaceModel]): SSM layer class to use
        d_model (int): Model dimension
        d_state (int): State dimension
        num_layers (int): Number of layers
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout rate
        d_conv (int, optional): Convolution dimension. Defaults to 4.
        expand_factor (int, optional): Expansion factor. Defaults to 2.
        conv_kernel_size (int, optional): Convolution kernel size. Defaults to 4.
        layer_kwargs (dict, optional): Additional arguments for SSM layer. Defaults to None.
    """
    
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
        conv_kernel_size: int = 4,
        layer_kwargs: dict = None,
    ):
        super().__init__()
        layer_kwargs = layer_kwargs or {}
        
        # Get block type from layer kwargs
        block_type = layer_kwargs.pop("block_type", "sequential")
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Select block type
        block_class = {
            'sequential': SequentialMambaBlock,
            'parallel': ParallelMambaBlock,
        }[block_type.lower()]
        
        # Stack of Mamba2 blocks
        self.layers = nn.ModuleList([
            block_class(
                ssm_layer=ssm_layer_class(
                    d_model=d_model * expand_factor,  # SSM operates on expanded dimension
                    d_state=d_state,
                    dropout=dropout,
                    **layer_kwargs
                ),
                d_model=d_model,
                d_conv=d_conv,
                expand_factor=expand_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
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
        """Forward pass through Mamba2 model.
        
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