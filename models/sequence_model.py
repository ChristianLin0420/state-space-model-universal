from typing import Optional, List, Type, Tuple
import torch
import torch.nn as nn

from .base import StateSpaceModel
from .utils import PositionalEmbedding

class SequenceModel(nn.Module):
    """Wrapper class for building deep state space models."""
    
    def __init__(
        self,
        layer_class: Type[StateSpaceModel],
        num_layers: int,
        d_model: int,
        d_state: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        layer_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize sequence model.
        
        Args:
            layer_class: State space layer class to use
            num_layers: Number of layers
            d_model: Model dimension
            d_state: State dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            layer_kwargs: Additional arguments for layer initialization
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Stack of state space layers
        layer_kwargs = layer_kwargs or {}
        self.layers = nn.ModuleList([
            layer_class(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                **layer_kwargs
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through all layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            states: Optional list of previous states for each layer
            
        Returns:
            tuple: (output, list_of_new_states)
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
            
        return x, new_states