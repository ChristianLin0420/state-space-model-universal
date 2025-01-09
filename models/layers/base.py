from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

class StateSpaceModel(nn.Module, ABC):
    """Base class for all state space models.
    
    This class defines the common interface that all state space models
    should implement, including S4, S5, Mamba, and Mamba2.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        **kwargs: Any
    ) -> None:
        """Initialize the state space model.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dropout = dropout
        
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state
            
        Returns:
            tuple: (output, new_state)
        """
        raise NotImplementedError
    
    @abstractmethod
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state tensor
        """
        raise NotImplementedError 