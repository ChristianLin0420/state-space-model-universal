from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding layer."""
    
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) / d_model ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional embeddings added
        """
        seq_len = x.size(1)
        return x + self.embedding[:, :seq_len]

class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    
    def __init__(self, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)

class MLP(nn.Module):
    """Multi-layer perceptron with residual connection."""
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ) -> None:
        super().__init__()
        d_hidden = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) 