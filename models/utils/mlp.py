import torch
import torch.nn as nn
import torch.nn.functional as F

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