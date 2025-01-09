import torch
import torch.nn as nn
from einops import rearrange

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding layer."""
    
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) / d_model ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.embedding[:, :seq_len]

class ConvBlock(nn.Module):
    """Depthwise separable convolution block."""
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        groups: int = None,
        activation: nn.Module = nn.SiLU()
    ) -> None:
        super().__init__()
        groups = groups or d_model
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv(x)
        x = rearrange(x, 'b d l -> b l d')
        return self.activation(x) 