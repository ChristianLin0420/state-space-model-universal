from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        is_causal: Whether to use causal attention mask
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        is_causal: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            q: Query tensor (B, L, D)
            k: Key tensor (B, S, D)
            v: Value tensor (B, S, D)
            mask: Optional attention mask (B, L, S)
            
        Returns:
            Output tensor (B, L, D)
        """
        B, L, _ = q.shape
        _, S, _ = k.shape
        
        # Project and reshape
        q = rearrange(self.q_proj(q), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.k_proj(k), 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(self.v_proj(v), 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        if self.is_causal:
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        return self.out_proj(out)

class PositionwiseFFN(nn.Module):
    """Position-wise feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 