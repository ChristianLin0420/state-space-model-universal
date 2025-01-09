from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..utils.attention import MultiHeadAttention, PositionwiseFFN

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        attention_dropout: Dropout rate for attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            is_causal=True
        )
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        # Feed-forward network
        self.ffn = PositionwiseFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, L, D)
            memory: Encoder output (B, S, D)
            tgt_mask: Optional target mask (B, L, L)
            memory_mask: Optional memory mask (B, L, S)
            
        Returns:
            Output tensor (B, L, D)
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = x + residual
        
        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, memory_mask)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        
        return x 