from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..layers.transformer_encoder_layer import TransformerEncoderLayer
from ..layers.transformer_decoder_layer import TransformerDecoderLayer
from ..utils.conv import PositionalEmbedding

class TransformerModel(nn.Module):
    """Standard Transformer encoder-decoder architecture.
    
    Args:
        d_model: Model dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        attention_dropout: Dropout rate for attention weights
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        d_model: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Embeddings
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input sequence.
        
        Args:
            src: Source tensor (B, L, D)
            src_mask: Optional source mask (B, L, L)
            
        Returns:
            Encoder output (B, L, D)
        """
        x = self.pos_emb(src)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        return self.norm(x)
        
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence.
        
        Args:
            tgt: Target tensor (B, L, D)
            memory: Encoder output (B, S, D)
            tgt_mask: Optional target mask (B, L, L)
            memory_mask: Optional memory mask (B, L, S)
            
        Returns:
            Decoder output (B, L, D)
        """
        x = self.pos_emb(tgt)
        
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
            
        return self.norm(x)
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            src: Source tensor (B, L, D)
            tgt: Target tensor (B, L, D)
            src_mask: Optional source mask (B, L, L)
            tgt_mask: Optional target mask (B, L, L)
            memory_mask: Optional memory mask (B, L, S)
            
        Returns:
            Output tensor (B, L, D)
        """
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask, memory_mask) 