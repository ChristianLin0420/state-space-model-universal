"""State Space Model Universal Library

This library provides implementations of various state space models and architectures.
Each model can be used with different SSM layers to create flexible architectures.

Available SSM Layers:
- S4Layer: Structured State Space Sequence Model
- S4DLayer: Diagonal State Space Model
- S5Layer: Simplified State Space Model
- MambaLayer: Selective State Space Model
- Mamba2Layer: State Space Duality Model (Sequential/Parallel)
- TransformerLayer: Transformer Layer

Available Architectures:
- H3Model: SSM with convolution and gating
- GatedMLPModel: Optional SSM with parallel gating
- MambaModel: SSM with selective scan mechanism
- Mamba2Model: SSM with dual sequential/parallel modes
- TransformerModel: Transformer Model

Example usage:
```python
from state_space_model_universal import create_model

# Create H3 model with S4 layer
model = create_model(
    architecture="h3",
    ssm_layer="s4",
    d_model=256,
    d_state=64,
    num_layers=4
)

# Create Mamba model with Mamba layer
model = create_model(
    architecture="mamba",
    ssm_layer="mamba",
    d_model=256,
    d_state=16,
    num_layers=4
)

# Create Mamba2 model with sequential mode
model = create_model(
    architecture="mamba2",
    ssm_layer="mamba2",
    d_model=256,
    d_state=16,
    num_layers=4,
    layer_kwargs={"block_type": "sequential"}  # or "parallel"
)

# Create Gated MLP model with S5 layer
model = create_model(
    architecture="gmlp",
    ssm_layer="s5",
    d_model=256,
    d_state=64,
    num_layers=4
)
```
"""

from typing import Optional, Literal, Dict, Any

# Import base class
from .layers.base import StateSpaceModel

# Import all SSM layers
from .layers.s4_layer import S4Layer
from .layers.s4d_layer import S4DLayer
from .layers.s5_layer import S5Layer
from .layers.mamba_layer import MambaLayer
from .layers.mamba2_layer import Mamba2Layer
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .layers.transformer_decoder_layer import TransformerDecoderLayer

# Import all architectures
from .architectures.h3 import H3Model
from .architectures.gated_mlp import GatedMLPModel
from .architectures.mamba import MambaModel
from .architectures.mamba2 import Mamba2Model, SequentialMambaBlock, ParallelMambaBlock
from .architectures.transformer import TransformerModel

# Layer name to class mapping
SSM_LAYERS = {
    "s4": S4Layer,
    "s4d": S4DLayer,
    "s5": S5Layer,
    "mamba": MambaLayer,
    "mamba2": Mamba2Layer,
}

# Architecture name to class mapping
ARCHITECTURES = {
    "h3": H3Model,
    "gmlp": GatedMLPModel,
    "mamba": MambaModel,
    "mamba2": Mamba2Model,
    "transformer": TransformerModel,  # Standard encoder-decoder Transformer
}

def create_model(
    architecture: Literal["h3", "gmlp", "mamba", "mamba2", "transformer"],
    ssm_layer: Optional[Literal["s4", "s4d", "s5", "mamba", "mamba2"]] = None,
    d_model: int = 256,
    d_state: int = 64,
    num_layers: int = 4,
    max_seq_len: int = 2048,
    dropout: float = 0.1,
    layer_kwargs: Optional[Dict[str, Any]] = None,
    architecture_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a model with specified architecture and SSM layer.
    
    Note: Transformer architecture does not use SSM layers.
    
    Args:
        architecture: Type of architecture to use
        ssm_layer: Type of SSM layer to use (optional for GMLP, not used for Transformer)
        d_model: Model dimension
        d_state: State dimension (not used for Transformer)
        num_layers: Number of layers
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        layer_kwargs: Additional arguments for layer (e.g., block_type for Mamba2)
        architecture_kwargs: Additional arguments for architecture
        
    Returns:
        Model instance with specified architecture and layer
        
    Example:
        >>> model = create_model("h3", "s4", d_model=256, d_state=64)
        >>> model = create_model("gmlp", ssm_layer=None)  # Pure MLP without SSM
        >>> model = create_model("transformer", num_layers=6)  # Standard Transformer
        >>> model = create_model("mamba2", "mamba2", layer_kwargs={"block_type": "sequential"})
    """
    # Validate inputs
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(ARCHITECTURES.keys())}")
    
    if architecture != "transformer" and ssm_layer is not None and ssm_layer not in SSM_LAYERS:
        raise ValueError(f"Unknown SSM layer: {ssm_layer}. "
                        f"Available: {list(SSM_LAYERS.keys())}")
    
    # Get architecture class
    arch_cls = ARCHITECTURES[architecture]
    
    # Special handling for Transformer
    if architecture == "transformer":
        if ssm_layer is not None:
            raise ValueError("Transformer architecture does not use SSM layers")
        return arch_cls(
            d_model=d_model,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            **(architecture_kwargs or {})
        )
    
    # Get layer class for SSM architectures
    layer_cls = SSM_LAYERS.get(ssm_layer) if ssm_layer else None
    
    # Create model
    return arch_cls(
        ssm_layer_class=layer_cls,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        layer_kwargs=layer_kwargs or {},
        **(architecture_kwargs or {})
    )

# For convenience, also export classes directly
__all__ = [
    # Factory function
    "create_model",
    
    # Base class
    "StateSpaceModel",
    
    # SSM Layers
    "S4Layer",
    "S4DLayer", 
    "S5Layer",
    "MambaLayer",
    "Mamba2Layer",
    
    # Transformer Layers
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    
    # Architectures
    "H3Model",
    "GatedMLPModel",
    "MambaModel",
    "Mamba2Model",
    "TransformerModel",
    
    # Mamba2 Blocks
    "SequentialMambaBlock",
    "ParallelMambaBlock",
] 