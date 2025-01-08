from typing import Dict, Any
import torch.nn as nn

from .s4 import S4Layer
from .s5 import S5Layer
from .mamba import MambaLayer
from .mamba2 import Mamba2Layer
from .sequence_model import SequenceModel
from .s4d import S4DLayer

MODEL_REGISTRY = {
    "s4": S4Layer,
    "s4d": S4DLayer,
    "s5": S5Layer,
    "mamba": MambaLayer,
    "mamba2": Mamba2Layer
}

def get_model(cfg: Dict[str, Any]) -> nn.Module:
    """Get model instance based on configuration.
    
    Args:
        cfg: Model configuration containing type and parameters
        
    Returns:
        Instantiated SequenceModel with specified layer type
    """
    model_type = cfg.pop("type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get the layer class
    layer_class = MODEL_REGISTRY[model_type]
    
    # Extract SequenceModel specific parameters
    num_layers = cfg.pop("num_layers", 1)
    max_seq_len = cfg.pop("max_seq_len", 2048)
    
    # Create SequenceModel with the specified layer type
    model = SequenceModel(
        layer_class=layer_class,
        num_layers=num_layers,
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        max_seq_len=max_seq_len,
        dropout=cfg.dropout,
        layer_kwargs=cfg  # Pass remaining config as layer kwargs
    )
    
    return model 