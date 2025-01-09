# State Space Model Universal

A PyTorch implementation exploring and unifying various State Space Models (SSMs) and Transformer architectures for sequence modeling tasks. This research project investigates different sequence modeling approaches and their combinations, aiming to provide insights into their relative strengths and applications.

## Introduction

State Space Models and Transformers represent two major approaches to sequence modeling. This project explores several key innovations in sequence modeling research:

- **S4 (Structured State Space)**: The foundational SSM architecture that introduced HiPPO-based initialization, enabling efficient modeling of long-range dependencies through a structured state space approach.

- **S4D (Diagonal State Space)**: A simplified variant that uses diagonal state matrices, significantly reducing computational complexity while maintaining performance through careful parameterization and initialization strategies.

- **S5 (Simplified State Space)**: Further optimization of the S4 architecture, focusing on practical efficiency while preserving the essential mathematical properties that make SSMs effective.

- **Mamba**: A recent advancement introducing selective state spaces with linear attention, combining the efficiency of SSMs with the adaptability of attention mechanisms.

- **Transformer**: The standard self-attention based architecture, serving as both a baseline and a component for hybrid approaches.

Our implementation focuses on:
- Investigating the interaction between SSMs and attention mechanisms
- Understanding the impact of various initialization schemes
- Exploring hybrid architectures that combine different approaches

## Model Architecture

### Core Implementations
Each variant is implemented as a standalone layer, capturing its unique theoretical contributions:

- `S4Layer`: 
  - Implements the original structured state space formulation
  - Uses HiPPO initialization for enhanced long-range modeling
  - Supports both convolutional and recurrent modes

- `S4DLayer`: 
  - Implements diagonal state matrices for efficiency
  - Features multiple initialization schemes (inverse, linear, legs)
  - Optimized for faster training and inference

- `S5Layer`: 
  - Simplified version of S4 focusing on practical efficiency
  - Direct port of the JAX reference implementation
  - Maintains key mathematical properties while reducing complexity

- `MambaLayer`: 
  - Implements selective state space mechanism
  - Features S4D-style initialization options
  - Optimized for linear-time sequence processing

- `TransformerLayer`:
  - Standard multi-head self-attention mechanism
  - Includes position-wise feed-forward network
  - Supports flexible attention configurations

### Architecture Variants
We explore several architectural approaches:

- `H3Model`: 
  - Combines SSMs with convolution and gating
  - Designed for hierarchical sequence processing
  - Effective for language modeling tasks

- `GatedMLPModel`: 
  - Flexible architecture supporting optional SSM integration
  - Can function as pure MLP or hybrid model
  - Useful for ablation studies

- `MambaModel`: 
  - Optimized for selective state space operations
  - Features parallel processing capabilities
  - Efficient for long sequence modeling

- `TransformerModel`:
  - Standard Transformer encoder architecture
  - Supports various attention mechanisms
  - Useful for comparison and hybrid experiments

## Project Structure
```
models/
├── architectures/       # High-level model architectures
│   ├── h3.py           # H3 implementation
│   ├── gated_mlp.py    # Gated MLP implementation
│   ├── mamba.py        # Mamba implementation
│   └── transformer.py   # Transformer implementation
├── layers/             # Core implementations
│   ├── base.py         # Abstract interface
│   ├── s4_layer.py     # S4 implementation
│   ├── s4d_layer.py    # S4D implementation
│   ├── s5_layer.py     # S5 implementation
│   ├── mamba_layer.py  # Mamba implementation
│   └── transformer_layer.py  # Transformer implementation
└── utils/              # Shared utilities
    ├── mlp.py          # MLP components
    ├── conv.py         # Convolution utilities
    └── hippo.py        # HiPPO initialization
```

## Setup and Requirements

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Support
```bash
# Build the Docker image
docker build -t ssm-universal .

# Run with GPU support
docker run -it --gpus all ssm-universal
```

## Usage Examples

### Basic Experimentation
```python
from state_space_model_universal import create_model

# H3 architecture with S4 layer for language modeling
model = create_model(
    architecture="h3",
    ssm_layer="s4",
    d_model=256,
    d_state=64,
    num_layers=4
)

# Standard Transformer model
model = create_model(
    architecture="transformer",
    ssm_layer="transformer",
    d_model=256,
    num_layers=4,
    layer_kwargs={
        "num_heads": 8,
        "attention_dropout": 0.1
    }
)
```

### Advanced Research Configurations
```python
# Mamba with custom initialization for ablation study
model = create_model(
    architecture="mamba",
    ssm_layer="mamba",
    d_model=256,
    d_state=16,
    num_layers=4,
    layer_kwargs={
        "init_method": "inv",    # Test S4D initialization
        "expand_factor": 2       # State space expansion
    },
    architecture_kwargs={
        "d_conv": 4,            # Local context size
        "dropout": 0.1          # Regularization
    }
)

# Transformer with custom attention settings
model = create_model(
    architecture="transformer",
    ssm_layer="transformer",
    d_model=512,
    num_layers=6,
    layer_kwargs={
        "num_heads": 16,        # Multi-head attention
        "ffn_ratio": 4,         # Feed-forward expansion
        "attention_dropout": 0.1
    }
)

# Pure MLP baseline comparison
model = create_model(
    architecture="gmlp",
    ssm_layer=None,
    d_model=256,
    num_layers=4,
    architecture_kwargs={
        "expand_factor": 4,
        "activation": "gelu"
    }
)
```

### Component-Level Research
```python
from state_space_model_universal import (
    S4Layer, MambaLayer, TransformerLayer,
    H3Model, TransformerModel
)

# Experiment with specific SSM configurations
ssm_layer = S4Layer(
    d_model=256,
    d_state=64,
    dropout=0.1,
    bidirectional=True,
    hippo_method="legendre"  # Test different HiPPO variants
)

# Custom Transformer configuration
transformer = TransformerLayer(
    d_model=512,
    d_state=None,  # Not used in Transformer
    num_heads=8,
    attention_dropout=0.1,
    ffn_ratio=4
)

# Hybrid architecture experiment
model = H3Model(
    ssm_layer_class=TransformerLayer,  # Use Transformer as base layer
    d_model=256,
    d_state=64,
    num_layers=4,
    layer_kwargs={
        "num_heads": 8,
        "attention_dropout": 0.1
    }
)
```

## Model Configuration

### Core Parameters
- `d_model`: Hidden dimension size (default: 256)
- `d_state`: State space dimension (default: 64)
- `num_layers`: Number of stacked layers (default: 4)
- `max_seq_len`: Maximum sequence length (default: 2048)
- `dropout`: Dropout rate (default: 0.1)

### Layer-Specific Settings
- S4Layer:
  - `bidirectional`: Enable bidirectional processing
  - `dt_min/dt_max`: Discretization step size bounds
  - `hippo_method`: HiPPO initialization variant

- S4DLayer:
  - `init_method`: State matrix initialization ["inv", "lin", "legs"]
  - `real_transform`: Complex-to-real transformation option

- MambaLayer:
  - `expand_factor`: State space dimension multiplier
  - `init_method`: S4D-style initialization options

- TransformerLayer:
  - `num_heads`: Number of attention heads
  - `attention_dropout`: Dropout rate for attention weights
  - `ffn_ratio`: Feed-forward network expansion ratio

## References

- S4: "Structured State Spaces for Sequence Modeling" (https://arxiv.org/abs/2111.00396)
  - Introduces the foundational SSM architecture with HiPPO initialization

- S4D: "On the Parameterization and Initialization of Diagonal State Space Models"
  - Presents efficient diagonal variant with theoretical insights

- S5: "Simplified State Space Layers for Sequence Modeling" (https://arxiv.org/abs/2208.04933)
  - Optimizes S4 for practical applications

- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (https://arxiv.org/abs/2312.00752)
  - Introduces selective state spaces with linear attention

- H3: "H3: Language Modeling with State Space Models and Linear Attention"
  - Demonstrates SSM effectiveness in language modeling

- Transformer: "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
  - Introduces the Transformer architecture

## License

MIT License
