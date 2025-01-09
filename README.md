# State Space Model Universal

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation exploring and unifying various State Space Models (SSMs) and Transformer architectures for sequence modeling tasks. This research project investigates different sequence modeling approaches and their combinations, aiming to provide insights into their relative strengths and applications.

[Overview](#introduction) •
[Models](#features) •
[Setup](#installation) •
[Examples](#usage) •
[Contributing](#contributing)

</div>

---

## Introduction

This research project explores and implements various state-of-the-art sequence modeling approaches, focusing on two major paradigms:

### State Space Models (SSMs)
SSMs represent a promising direction in sequence modeling research:
- **Linear Time Complexity**: O(N) processing for sequences of length N
- **Efficient Long-range Modeling**: Structured state representations for capturing dependencies
- **Theoretical Foundations**: Based on continuous-time dynamical systems
- **Hardware Efficiency**: Particularly well-suited for modern accelerators

### Transformers
The established baseline for sequence modeling, Transformers provide:
- **Parallel Processing**: Efficient batch processing of sequence elements
- **Global Context**: Direct modeling of relationships between any positions
- **Flexible Representations**: Self-attention mechanism for adaptive feature extraction
- **Proven Effectiveness**: State-of-the-art results across various domains

This project aims to:
- ✨ Implement and compare different sequence modeling architectures
- 🔧 Explore combinations of SSMs and attention mechanisms
- 🔄 Investigate hybrid approaches and their effectiveness
- 📊 Provide empirical insights into model behaviors

### Key Implementations

Our project includes several state-of-the-art architectures:

#### State Space Models
- **S4 (Structured State Space)**
  - HiPPO-based initialization for enhanced long-range modeling
  - Structured parameterization for stability and expressiveness
  - Bidirectional processing capability

- **S4D (Diagonal State Space)**
  - Optimized diagonal state matrices
  - Improved training efficiency
  - Maintained performance with reduced complexity

- **S5 (Simplified State Space)**
  - Streamlined architecture for practical applications
  - Reduced parameter count without sacrificing capability
  - Enhanced numerical stability

- **Mamba**
  - Selective state space mechanism
  - Data-dependent sparsity
  - Linear-time attention alternative

#### Hybrid Approaches
- **H3**: Experimental combination of SSMs with convolution and gating
- **Gated MLP**: Investigation of SSM components with MLP architectures

---

## Features

### State Space Models (SSMs)
Our SSM implementations provide:
- Efficient sequence processing with linear complexity
- Multiple initialization schemes for different use cases
- Comprehensive configuration options for research
- Built-in support for both training and inference

### Transformer Components
The Transformer implementation offers:
- Full encoder-decoder architecture with modern improvements
- Flexible attention patterns and masking
- Optimized multi-head attention implementation
- Customizable position embeddings

### Architectures
Each architecture is designed for:
- Modular composition of different components
- Easy experimentation with hybrid approaches
- Efficient scaling to different model sizes
- Research-friendly configuration options

---

## Installation

### Prerequisites
Ensure your system meets these requirements:
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA toolkit (optional, for GPU support)
- 4GB+ RAM for basic usage, 8GB+ recommended

### Basic Installation
For most users, the following is sufficient:
```bash
pip install -e .
```

### Development Installation
For contributors and researchers:
```bash
# Clone the repository
git clone https://github.com/yourusername/state-space-model-universal.git
cd state-space-model-universal

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Creating Models

The library provides several ways to create and customize models:

#### Basic Model Creation
For quick experimentation and standard use cases:
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

# Create Mamba model
model = create_model(
    architecture="mamba",
    ssm_layer="mamba",
    d_model=256,
    d_state=16,
    num_layers=4
)

# Create standard Transformer
model = create_model(
    architecture="transformer",
    d_model=512,
    num_layers=6,  # Same number for encoder and decoder
    max_seq_len=2048
)
```

#### Advanced Configuration
For research and specialized applications:
```python
# H3 model with custom layer settings
model = create_model(
    architecture="h3",
    ssm_layer="s4d",
    d_model=512,
    d_state=64,
    num_layers=6,
    layer_kwargs={
        "init_method": "legs",  # S4D initialization method
        "bidirectional": True,  # Enable bidirectional processing
    },
    architecture_kwargs={
        "activation": "gelu",
        "dropout": 0.1,
    }
)

# Transformer with custom attention settings
model = create_model(
    architecture="transformer",
    d_model=768,
    num_layers=12,
    architecture_kwargs={
        "num_heads": 12,
        "d_ff": 3072,
        "attention_dropout": 0.1,
        "max_seq_len": 4096,
    }
)
```

### Using Individual Components

#### SSM Layers
Fine-grained control over SSM components:
```python
from state_space_model_universal import S4Layer, MambaLayer

# S4 layer with advanced configuration
s4 = S4Layer(
    d_model=256,
    d_state=64,
    bidirectional=True,
    dropout=0.1,
    init_method="legs"
)
output = s4(input_tensor)

# Mamba layer with custom settings
mamba = MambaLayer(
    d_model=256,
    d_state=16,
    expand_factor=2,
    init_method="inv"
)
output = mamba(input_tensor)
```

#### Transformer Components
Modular Transformer building blocks:
```python
from state_space_model_universal import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerModel
)

# Single encoder layer with custom settings
encoder_layer = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    attention_dropout=0.1
)
encoded = encoder_layer(x, mask=None)

# Single decoder layer with memory
decoder_layer = TransformerDecoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)
decoded = decoder_layer(x, memory, tgt_mask=None, memory_mask=None)

# Full Transformer with custom configuration
transformer = TransformerModel(
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    attention_dropout=0.1,
    max_seq_len=2048
)

# Forward pass with masking
src_mask = torch.ones(batch_size, src_len, src_len)  # Optional source mask
tgt_mask = torch.triu(torch.ones(batch_size, tgt_len, tgt_len), diagonal=1)  # Causal mask
output = transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
```

### Training and Evaluation

Standard PyTorch training loop with our models:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize model and training components
model = create_model("h3", "s4", d_model=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch.input)
        loss = criterion(output, batch.target)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        output = model(batch.input)
        # Compute metrics
```

---

## Architecture Details

### State Space Models
Each SSM variant offers unique advantages:

- **S4**: 
  - Full state space model with structured matrices
  - HiPPO-based initialization for enhanced modeling
  - Complex-valued state representations

- **S4D**: 
  - Diagonal version of S4 for improved efficiency
  - Multiple initialization options
  - Real-valued computations

- **S5**: 
  - Simplified version with similar performance
  - Reduced parameter count
  - Enhanced training stability

- **Mamba**: 
  - Selective state space model
  - Data-dependent sparsity
  - Linear-time attention mechanism

### Transformer Architecture
Our Transformer implementation follows the standard encoder-decoder architecture with modern improvements:

#### Encoder
- **Self-attention**: Multi-head attention for input sequence relationships
- **Position-wise FFN**: Two-layer feed-forward network
- **Layer Norm**: Pre-norm configuration for stability
- **Residual Connections**: For gradient flow

#### Decoder
- **Masked Self-attention**: For autoregressive generation
- **Cross-attention**: Attention to encoder outputs
- **Position-wise FFN**: Similar to encoder
- **Layer Norm & Residuals**: For training stability

### Hybrid Architectures
Novel combinations of different approaches:

- **H3**: 
  - SSM for sequence modeling
  - Convolution for local processing
  - Gating for adaptive computation

- **Gated MLP**: 
  - Optional SSM components
  - Parallel gating mechanism
  - Flexible architecture

---

## Project Structure
```
models/
├── architectures/       # High-level model architectures
│   ├── h3.py           # H3 implementation
│   ├── gated_mlp.py    # Gated MLP implementation
│   ├── mamba.py        # Mamba implementation
│   └── transformer.py   # Transformer implementation
├── layers/             # Core layer implementations
│   ├── base.py         # Abstract base classes
│   ├── s4_layer.py     # S4 implementation
│   ├── s4d_layer.py    # S4D implementation
│   ├── s5_layer.py     # S5 implementation
│   ├── mamba_layer.py  # Mamba implementation
│   ├── transformer_encoder_layer.py
│   └── transformer_decoder_layer.py
└── utils/              # Shared utilities
    ├── attention.py    # Attention mechanisms
    ├── mlp.py          # MLP components
    ├── conv.py         # Convolution utilities
    └── hippo.py        # HiPPO initialization
```

---

## References

### State Space Models
1. [S4: Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2111.00396)
   - Original SSM architecture with HiPPO initialization
   - Theoretical foundations and empirical results

2. [S4D: Linear-Time State Space Models with Diagonal State](https://arxiv.org/abs/2206.11893)
   - Efficient diagonal variant
   - Improved training dynamics

3. [S5: Simple State Space for Sequence Modeling](https://arxiv.org/abs/2208.04933)
   - Simplified architecture
   - Practical improvements

4. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
   - Selective state space mechanism
   - Modern architecture improvements

### Transformer and Attention
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Original Transformer architecture
   - Multi-head attention mechanism

2. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
   - Extended context modeling
   - Relative position embeddings

---

## Contributing

We welcome contributions! Here's how you can help:

- 🐛 Report bugs and issues
- 💡 Propose new features
- 📝 Improve documentation
- 🔧 Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by the State Space Model Universal team
</div>
