# State Space Model Universal

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation exploring and unifying various State Space Models (SSMs) and Transformer architectures for sequence modeling tasks. This research project investigates different sequence modeling approaches and their combinations, aiming to provide insights into their relative strengths and applications.

[Overview](#introduction) •
[Models](#implemented-models) •
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

## Implemented Models

### State Space Models (SSMs)
Our implementations explore various SSM architectures:
- Efficient sequence processing with linear complexity
- Different initialization schemes for comparison
- Experimental configuration options
- Performance analysis capabilities

### Transformer Components
Our Transformer implementation includes:
- Standard encoder-decoder architecture
- Configurable attention patterns and masking
- Multi-head attention implementation
- Position embedding variants

### Experimental Architectures
Our research investigates:
- Modular combinations of different components
- Novel hybrid approaches
- Scaling behavior analysis
- Comparative performance studies

## Setup

### Prerequisites
Required for running experiments:
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA toolkit (optional, for GPU support)
- 4GB+ RAM for basic usage, 8GB+ recommended

### Basic Setup
For running experiments:
```bash
pip install -e .
```

### Development Setup
For extending the research:
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

## Usage Examples

### Model Creation

The project provides various model implementations for experimentation:

#### Basic Examples
Simple configurations for initial experiments:
```python
from state_space_model_universal import create_model

# Experiment with H3 and S4
model = create_model(
    architecture="h3",
    ssm_layer="s4",
    d_model=256,
    d_state=64,
    num_layers=4
)

# Try Mamba architecture
model = create_model(
    architecture="mamba",
    ssm_layer="mamba",
    d_model=256,
    d_state=16,
    num_layers=4
)

# Baseline Transformer
model = create_model(
    architecture="transformer",
    d_model=512,
    num_layers=6,  # Same number for encoder and decoder
    max_seq_len=2048
)
```

#### Research Configurations
For detailed experimental studies:
```python
# H3 with custom settings for ablation study
model = create_model(
    architecture="h3",
    ssm_layer="s4d",
    d_model=512,
    d_state=64,
    num_layers=6,
    layer_kwargs={
        "init_method": "legs",  # Test different initialization
        "bidirectional": True,  # Study bidirectional effects
    },
    architecture_kwargs={
        "activation": "gelu",
        "dropout": 0.1,
    }
)

# Transformer with modified attention
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

### Component-Level Research

#### SSM Variants
Experiment with specific SSM configurations:
```python
from state_space_model_universal import S4Layer, MambaLayer

# S4 configuration study
s4 = S4Layer(
    d_model=256,
    d_state=64,
    bidirectional=True,
    dropout=0.1,
    init_method="legs"  # Test different HiPPO variants
)
output = s4(input_tensor)

# Mamba parameter exploration
mamba = MambaLayer(
    d_model=256,
    d_state=16,
    expand_factor=2,  # Study expansion effects
    init_method="inv"
)
output = mamba(input_tensor)
```

#### Transformer Analysis
Study Transformer components:
```python
from state_space_model_universal import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerModel
)

# Encoder attention analysis
encoder_layer = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    attention_dropout=0.1
)
encoded = encoder_layer(x, mask=None)

# Cross-attention study
decoder_layer = TransformerDecoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)
decoded = decoder_layer(x, memory, tgt_mask=None, memory_mask=None)

# Full architecture experiments
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

# Masking pattern analysis
src_mask = torch.ones(batch_size, src_len, src_len)
tgt_mask = torch.triu(torch.ones(batch_size, tgt_len, tgt_len), diagonal=1)
output = transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
```

### Experimental Setup

Example training configuration:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Setup experimental model
model = create_model("h3", "s4", d_model=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training procedure
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch.input)
        loss = criterion(output, batch.target)
        loss.backward()
        optimizer.step()

# Evaluation protocol
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        output = model(batch.input)
        # Compute metrics
```

---

## Research Components

### State Space Models
Each variant represents different research directions:

- **S4**: 
  - Investigation of structured matrices
  - Study of HiPPO initialization effects
  - Analysis of complex-valued representations

- **S4D**: 
  - Exploration of diagonal approximations
  - Comparison of initialization methods
  - Real-valued computation efficiency

- **S5**: 
  - Study of architectural simplifications
  - Parameter efficiency analysis
  - Stability investigations

- **Mamba**: 
  - Research on selective mechanisms
  - Analysis of sparsity patterns
  - Efficiency comparisons

### Transformer Studies
Our Transformer implementation enables research on:

#### Encoder Analysis
- **Self-attention**: Pattern analysis
- **Position-wise FFN**: Architecture studies
- **Layer Norm**: Stability research
- **Residual Connections**: Gradient flow analysis

#### Decoder Investigation
- **Masked Self-attention**: Causality studies
- **Cross-attention**: Information flow analysis
- **Position-wise FFN**: Component interaction research
- **Layer Norm & Residuals**: Training dynamics

### Hybrid Architecture Research
Experimental combinations under study:

- **H3**: 
  - SSM-convolution interactions
  - Gating mechanism analysis
  - Component synergy studies

- **Gated MLP**: 
  - SSM integration effects
  - Gating pattern analysis
  - Architecture flexibility studies

---

## Project Organization
```
models/
├── architectures/       # Architecture implementations
│   ├── h3.py           # H3 experiments
│   ├── gated_mlp.py    # Gated MLP studies
│   ├── mamba.py        # Mamba research
│   └── transformer.py   # Transformer analysis
├── layers/             # Core components
│   ├── base.py         # Base classes
│   ├── s4_layer.py     # S4 implementation
│   ├── s4d_layer.py    # S4D variant
│   ├── s5_layer.py     # S5 version
│   ├── mamba_layer.py  # Mamba implementation
│   ├── transformer_encoder_layer.py
│   └── transformer_decoder_layer.py
└── utils/              # Research utilities
    ├── attention.py    # Attention mechanisms
    ├── mlp.py          # MLP components
    ├── conv.py         # Convolution studies
    └── hippo.py        # HiPPO initialization
```

---

## References

### State Space Models
1. [S4: Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2111.00396)
   - Foundational SSM architecture
   - HiPPO initialization theory

2. [S4D: Linear-Time State Space Models with Diagonal State](https://arxiv.org/abs/2206.11893)
   - Diagonal approximation study
   - Training dynamics analysis

3. [S5: Simple State Space for Sequence Modeling](https://arxiv.org/abs/2208.04933)
   - Architecture simplification
   - Efficiency investigation

4. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
   - Selective mechanism research
   - Sparsity pattern analysis

### Transformer Research
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Attention mechanism foundations
   - Architecture principles

2. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
   - Context extension studies
   - Position representation research

---

## Contributing

Contributions to this research are welcome:

- 🐛 Report experimental issues
- 💡 Suggest new research directions
- 📝 Improve documentation
- 🔧 Add new model variants

For significant changes, please open an issue first to discuss the proposed research direction.

---

## License

This research project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by the State Space Model Universal research team
</div>
