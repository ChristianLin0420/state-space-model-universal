# State Space Model Universal

A research project implementing state-of-the-art sequence modeling architectures, focusing on State Space Models (SSMs) and their variants.

## Overview

This project implements various state-of-the-art sequence modeling architectures, with a particular focus on State Space Models (SSMs). The implementations include:

### SSM Layers
- **S4**: Structured State Space Sequence Model
- **S4D**: Diagonal State Space Model
- **S5**: Simplified State Space Model
- **Mamba2**: State Space Duality Model with two variants:
  - Sequential: SSM parameters are produced as a function of the input
  - Parallel: SSM parameters are produced at the beginning of the block

### Architectures
- **H3**: SSM with convolution and gating mechanism
- **Gated MLP**: MLP with optional SSM and parallel gating
- **Mamba2**: Two block variants for flexible sequence modeling:
  1. Sequential Mamba Block:
     - SSM parameters (A, B, C) are produced as a function of the SSM input X
     - Uses sequential linear projections
     - Includes convolution and gating mechanism
  2. Parallel Mamba Block:
     - SSM parameters (A, B, C) are produced at the beginning of the block
     - Includes normalization layer before SSM
     - Shares parameters across heads (MVA-style)

## Installation

```bash
git clone https://github.com/yourusername/state-space-model-universal.git
cd state-space-model-universal
pip install -e .
```

## Usage

### Basic Usage

```python
import state_space_model as ssm

# Create a Sequential Mamba2 model
model = ssm.create_model(
    architecture='mamba2',
    block_type='sequential',  # or 'parallel'
    d_model=256,
    d_state=16,
    n_layer=4
)

# Create an H3 model
model = ssm.create_model(
    architecture='h3',
    d_model=256,
    d_state=64,
    n_layer=4
)
```

### Advanced Configuration

```python
# Parallel Mamba2 with custom settings
model = ssm.create_model(
    architecture='mamba2',
    block_type='parallel',
    d_model=512,
    d_state=32,
    n_layer=6,
    d_conv=8,
    expand_factor=4,
    conv_kernel_size=7,
    dropout=0.1
)
```

## Project Structure

```
state-space-model-universal/
├── models/
│   ├── architectures/
│   │   ├── h3.py
│   │   ├── gated_mlp.py
│   │   └── mamba2.py
│   ├── layers/
│   │   ├── base.py
│   │   ├── s4_layer.py
│   │   ├── s4d_layer.py
│   │   ├── s5_layer.py
│   │   └── mamba2_layer.py
│   └── utils/
│       ├── mlp.py
│       └── conv.py
├── setup.py
└── requirements.txt
```

## References

1. S4: Structured State Space Sequence Model
   - Paper: [Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2111.00396)

2. S4D: Diagonal State Space Model
   - Paper: [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)

3. S5: Simplified State Space Model
   - Paper: [Simple State Space Models](https://arxiv.org/abs/2303.11245)

4. Mamba2: State Space Duality
   - Paper: [Mamba2: State Space Model with State Space Duality](https://arxiv.org/abs/2402.xxxxx)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
