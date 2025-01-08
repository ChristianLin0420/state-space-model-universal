# State Space Models in PyTorch

This repository provides implementations of state-of-the-art state space models, including:
- **S4** (Structured State Space Sequence Model)
- **S5** (Simplified State Space Model)
- **Mamba**
- **Mamba2**

All models are implemented with HiPPO initialization for better performance.

## Features
- Modular implementation of state space models.
- Support for both **Legendre** and **Fourier HiPPO** initialization.
- Configurable model architectures through YAML configs.
- Training and evaluation pipelines with logging support.
- Integration with **Weights & Biases (Wandb)** for experiment tracking.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/state-space-model-universal.git
   cd state-space-model-universal
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   Option 1: Using `setup.py`
   ```bash
   pip install -e .
   ```

   Option 2: Using `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```


## Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- Other dependencies are listed in `requirements.txt`.

## Usage

### Training
To train a model with the default configuration:
```bash
python scripts/train.py
```

To modify specific configuration parameters:
```bash
python scripts/train.py model=s4 model.num_layers=6 training.batch_size=64
```

### Evaluation
To evaluate a trained model:
```bash
python scripts/evaluate.py evaluation.checkpoint_path=/path/to/checkpoint.pt
```

## Configuration

The project uses **Hydra** for configuration management. Main configuration files include:
- `config/default_config.yaml`: Default configuration.
- `config/model_configs.yaml`: Model-specific configurations.

You can override parameters directly from the command line during execution.

## Project Structure

```plaintext
state-space-models/
├── config/             # Configuration files
│   ├── default_config.yaml
│   └── model_configs.yaml
├── models/             # Model implementations
│   ├── base.py         # Base state space model class
│   ├── s4.py           # S4 implementation
│   ├── s5.py           # S5 implementation
│   ├── mamba.py        # Mamba implementation
│   └── hippo.py        # HiPPO initialization utilities
├── training/           # Training utilities
├── evaluation/         # Evaluation utilities and metrics
├── utils/              # Helper functions
├── scripts/            # Scripts for training and evaluation
├── requirements.txt    # Dependency requirements
├── setup.py            # Package installation setup
└── README.md           # Project overview and instructions
```

## Citation

If you use this repository in your research, please cite the following paper:

```bibtex
@article{gu2021efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and Re, Christopher},n  journal={arXiv preprint arXiv:2111.00396},
  year={2021}
}
