from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, random_split
import numpy as np

class SequenceDataset(Dataset):
    """Dataset class for sequence data."""
    
    def __init__(
        self,
        sequence_length: int,
        num_samples: int = 1000,
        input_dim: int = 1,
        synthetic: bool = True
    ) -> None:
        """Initialize dataset.
        
        Args:
            sequence_length: Length of sequences
            num_samples: Number of samples to generate
            input_dim: Input dimension
            synthetic: Whether to generate synthetic data
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        if synthetic:
            self.data = self._generate_synthetic_data(num_samples)
        else:
            raise NotImplementedError("Only synthetic data is supported currently")
            
    def _generate_synthetic_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequence data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of input and target tensors
        """
        # Generate random frequencies for sinusoidal signals
        frequencies = torch.rand(num_samples, self.input_dim) * 2 * np.pi
        
        # Generate time steps
        t = torch.linspace(0, 1, self.sequence_length)
        
        # Generate input signals
        inputs = torch.zeros(num_samples, self.sequence_length, self.input_dim)
        targets = torch.zeros(num_samples, self.sequence_length, self.input_dim)
        
        for i in range(num_samples):
            for j in range(self.input_dim):
                # Input signal: sine wave
                inputs[i, :, j] = torch.sin(frequencies[i, j] * t)
                # Target signal: cosine wave (phase shifted)
                targets[i, :, j] = torch.cos(frequencies[i, j] * t)
                
        return inputs, targets
        
    def __len__(self) -> int:
        return len(self.data[0])
        
    def __getitem__(self, idx: int) -> dict:
        return {
            "input": self.data[0][idx],
            "target": self.data[1][idx]
        }

def get_dataset(
    cfg: dict,
    train: bool = True
) -> Tuple[Dataset, Optional[Dataset]]:
    """Get dataset based on configuration.
    
    Args:
        cfg: Dataset configuration
        train: Whether to return training dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset) or single dataset
    """
    dataset = SequenceDataset(
        sequence_length=cfg.sequence_length,
        synthetic=cfg.dataset == "synthetic"
    )
    
    if train:
        # Split dataset into train and validation
        train_size = int(len(dataset) * cfg.train_split)
        val_size = int(len(dataset) * cfg.val_split)
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, _ = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )
        
        return train_dataset, val_dataset
    else:
        return dataset 