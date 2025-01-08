import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import wandb
from omegaconf import DictConfig

def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration.
    
    Args:
        cfg: Configuration object
    """
    # Create save directory
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        save_dir: Directory to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    save_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, save_path)
    logging.info(f'Saved checkpoint to {save_path}')

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (model, optimizer, checkpoint_dict)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model, optimizer, checkpoint

def log_metrics(metrics: Dict[str, float], step: int) -> None:
    """Log metrics to wandb and console.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step/epoch
    """
    # Log to wandb
    wandb.log(metrics, step=step)
    
    # Log to console
    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logging.info(f'Step {step} | {metrics_str}') 