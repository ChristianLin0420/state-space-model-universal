import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import wandb

from training.trainer import Trainer
from models import get_model
from data.dataset import get_dataset
from utils.logging_utils import setup_logging

@hydra.main(config_path="../config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(cfg)
    
    # Initialize wandb
    wandb.init(
        project=cfg.logging.wandb_project,
        config=cfg
    )
    
    # Create model
    model = get_model(cfg.model)
    
    # Setup data
    train_dataset, val_dataset = get_dataset(cfg.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        cfg=cfg.training
    )
    
    # Train model
    trainer.fit(train_loader, val_loader)
    
if __name__ == "__main__":
    main() 