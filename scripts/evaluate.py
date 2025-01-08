import hydra
from omegaconf import DictConfig
import torch
import wandb
import logging
from pathlib import Path

from models import get_model
from data.dataset import get_dataset
from evaluation.metrics import compute_prediction_error
from utils.logging_utils import setup_logging, load_checkpoint

@hydra.main(config_path="../config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(cfg)
    
    # Initialize wandb in evaluation mode
    wandb.init(
        project=cfg.logging.wandb_project,
        config=cfg,
        mode="online" if cfg.logging.log_to_wandb else "disabled"
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = get_model(cfg.model)
    
    # Load checkpoint
    checkpoint_path = Path(cfg.evaluation.checkpoint_path)
    model, _, checkpoint = load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Get test dataset
    test_dataset = get_dataset(cfg.data, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    # Compute metrics
    metrics = compute_prediction_error(
        model,
        test_loader,
        device,
        max_samples=cfg.evaluation.max_samples
    )
    
    # Log metrics
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.4f}")
        wandb.log({f"test_{name}": value})
    
    # Save metrics to file
    metrics_path = Path(cfg.logging.save_dir) / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    
if __name__ == "__main__":
    main() 