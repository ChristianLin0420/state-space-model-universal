from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

class Trainer:
    """Trainer class for state space models."""
    
    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any]
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            cfg: Training configuration
        """
        self.model = model
        self.cfg = cfg
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_epochs,
            eta_min=1e-6
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            Dictionary of metrics
        """
        x, y = batch["input"].to(self.device), batch["target"].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output, _ = self.model(x)
        loss = self.criterion(output, y)
        
        # Backward pass
        loss.backward()
        if self.cfg.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.gradient_clip_val
            )
        self.optimizer.step()
        
        return {"loss": loss.item()}
        
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single validation step.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            Dictionary of metrics
        """
        x, y = batch["input"].to(self.device), batch["target"].to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(x)
            loss = self.criterion(output, y)
            
        return {"val_loss": loss.item()}
        
    def fit(self, train_loader, val_loader):
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        for epoch in range(self.cfg.max_epochs):
            # Training
            self.model.train()
            train_metrics = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
                
            # Validation
            self.model.eval()
            val_metrics = []
            
            for batch in val_loader:
                metrics = self.validation_step(batch)
                val_metrics.append(metrics)
                
            # Log metrics
            metrics = {
                "epoch": epoch,
                "train_loss": sum(m["loss"] for m in train_metrics) / len(train_metrics),
                "val_loss": sum(m["val_loss"] for m in val_metrics) / len(val_metrics)
            }
            wandb.log(metrics)
            
            # Update scheduler
            self.scheduler.step() 