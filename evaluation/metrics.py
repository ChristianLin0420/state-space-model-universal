from typing import Dict, Optional
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SequenceMetrics:
    """Metrics calculator for sequence models."""
    
    @staticmethod
    def compute_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute various metrics for sequence prediction.
        
        Args:
            pred: Predicted sequences (batch_size, seq_len, dim)
            target: Target sequences (batch_size, seq_len, dim)
            mask: Optional mask for valid positions (batch_size, seq_len)
            
        Returns:
            Dictionary of computed metrics
        """
        # Move tensors to CPU and convert to numpy
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if mask is not None:
            mask = mask.cpu().numpy()
            
        # Initialize metrics dictionary
        metrics = {}
        
        # Mean Squared Error
        if mask is not None:
            mse = mean_squared_error(
                target[mask],
                pred[mask]
            )
        else:
            mse = mean_squared_error(target.reshape(-1), pred.reshape(-1))
        metrics['mse'] = float(mse)
        
        # Root Mean Squared Error
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        if mask is not None:
            mae = mean_absolute_error(
                target[mask],
                pred[mask]
            )
        else:
            mae = mean_absolute_error(target.reshape(-1), pred.reshape(-1))
        metrics['mae'] = float(mae)
        
        # Normalized MSE
        target_var = np.var(target) + 1e-8
        metrics['nmse'] = float(mse / target_var)
        
        return metrics

def compute_prediction_error(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Compute prediction error metrics for a model.
    
    Args:
        model: The sequence model to evaluate
        dataloader: DataLoader containing test data
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary of error metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break
                
            x = batch['input'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            output, _ = model(x)
            
            all_preds.append(output)
            all_targets.append(target)
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = SequenceMetrics.compute_metrics(predictions, targets)
    
    return metrics 