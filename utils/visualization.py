from typing import List, Optional
import matplotlib.pyplot as plt
import torch
import wandb

def plot_sequences(
    input_seq: torch.Tensor,
    target_seq: torch.Tensor,
    pred_seq: torch.Tensor,
    sample_idx: int = 0,
    channel_idx: int = 0,
    title: Optional[str] = None
) -> None:
    """Plot input, target, and predicted sequences.
    
    Args:
        input_seq: Input sequence tensor (batch_size, seq_len, channels)
        target_seq: Target sequence tensor
        pred_seq: Predicted sequence tensor
        sample_idx: Index of sample to plot
        channel_idx: Index of channel to plot
        title: Optional plot title
    """
    plt.figure(figsize=(12, 4))
    
    # Plot sequences
    t = torch.arange(input_seq.shape[1])
    plt.plot(t, input_seq[sample_idx, :, channel_idx].cpu(), 
             label='Input', alpha=0.7)
    plt.plot(t, target_seq[sample_idx, :, channel_idx].cpu(), 
             label='Target', alpha=0.7)
    plt.plot(t, pred_seq[sample_idx, :, channel_idx].detach().cpu(), 
             label='Prediction', alpha=0.7)
    
    plt.legend()
    plt.grid(True)
    if title:
        plt.title(title)
    
    # Log to wandb
    wandb.log({"sequence_plot": wandb.Image(plt)})
    plt.close() 