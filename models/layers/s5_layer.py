from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel
from ..utils.hippo import hippo_initializer

class S5Layer(StateSpaceModel):
    """Simplified State Space Sequence (S5) model implementation.
    
    Direct PyTorch port of the JAX implementation from the paper.
    Paper: https://arxiv.org/abs/2208.04933
    
    Args:
        d_model: Input/output dimension
        d_state: State dimension
        dropout: Dropout rate
        dt_min: Minimum step size for discretization
        dt_max: Maximum step size for discretization
        init_scale: Initialization scale for SSM parameters
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        
        # Initialize parameters following JAX implementation
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize step sizes (Δ)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Initialize SSM parameters
        Lambda, B = hippo_initializer(d_state)  # Get HiPPO initialization
        self.Lambda = nn.Parameter(init_scale * Lambda.unsqueeze(0).repeat(d_model, 1))
        self.B = nn.Parameter(init_scale * B.unsqueeze(0).repeat(d_model, 1))
        self.C = nn.Parameter(init_scale * torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.zeros(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
    def discretize(
        self,
        Lambda: torch.Tensor,
        B_tilde: torch.Tensor,
        Delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize a diagonalized, continuous-time linear SSM.
        
        Direct port of the discretize function from JAX implementation.
        
        Args:
            Lambda: Diagonal state matrix (P,)
            B_tilde: Input matrix (P, H)
            Delta: Discretization step sizes (P,)
            
        Returns:
            Tuple of discretized (Lambda_bar, B_bar)
        """
        # Identity = np.ones(Lambda.shape[0])
        Identity = torch.ones_like(Lambda)
        
        # Lambda_bar = np.exp(Lambda * Delta)
        Lambda_bar = torch.exp(Lambda * Delta.unsqueeze(-1))
        
        # B_bar = (1 / Lambda_bar - Identity) * B_tilde
        B_bar = ((1 / Lambda) - Identity) * B_tilde
        
        return Lambda_bar, B_bar
        
    def binary_operator(
        self,
        element_i: Tuple[torch.Tensor, torch.Tensor],
        element_j: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binary operator for parallel scan of linear recurrence.
        
        Direct port of binary_operator from JAX implementation.
        
        Args:
            element_i: Tuple containing A_i and Bu_i at position i
            element_j: Tuple containing A_j and Bu_j at position j
            
        Returns:
            New element (A_out, Bu_out)
        """
        A_i, Bu_i = element_i
        A_j, Bu_j = element_j
        
        return A_j * A_i, A_j * Bu_i + Bu_j
        
    def apply_ssm(
        self,
        Lambda_bar: torch.Tensor,  # (P,)
        B_bar: torch.Tensor,       # (P, H)
        C_tilde: torch.Tensor,     # (H, P)
        D: torch.Tensor,           # (H,)
        input_sequence: torch.Tensor  # (L, H)
    ) -> torch.Tensor:
        """Compute the LxH output of discretized SSM given an LxH input.
        
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix     (P,)
            B_bar (complex64): discretized input matrix                  (P, H)
            C_tilde (complex64): output matrix                          (H, P)
            D (float32): feedthrough matrix                             (H,)
            input_sequence (float32): input sequence of features        (L, H)
            
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)     (L, H)
        """
        # Prepare elements required to initialize parallel scan
        Lambda_elements = torch.repeat_interleave(
            Lambda_bar[None, ...],  # Add sequence length dimension
            input_sequence.shape[0],
            dim=0
        )  # Shape: (L, P)
        
        # Compute Bu_elements using vmap for parallel computation
        Bu_elements = B_bar.unsqueeze(0) * input_sequence.unsqueeze(-1)  # Shape: (L, P)
        
        # Pack elements for parallel scan
        elements = (Lambda_elements, Bu_elements)  # (L, P), (L, P)
        
        # Compute latent state sequence given input sequence using parallel scan
        _, xs = self._parallel_scan(self.binary_operator, elements)  # Shape: (L, P)
        
        # Compute SSM output sequence
        ys = torch.einsum('lp,hp->lh', xs, C_tilde) + D * input_sequence
        
        return ys
        
    def _parallel_scan(
        self,
        binary_op,
        elements: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel scan implementation for PyTorch.
        
        Args:
            binary_op: Binary operator function for the scan
            elements: Tuple of (Lambda_elements, Bu_elements)
            
        Returns:
            Tuple of final state and all intermediate states
        """
        Lambda_elements, Bu_elements = elements
        L = Lambda_elements.shape[0]
        
        # Tree reduction phase (up-sweep)
        def reduction_step(size: int):
            for i in range(0, L-size, size*2):
                j = i + size
                # Apply binary operator to pairs
                Lambda_elements[i], Bu_elements[i] = binary_op(
                    (Lambda_elements[i], Bu_elements[i]),
                    (Lambda_elements[j], Bu_elements[j])
                )
        
        # Tree distribution phase (down-sweep)
        def distribution_step(size: int):
            for i in range(0, L-size, size*2):
                j = i + size
                # Save old values
                Lambda_old = Lambda_elements[i].clone()
                Bu_old = Bu_elements[i].clone()
                # Update pairs
                Lambda_elements[j], Bu_elements[j] = binary_op(
                    (Lambda_old, Bu_old),
                    (Lambda_elements[j], Bu_elements[j])
                )
        
        # Up-sweep: Reduction phase
        step = 1
        while step < L:
            reduction_step(step)
            step *= 2
        
        # Down-sweep: Distribution phase
        step = L // 2
        while step > 0:
            distribution_step(step)
            step //= 2
        
        return Lambda_elements, Bu_elements
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of S5 layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            state: Not used in S5 implementation
            
        Returns:
            tuple: (output, None)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get discretization step sizes
        Delta = torch.exp(self.log_dt)
        
        # Discretize SSM
        Lambda_bar, B_bar = self.discretize(self.Lambda, self.B, Delta)
        
        # Process each sequence in batch
        outputs = []
        for b in range(batch_size):
            y = self.apply_ssm(
                Lambda_bar,
                B_bar,
                self.C,
                self.D,
                x[b].transpose(0, 1)  # (d_model, seq_len)
            )
            outputs.append(y.transpose(0, 1))  # Back to (seq_len, d_model)
            
        output = torch.stack(outputs, dim=0)
        output = self.dropout(output)
        
        return output, None
        
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Note: S5 doesn't use explicit state tracking, kept for interface consistency.
        """
        return torch.zeros(batch_size, self.d_state, device=self.Lambda.device) 