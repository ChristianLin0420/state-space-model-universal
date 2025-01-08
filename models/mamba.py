from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .base import StateSpaceModel
from .hippo import hippo_initializer

class MambaLayer(StateSpaceModel):
    """Mamba model implementation following Algorithm 2: SSM + Selection (S6).
    
    Supports both recurrent (sequential) and convolutional (parallel) computation modes.
    The model uses a selective state space mechanism with input-dependent parameters.
    
    Paper: https://arxiv.org/abs/2312.00752
    
    Args:
        d_model: Input/output dimension
        d_state: State dimension for the SSM
        dropout: Dropout rate
        expansion_factor: Factor for expanding inner dimension (default: 2)
        conv_kernel: Kernel size for local convolution (default: 4)
        mode: Computation mode, either "recurrent" or "conv" (default: "conv")
            - "recurrent": Sequential computation (better for training)
            - "conv": Parallel computation using FFT (faster for inference)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        expansion_factor: int = 2,
        conv_kernel: int = 4,
        mode: Literal["recurrent", "conv"] = "conv",
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        
        self.d_inner = expansion_factor * d_model
        self.conv_kernel = conv_kernel
        self.mode = mode
        
        # Initialize A matrix (Parameter in Algorithm 2)
        A = hippo_initializer(d_state, method="legendre")
        self.A = nn.Parameter(A.unsqueeze(0).repeat(self.d_inner, 1, 1))
        
        # Input-dependent parameter networks (sB, sC, sΔ in Algorithm 2)
        self.s_B = nn.Linear(d_model, self.d_inner * d_state)
        self.s_C = nn.Linear(d_model, self.d_inner * d_state)
        self.s_Delta = nn.Sequential(
            nn.Linear(d_model, self.d_inner),
            nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=conv_kernel,
                padding=conv_kernel - 1,
                groups=self.d_inner
            ),
            nn.Linear(self.d_inner, self.d_inner)
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def discretize(
        self,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize the continuous-time SSM (Algorithm 2, Line 5).
        
        Args:
            delta: Time step tensor of shape (batch_size, seq_len, d_inner)
            A: State matrix of shape (d_inner, d_state, d_state)
            B: Input matrix of shape (batch_size, seq_len, d_inner, d_state)
            
        Returns:
            Tuple of:
                - dA: Discretized state matrix (batch_size, seq_len, d_inner, d_state, d_state)
                - dB: Discretized input matrix (batch_size, seq_len, d_inner, d_state)
        """
        # Discretize A: Ā = exp(Δ ⊗ A)
        dA = torch.matrix_exp(A.unsqueeze(0) * delta.unsqueeze(-1).unsqueeze(-1))
        
        # Discretize B: B̄ = Δ ⊗ B
        dB = delta.unsqueeze(-1) * B
        
        return dA, dB
        
    def ssm_recurrent(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Sequential (recurrent) implementation of SSM.
        
        Implements the state space model using sequential updates:
            h_t = Āh_{t-1} + B̄x_t
            y_t = Ch_t
            
        Args:
            A_bar: Discretized state matrix (batch_size, seq_len, d_inner, d_state, d_state)
            B_bar: Discretized input matrix (batch_size, seq_len, d_inner, d_state)
            C: Output matrix (batch_size, seq_len, d_inner, d_state)
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_inner)
        """
        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        # Sequential scan
        for t in range(seq_len):
            # Update state: h_t = Āh_{t-1} + B̄x_t
            state = torch.bmm(A_bar[:, t], state.unsqueeze(-1)).squeeze(-1) + \
                   B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Compute output: y_t = Ch_t
            y = torch.einsum('bd,md->bm', state, C)
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)
        
    def ssm_conv(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Parallel (convolutional) implementation of SSM.
        
        Implements the state space model using frequency domain computation:
            Y(s) = H(s)X(s), where H(s) = C(sI - A)^{-1}B
            
        Args:
            A_bar: Discretized state matrix (batch_size, seq_len, d_inner, d_state, d_state)
            B_bar: Discretized input matrix (batch_size, seq_len, d_inner, d_state)
            C: Output matrix (batch_size, seq_len, d_inner, d_state)
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_inner)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute state space convolution in frequency domain
        # Convert to frequency domain
        x_fft = torch.fft.rfft(x, n=seq_len, dim=1)
        
        # Compute frequency response of the SSM
        freqs = torch.fft.rfftfreq(seq_len, d=1.0, device=x.device)
        omega = 2 * math.pi * freqs.view(1, -1, 1)  # [1, seq_len//2 + 1, 1]
        
        # Compute transfer function H(s) = C(sI - A)^{-1}B
        I = torch.eye(self.d_state, device=x.device)
        H = []
        
        for k in range(len(freqs)):
            # Compute (sI - A)^{-1}
            s = 1j * omega[0, k]
            inv = torch.inverse(s * I - A_bar)
            
            # Compute transfer function
            h = torch.einsum('md,dn,bn->bm', C, inv, B_bar)
            H.append(h)
            
        H = torch.stack(H, dim=1)  # [batch, freq, d_inner]
        
        # Apply transfer function in frequency domain
        y_fft = x_fft.unsqueeze(-1) * H
        
        # Convert back to time domain
        y = torch.fft.irfft(y_fft, n=seq_len, dim=1)
        
        return y
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementing Algorithm 2: SSM + Selection.
        
        The forward pass consists of:
        1. Computing input-dependent parameters B(x) and C(x)
        2. Computing time steps Δ(x)
        3. Discretizing the SSM
        4. Applying the SSM using either recurrent or convolutional mode
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Not used in this implementation (kept for interface consistency)
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - None (for interface consistency)
        """
        batch_size, seq_len, _ = x.shape
        
        # Line 2-3: Compute input-dependent B and C
        B = self.s_B(x).view(batch_size, seq_len, self.d_inner, -1)
        C = self.s_C(x).view(batch_size, seq_len, self.d_inner, -1)
        
        # Line 4: Compute Δ
        delta = self.s_Delta(x)
        delta = F.softplus(delta)  # Ensure positivity
        
        # Line 5: Discretize the SSM
        A_bar, B_bar = self.discretize(delta, self.A, B)
        
        # Line 6: Apply SSM (either recurrent or convolutional mode)
        if self.mode == "recurrent" or (self.training and self.mode == "conv"):
            y = self.ssm_recurrent(A_bar, B_bar, C, x)
        else:
            y = self.ssm_conv(A_bar, B_bar, C, x)
        
        # Project output back to model dimension
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y, None
    
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            Zero-initialized state tensor of shape (batch_size, d_state)
        """
        return torch.zeros(batch_size, self.d_state, device=self.A.device) 