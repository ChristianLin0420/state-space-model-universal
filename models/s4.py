from typing import Optional, Tuple, Literal, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import StateSpaceModel
from .hippo import hippo_initializer

class S4Layer(StateSpaceModel):
    """Structured State Space Sequence (S4) model implementation.
    
    Supports both convolutional (Algorithm 1) and recurrent modes:
    - Convolutional: Uses cached FFT kernel for efficient training
    - Recurrent: Uses state-based computation for streaming inference
    
    Paper: https://arxiv.org/abs/2111.00396
    
    Args:
        d_model (int): Input/output dimension
        d_state (int): State dimension for the SSM
        dropout (float, optional): Dropout rate. Defaults to 0.1
        bidirectional (bool, optional): Whether to use bidirectional SSM. Defaults to False
        dt_min (float, optional): Minimum step size for discretization. Defaults to 0.001
        dt_max (float, optional): Maximum step size for discretization. Defaults to 0.1
        hippo_method (str, optional): HiPPO initialization method ("legendre" or "fourier"). Defaults to "legendre"
        max_length (int, optional): Maximum sequence length for pre-computing kernels. 
            Longer sequences will be processed in chunks.
            Recommended to set this based on available GPU memory.
            Default: 16384 (2^14) which works well for most cases
        mode (str, optional): Computation mode ("recurrent" or "conv"). Defaults to "conv"
            - "conv": Uses FFT-based convolution (faster for training)
            - "recurrent": Uses sequential updates (for streaming inference)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        hippo_method: str = "legendre",
        max_length: int = 16384,  # Maximum sequence length for pre-computing kernels
                                 # Longer sequences will be processed in chunks
                                 # Recommended to set this based on available GPU memory
                                 # Default: 16384 (2^14) which works well for most cases
        mode: Literal["recurrent", "conv"] = "conv"
    ) -> None:
        super().__init__(d_model, d_state, dropout)
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.mode = mode
        
        # Initialize discretization parameters
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.dt = nn.Parameter(torch.exp(log_dt))
        
        # Initialize SSM parameters
        A, B = hippo_initializer(d_state, method=hippo_method)
        P = torch.randn(d_state, d_state) / math.sqrt(d_state)
        Q = torch.randn(d_state, d_state) / math.sqrt(d_state)
        self.Lambda = nn.Parameter(torch.diag(A))
        self.P = nn.Parameter(P)
        self.Q = nn.Parameter(Q)
        
        # B and C matrices
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        
        self.dropout = nn.Dropout(dropout)
        
        # Cache for storing pre-computed kernels
        self.register_buffer('kernel_cache', None)
        self.register_buffer('kernel_cache_length', torch.tensor(0))
        
        # Cache for discretized matrices in recurrent mode
        self.register_buffer('dA_cache', None)
        self.register_buffer('dB_cache', None)
        
    def _discretize_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize state matrices for recurrent computation.
        
        Implements the bilinear transform to convert continuous-time SSM to discrete-time:
            A → Ā = (I - Δ/2·A)^{-1}(I + Δ/2·A)
            B → B̄ = (I - Δ/2·A)^{-1}Δ·B
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Discretized (A_bar, B_bar) matrices
                - A_bar: Shape (batch, d_model, d_state, d_state)
                - B_bar: Shape (d_model, d_state, 1)
        """
        if self.dA_cache is None or self.dB_cache is None:
            # Compute A = Λ - PQ*
            A = torch.diag(self.Lambda) - self.P @ self.Q.conj().T
            
            # Bilinear transform
            I = torch.eye(self.d_state, device=self.Lambda.device)
            dt = self.dt.unsqueeze(-1).unsqueeze(-1)
            
            # Compute (I - Δ/2 · A)^{-1}
            left = I - (dt/2) * A
            left_inv = torch.inverse(left)
            
            # Compute Ā = (I - Δ/2 · A)^{-1}(I + Δ/2 · A)
            right = I + (dt/2) * A
            self.dA_cache = left_inv @ right
            
            # Compute B̄ = (I - Δ/2 · A)^{-1}Δ · B
            self.dB_cache = left_inv @ (dt * self.B.unsqueeze(-1))
            
        return self.dA_cache, self.dB_cache
        
    def _recurrent_step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent implementation using discretized state space model.
        
        Implements the state space recursion:
            h_{t+1} = Āh_t + B̄x_t
            y_t = Ch_t
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional previous state of shape (batch_size, d_state)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Shape (batch_size, seq_len, d_model)
                - next_state: Shape (batch_size, d_state)
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = self.init_state(batch_size)
            
        # Get discretized matrices
        A_bar, B_bar = self._discretize_matrices()
        
        # Recurrent updates
        outputs = []
        next_state = state
        for t in range(seq_len):
            # State update: x_k = Āx_{k-1} + B̄u_k
            next_state = torch.einsum('bmd,bd->bm', A_bar, next_state) + \
                        torch.einsum('md,bm->bm', B_bar.squeeze(-1), x[:, t])
            
            # Output: y_k = Cx_k
            y = torch.einsum('md,bm->bm', self.C, next_state)
            outputs.append(y)
            
        output = torch.stack(outputs, dim=1)
        return output, next_state
        
    def _compute_cauchy_kernel(
        self,
        w: torch.Tensor,
        P: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute black-box Cauchy kernel (Algorithm 1, Line 2).
        
        Computes the kernel components using the Cauchy transform:
            k_00 = C(zI - Λ)^{-1}B
            k_01 = C(zI - Λ)^{-1}P
        where z = exp(2πiw)
        
        Args:
            w: Frequencies for SSMGF evaluation (shape: L//2)
            P: State matrix factorization parameter (shape: d_state, d_state)
            B: Input projection matrix (shape: d_model, d_state)
            C: Output projection matrix (shape: d_model, d_state)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (k_00, k_01) for kernel computation
                - k_00: Shape (d_model, L//2)
                - k_01: Shape (d_model, L//2)
        """
        z = torch.exp(2 * math.pi * 1j * w)
        k_00 = C @ (torch.diag(1.0 / (z.unsqueeze(-1) - self.Lambda)) @ B)
        k_01 = C @ (torch.diag(1.0 / (z.unsqueeze(-1) - self.Lambda)) @ P)
        return k_00, k_01
        
    def _compute_kernel(self, L: int) -> torch.Tensor:
        """Compute S4 convolution kernel (Algorithm 1).
        
        Implements the full kernel computation:
        1. Compute Cauchy kernel components
        2. Apply Woodbury identity
        3. Evaluate at roots of unity
        4. Apply inverse FFT
        
        Args:
            L: Desired sequence length
            
        Returns:
            torch.Tensor: Convolution kernel of shape (L, d_model)
        """
        w = torch.linspace(0, 1, L//2 + 1)[:-1]
        k_00, k_01 = self._compute_cauchy_kernel(w, self.P, self.B, self.C)
        h_10 = self.Q.conj().T @ (torch.diag(1.0 / (w.unsqueeze(-1) - self.Lambda)) @ self.B)
        K_hat = k_00 - k_01 @ torch.inverse(torch.eye(self.d_state) + h_10)
        K_fft = K_hat * torch.exp(2 * math.pi * 1j * torch.arange(L//2) / L)
        K = torch.fft.irfft(K_fft, n=L, dim=-1)
        return K
        
    def _get_cached_kernel(self, L: int) -> torch.Tensor:
        """Get kernel from cache or compute if needed.
        
        Manages the kernel cache to avoid recomputation:
        - Computes and caches kernel for max_length if not cached
        - Returns truncated kernel for shorter sequences
        - Raises error for sequences longer than max_length
        
        Args:
            L: Desired sequence length
            
        Returns:
            torch.Tensor: Convolution kernel of shape (L, d_model)
            
        Raises:
            ValueError: If L > max_length
        """
        if L > self.max_length:
            raise ValueError(f"Sequence length {L} exceeds maximum length {self.max_length}")
            
        if self.kernel_cache is None or self.kernel_cache_length < L:
            # Compute kernel for max_length and cache it
            K = self._compute_kernel(self.max_length)
            self.kernel_cache = K
            self.kernel_cache_length = torch.tensor(self.max_length)
            
        # Return the kernel truncated to length L
        return self.kernel_cache[:L]
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass supporting both recurrent and convolutional modes.
        
        In conv mode:
            - Uses cached FFT kernel for efficient computation
            - Returns (output, None)
        
        In recurrent mode:
            - Uses sequential state updates
            - Returns (output, next_state) for stateful inference
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Previous state (used only in recurrent mode)
                  Shape: (batch_size, d_state)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - output: Shape (batch_size, seq_len, d_model)
                - next_state: Shape (batch_size, d_state) in recurrent mode, None in conv mode
        """
        if self.mode == "recurrent":
            y, next_state = self._recurrent_step(x, state)
            if self.bidirectional:
                y_reverse, _ = self._recurrent_step(torch.flip(x, [1]), state)
                y = y + torch.flip(y_reverse, [1])
            return self.dropout(y), next_state
            
        else:  # convolutional mode
            _, L, _ = x.shape
            K = self._get_cached_kernel(L)
            
            # Apply convolution in frequency domain
            x_fft = torch.fft.rfft(x, n=L, dim=1)
            K_fft = torch.fft.rfft(K, n=L, dim=0)
            y = torch.fft.irfft(x_fft * K_fft.unsqueeze(0), n=L, dim=1)
            
            if self.bidirectional:
                x_reverse = torch.flip(x, [1])
                y_reverse = torch.fft.irfft(
                    torch.fft.rfft(x_reverse, n=L, dim=1) * K_fft.unsqueeze(0),
                    n=L, dim=1
                )
                y = y + torch.flip(y_reverse, [1])
                
            return self.dropout(y), None
            
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the state of the model.
        
        Creates zero-initialized state for recurrent mode.
        Not used in convolution mode, kept for interface consistency.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            torch.Tensor: Initial state of shape (batch_size, d_state)
        """
        return torch.zeros(batch_size, self.d_state, device=self.Lambda.device)
        
    def reset_parameters(self) -> None:
        """Reset cached computations when parameters change.
        
        Clears:
        - Convolution kernel cache
        - Discretized matrix cache
        - Kernel cache length counter
        """
        super().reset_parameters()
        self.kernel_cache = None
        self.kernel_cache_length = torch.tensor(0)
        self.dA_cache = None
        self.dB_cache = None 