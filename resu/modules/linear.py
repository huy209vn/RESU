"""
RESULinear: Drop-in replacement for nn.Linear with RESU support.

Modes:
- DENSE: Standard forward pass (no pruning)
- SPARSE: Forward with mask applied (standard sparse)
- RESU: Forward with effective weights (resurrection phase)

Seamlessly transitions between modes during training cycle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, Literal
from enum import Enum, auto
import math

from ..core.mask import SparseMask
from ..core.resurrection import ResurrectionEmbedding, StorageMode
from ..core.effective import effective_weight, effective_weight_dense
from ..core.selective_simple import SimpleSelector, SimpleSelectionConfig, SelectionStrategy

# Tensor core support for 2:4 structured sparsity
try:
    from torch.sparse import to_sparse_semi_structured
    SEMI_STRUCTURED_AVAILABLE = True
except ImportError:
    SEMI_STRUCTURED_AVAILABLE = False
    to_sparse_semi_structured = None


class _SemiStructuredLinear(torch.autograd.Function):
    """Custom autograd for semi-structured sparse linear with FP32 weights.

    Forward: Convert FP32 → FP16 → sparse → matmul → FP32 (tensor core accelerated)
    Backward: Standard dense gradient computation (gradients flow to FP32 weights)
    """

    @staticmethod
    def forward(ctx, x, weight, bias):  # type: ignore[override]
        # Save for backward
        ctx.save_for_backward(x, weight, bias)

        # Convert to FP16 for tensor cores
        W_fp16 = weight.half()
        x_fp16 = x.half()

        # Convert to semi-structured sparse (guarded import at module level)
        assert to_sparse_semi_structured is not None, "Semi-structured sparse not available"
        W_sparse = to_sparse_semi_structured(W_fp16)

        # Forward with tensor cores
        out = F.linear(x_fp16, W_sparse, bias.half() if bias is not None else None)

        # Return in original dtype
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        x, weight, bias = ctx.saved_tensors

        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. input: grad_output @ weight
            grad_x = grad_output @ weight

        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. weight: grad_output.T @ x
            # Reshape for matmul: grad_output is (..., out_features), x is (..., in_features)
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_weight = grad_output_2d.t() @ x_2d

        if bias is not None and ctx.needs_input_grad[2]:
            # Gradient w.r.t. bias: sum over batch dims
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_x, grad_weight, grad_bias


# Legacy import for backward compatibility (deprecated)
try:
    from ..core.selective import SelectionConfig  # Deprecated, use SimpleSelectionConfig
except ImportError:
    SelectionConfig = SimpleSelectionConfig


class RESUMode(Enum):
    """Operating mode for RESU modules."""
    DENSE = auto()           # No pruning, standard forward
    SPARSE = auto()          # Pruned, mask applied
    RESU = auto()            # Resurrection phase, θ active
    QRESU = auto()           # Quantized RESU (4/8-bit W_A)
    QRESU_SELECTIVE = auto() # Quantized RESU with selective filtering


class RESULinear(nn.Module):
    """Linear layer with RESU (Resurrection of Sparse Units) support.
    
    Drop-in replacement for nn.Linear that supports:
    - Dynamic pruning with mask management
    - RESU resurrection phase
    - Seamless mode transitions
    - Activation capture for Wanda pruning
    
    Example:
        # Create layer
        layer = RESULinear(512, 256)
        
        # Prune to 50% sparsity
        layer.prune_by_magnitude(0.5)
        
        # Enter RESU mode
        layer.enter_resu_mode(epsilon=0.1)
        
        # Forward pass uses effective weights
        out = layer(x)
        
        # After RESU phase, commit and exit
        layer.exit_resu_mode(commit=True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        storage_mode: StorageMode = StorageMode.COMPACT,
        sparse_threshold: float = 0.7,  # Use sparse ops when sparsity > this
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias term
            device: Target device
            dtype: Data type
            storage_mode: How to store resurrection parameters
            sparse_threshold: Minimum sparsity to use torch.sparse operations
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.storage_mode = storage_mode
        self.sparse_threshold = sparse_threshold
        
        # Core parameters
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        # RESU state (not parameters)
        self._mode = RESUMode.DENSE
        self._mask: Optional[SparseMask] = None
        self._resurrection: Optional[ResurrectionEmbedding] = None
        self._selective: Optional[Any] = None  # Deprecated, use simple selection

        # Performance optimization: cached active weights during RESU
        self._W_active_cached: Optional[torch.Tensor] = None

        # QRESU state (quantization)
        self._W_A_quantized: Optional[torch.Tensor] = None
        self._qparams: Optional[tuple] = None
        self._qscheme: Literal["per_channel", "per_tensor"] = "per_channel"
        self._qbits: int = 4  # 4 or 8
        self._theta: Optional[torch.Tensor] = None  # Flat 1D tensor for QRESU
        self._qscale: Optional[torch.Tensor] = None
        self._qzero: Optional[torch.Tensor] = None

        # QRESU-Selective state (EMA tracking for selective filtering)
        self._selective_config: Optional[SelectionConfig] = None
        self._selective_lr: float = 1e-4
        self._ema_m: Optional[torch.Tensor] = None  # Momentum EMA
        self._ema_v: Optional[torch.Tensor] = None  # Magnitude EMA
        self._consistency: Optional[torch.Tensor] = None  # Consistency scores
        self._selective_step_count: int = 0
        self._grad_hook_handle: Optional[Any] = None

        # Activation capture for Wanda
        self._capture_activations = False
        self._last_input: Optional[torch.Tensor] = None
        self._activation_norms: Optional[torch.Tensor] = None

        # Tensor core acceleration for 2:4 structured sparsity
        self._use_tensor_cores = False
        self._structured_n: Optional[int] = None
        self._structured_m: Optional[int] = None
        self._weight_sparse: Optional[torch.Tensor] = None  # Cached sparse representation

        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def mode(self) -> RESUMode:
        """Current operating mode."""
        return self._mode
    
    @property
    def mask(self) -> Optional[SparseMask]:
        """Current sparsity mask."""
        return self._mask
    
    @property
    def sparsity(self) -> float:
        """Current sparsity level (0 if dense)."""
        if self._mask is None:
            return 0.0
        return self._mask.sparsity
    
    @property
    def resurrection(self) -> Optional[ResurrectionEmbedding]:
        """Resurrection embedding (only during RESU mode)."""
        return self._resurrection
    
    @property
    def is_sparse(self) -> bool:
        """Whether layer is currently sparse."""
        return self._mode in (RESUMode.SPARSE, RESUMode.RESU)
    
    @property
    def is_resu_active(self) -> bool:
        """Whether in RESU mode."""
        return self._mode == RESUMode.RESU
    
    @property
    def device(self) -> torch.device:
        return self.weight.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype
    
    # =========================================================================
    # Mask Management
    # =========================================================================
    
    def set_mask(self, mask: Union[torch.Tensor, SparseMask]):
        """Set pruning mask.
        
        Args:
            mask: Binary mask (1=active, 0=pruned) or SparseMask object
        """
        if isinstance(mask, SparseMask):
            self._mask = mask
        else:
            self._mask = SparseMask(mask.to(self.device))
        
        if self._mode == RESUMode.DENSE:
            self._mode = RESUMode.SPARSE
    
    def clear_mask(self):
        """Remove mask and return to dense mode."""
        self._mask = None
        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.DENSE
    
    # =========================================================================
    # Pruning
    # =========================================================================
    
    def prune_by_magnitude(self, sparsity: float):
        """Prune smallest magnitude weights.
        
        Args:
            sparsity: Fraction to prune (0-1)
        """
        mask = SparseMask.from_magnitude(self.weight.data, sparsity)
        self.set_mask(mask)
        
        # Zero out pruned weights
        with torch.no_grad():
            self.weight.data *= self._mask.mask
   
    def apply_mask(self):
        """Apply mask to weights (zero out pruned positions)."""
        if self._mask is not None:
            with torch.no_grad():
                self.weight.data *= self._mask.mask
    
    # =========================================================================
    # RESU Mode Management
    # =========================================================================
    
    def enter_resu_mode(
        self,
        epsilon: float = 0.1,
        use_selective: bool = True,
        selective_config: Optional[SelectionConfig] = None,
        selection_ratio: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        freeze_active: bool = True,
    ):
        """Enter RESU mode - enable resurrection parameter training.

        IN-PLACE STORAGE: θ is stored directly in W's pruned positions.
        This matches the paper's Proposition 2: "no additional memory required".

        MEMORY-OPTIMIZED: Uses only int32 indices, NO dense masks stored.

        Args:
            epsilon: Initialization scale for θ
            use_selective: Use RESU-Selective filtering
            selective_config: Configuration for selective updates (deprecated, use selection_ratio instead)
            selection_ratio: Fraction of pruned parameters to update (0.01 = 1%, 0.2 = 20%, etc.)
            lr: Learning rate for θ
            weight_decay: Weight decay for θ
            freeze_active: If True, freeze active weights (only train θ).
                          If False, train both active weights and θ.
        """
        if self._mask is None:
            raise RuntimeError("Cannot enter RESU mode without mask. Call set_mask() first.")

        if self._mode == RESUMode.RESU:
            return  # Already in RESU mode

        # If indices were cleared (re-entering RESU), rebuild from weight sparsity
        if len(self._mask._indices) == 0:
            # Rebuild SparseMask from current weight pattern (0 = pruned)
            pruned_indices = (self.weight.data == 0).flatten().nonzero(as_tuple=True)[0]
            self._mask = SparseMask(pruned_indices, self.weight.shape, device=self.device)

        # Get dense mask ONCE (triggers cache, then we clear it)
        dense_mask = self._mask.mask.float()  # 1=active, 0=pruned
        n_pruned = self._mask.n_pruned

        # Compute active weight statistics
        active_weights = self.weight.data[dense_mask.bool()]
        active_std = active_weights.std().item() if len(active_weights) > 0 else 1.0

        # CRITICAL: Store θ IN pruned positions of W (in-place)
        pruned_mask = ~dense_mask.bool()
        with torch.no_grad():
            theta_init = torch.randn(n_pruned, device=self.device, dtype=self.dtype)
            theta_init *= epsilon * active_std
            self.weight.data[pruned_mask] = theta_init

        # Keep W trainable
        self.weight.requires_grad_(True)

        # Remove old hook if exists
        if hasattr(self, '_resu_grad_hook_handle'):
            self._resu_grad_hook_handle.remove()
            delattr(self, '_resu_grad_hook_handle')

        # Store config
        self._freeze_active = freeze_active
        self._use_selective = use_selective

        # BUILD GRADIENT MASK: Bool tensor, True = trainable, False = frozen
        # Using bool instead of float32 saves 75% memory (1 byte vs 4 bytes)
        if use_selective:
            ratio = selection_ratio
            if selective_config is not None and hasattr(selective_config, 'k_select_ratio'):
                ratio = selective_config.k_select_ratio

            k_select = max(1, int(n_pruned * ratio))

            # Build selective mask: True at selected pruned positions only
            self._grad_mask = torch.zeros(self.weight.numel(), dtype=torch.bool, device=self.device)
            pruned_indices = self._mask.pruned_indices
            perm = torch.randperm(n_pruned, device=self.device)[:k_select]
            selected_indices = pruned_indices[perm]
            self._grad_mask[selected_indices] = True
            self._grad_mask = self._grad_mask.view(self.weight.shape)

            self._selection_ratio = k_select / n_pruned
            print(f"RESU-Selective: Selected {k_select}/{n_pruned} pruned positions ({self._selection_ratio:.1%})")

        elif freeze_active:
            # Train only pruned positions: grad_mask = True where pruned
            self._grad_mask = ~dense_mask.bool()

        else:
            # Train everything (no mask needed)
            self._grad_mask = None

        # Clear SparseMask caches AND indices to reclaim memory
        # We don't need _mask during RESU mode - _grad_mask contains all info
        # Store essential stats before clearing
        self._n_pruned = self._mask.n_pruned
        self._n_active = self._mask.n_active
        self._mask._mask_cache = None
        self._mask._active_indices_cache = None
        self._mask._pruned_indices_cache = None
        # Clear indices (32 MB savings at 50% sparsity)
        self._mask._indices = torch.tensor([], dtype=torch.int32, device=self.device)

        # Register gradient hook - in-place masked_fill_ to avoid allocation!
        if self._grad_mask is not None:
            def grad_mask_hook(grad):
                """Mask gradients in-place: zero where _grad_mask is False (frozen positions)."""
                return grad.masked_fill_(~self._grad_mask, 0)  # In-place!

            self._resu_grad_hook_handle = self.weight.register_hook(grad_mask_hook)

        # No separate resurrection object - θ lives in W!
        self._resurrection = None
        self._selective = None

        self._mode = RESUMode.RESU

    def enter_resu_mode_structured(
        self,
        n: int = 2,
        m: int = 4,
        dim: int = 1,
        epsilon: float = 0.1,
    ):
        """Enter RESU mode with EXACT N:M structure from the start.

        Unlike regular RESU (trains all pruned positions), this:
        1. Takes current partial N:M mask (≤N per M)
        2. Computes ONLY which positions need θ to reach exactly N per M
        3. Initializes θ in ONLY those fill positions
        4. Result: EXACT N:M structure throughout training!

        MEMORY-OPTIMIZED: Uses only int32 indices, NO dense masks stored.

        Training uses dense ops (tensor cores need sparse-as-parameter, complex).
        After training, call to_sparse_inference() for tensor core inference.

        Args:
            n: Number per group (default 2)
            m: Group size (default 4)
            dim: Dimension for N:M pattern
            epsilon: Initialization scale for θ
        """
        from ..core.structured import compute_nm_fill_positions

        if self._mask is None:
            raise RuntimeError("Cannot enter RESU mode without mask. Call set_mask() first.")

        if self._mode == RESUMode.RESU:
            return

        # Get current mask as float tensor (temporary, not stored)
        current_mask = self._mask.mask.float()

        # Compute which positions need θ to reach exact N:M (returns dense masks)
        theta_mask_dense, final_mask_dense = compute_nm_fill_positions(current_mask, n, m, dim)

        n_theta = int(theta_mask_dense.sum().item())

        # Statistics
        n_active = self._mask.n_active
        n_final = int(final_mask_dense.sum().item())
        total = self.weight.numel()

        print(f"RESU-Structured {n}:{m}:")
        print(f"  Original active: {n_active} ({n_active/total*100:.1f}%)")
        print(f"  θ positions to fill: {n_theta}")
        print(f"  Final active (exact {n}:{m}): {n_final} ({n_final/total*100:.1f}%)")

        # Initialize θ ONLY in fill positions
        active_weights = self.weight.data[current_mask.bool()]
        active_std = active_weights.std().item() if len(active_weights) > 0 else 1.0

        theta_positions = theta_mask_dense.bool()
        with torch.no_grad():
            theta_init = torch.randn(n_theta, device=self.device, dtype=self.dtype)
            theta_init *= epsilon * active_std
            self.weight.data[theta_positions] = theta_init

        # Update mask to exact N:M
        pruned_indices = (~final_mask_dense.bool()).flatten().nonzero(as_tuple=True)[0]
        self._mask = SparseMask(pruned_indices, final_mask_dense.shape, device=self.device)

        # Store gradient mask as BOOL: True at θ positions (trainable), False elsewhere (frozen)
        # Bool saves 75% memory vs float32 (1 byte vs 4 bytes)
        self._grad_mask = theta_mask_dense.bool()

        # Clear SparseMask caches AND indices to reclaim memory
        # Store essential stats before clearing
        self._n_pruned = self._mask.n_pruned
        self._n_active = self._mask.n_active
        self._mask._mask_cache = None
        self._mask._active_indices_cache = None
        self._mask._pruned_indices_cache = None
        # Clear indices (32 MB savings at 50% sparsity)
        self._mask._indices = torch.tensor([], dtype=torch.int32, device=self.device)

        # Gradient hook: only update θ positions via in-place masked_fill_
        if hasattr(self, '_resu_grad_hook_handle'):
            self._resu_grad_hook_handle.remove()

        def grad_mask_hook(grad):
            """Only allow gradients at θ positions (in-place, zero allocations)."""
            return grad.masked_fill_(~self._grad_mask, 0)

        self._resu_grad_hook_handle = self.weight.register_hook(grad_mask_hook)

        # Tensor cores disabled during training (need sparse-as-parameter for proper gradient flow)
        # Use to_sparse_inference() after exit_resu_mode() for tensor core inference
        self._use_tensor_cores = False
        self._structured_n = n
        self._structured_m = m
        self._weight_sparse = None

        self.weight.requires_grad_(True)
        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.RESU

    def exit_resu_mode(self, commit: bool = True):
        """Exit RESU mode.

        IN-PLACE MODE: θ is already in W, so commit is automatic.
        Just need to remove gradient hook and change mode.

        Args:
            commit: If True, keep θ values in W. If False, restore to zeros.
        """
        if self._mode != RESUMode.RESU:
            return

        # Remove gradient masking hook
        if hasattr(self, '_resu_grad_hook_handle'):
            self._resu_grad_hook_handle.remove()
            delattr(self, '_resu_grad_hook_handle')

        if not commit:
            # If not committing, zero out pruned positions (θ positions)
            # Use _grad_mask if available (True = pruned), else fall back to _mask
            with torch.no_grad():
                if self._grad_mask is not None:
                    self.weight.data[self._grad_mask] = 0.0
                elif self._mask is not None:
                    pruned_mask = ~self._mask.mask.bool()
                    self.weight.data[pruned_mask] = 0.0

        # W already trainable, keep it that way
        self.weight.requires_grad_(True)

        # Clean up gradient mask to free memory
        self._grad_mask = None
        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.SPARSE

    def exit_resu_mode_structured(self, n: int = 2, m: int = 4, dim: int = 1):
        """Exit RESU mode with EXACT N:M structured commit.

        This is the final step of structured densification:
        1. Wanda++ → partial 2:4 (≤N per M)
        2. RESU → trains θ for pruned positions
        3. THIS → commits to EXACT N:M by picking best N per M group

        For each group of M:
        - Candidates = {active weights} ∪ {trained θ}
        - Keep top-N by magnitude
        - Result: EXACTLY N:M structure!

        Args:
            n: Number to keep per group (default 2)
            m: Group size (default 4)
            dim: Dimension for N:M pattern (default 1 = columns)
        """
        from ..core.structured import commit_structured_nm

        if self._mode != RESUMode.RESU:
            return

        # Remove gradient masking hook
        if hasattr(self, '_resu_grad_hook_handle'):
            self._resu_grad_hook_handle.remove()
            delattr(self, '_resu_grad_hook_handle')

        if self._mask is not None:
            with torch.no_grad():
                mask = self._mask.mask.float()

                # In-place mode: W contains both active weights and θ
                # Active weights are where mask=1, θ where mask=0
                W_active = self.weight.data * mask
                theta = self.weight.data * (1 - mask)

                # Commit with structured constraint
                W_committed, mask_new = commit_structured_nm(
                    W_active, theta, mask, n, m, dim
                )

                # Update weight and mask
                self.weight.data.copy_(W_committed)

                # Convert mask tensor to SparseMask (need pruned indices)
                pruned_indices = (~mask_new.bool()).flatten().nonzero(as_tuple=True)[0]
                self._mask = SparseMask(
                    pruned_indices,
                    mask_new.shape,
                    device=self.weight.device
                )

        # Restore standard training state
        self.weight.requires_grad_(True)
        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.SPARSE

    def to_sparse_inference(self) -> "RESULinear":
        """Convert to sparse format for tensor core inference.

        Call this AFTER exit_resu_mode() when weights are finalized.
        Creates cached sparse representation for fast inference.

        Returns:
            self (for chaining)
        """
        if not SEMI_STRUCTURED_AVAILABLE:
            print("Warning: Semi-structured sparse not available (need PyTorch 2.1+)")
            return self

        if self._mask is None:
            print("Warning: No mask set, cannot convert to sparse")
            return self

        # Convert to FP16 sparse ONCE for inference
        assert to_sparse_semi_structured is not None
        with torch.no_grad():
            W_fp16 = self.weight.data.half()
            self._weight_sparse = to_sparse_semi_structured(W_fp16)
            self._use_tensor_cores = True

        print(f"Converted to 2:4 sparse for tensor core inference")
        return self

    def resu_step(self, grad_matrix: torch.Tensor) -> Optional[dict]:
        """Perform one RESU update step.

        IN-PLACE MODE: Manually update θ (stored in W[pruned_positions])
        using simple SGD. This maintains backward compatibility with code
        that doesn't use an optimizer.

        Args:
            grad_matrix: Gradient w.r.t. effective weights

        Returns:
            Update statistics (if selective), else None
        """
        if self._mode not in [RESUMode.RESU, RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
            raise RuntimeError("Not in RESU mode")

        if self._mask is None:
            return None

        # Extract gradients at pruned positions
        pruned_mask = ~self._mask.mask.bool()
        grad_pruned = grad_matrix[pruned_mask]

        # Simple SGD update on θ (in-place)
        lr = 1e-4  # Default learning rate
        with torch.no_grad():
            self.weight.data[pruned_mask] -= lr * grad_pruned

        return None

    # =========================================================================
    # QRESU Mode Management
    # =========================================================================

    def enter_qresu_mode(
        self,
        bits: int = 4,
        epsilon: float = 0.1,
        qscheme: Literal["per_channel", "per_tensor"] = "per_channel",
    ):
        """Enter QRESU mode - quantize W_A and train θ in full precision.

        OPTIMIZED STORAGE: θ stored as flat 1D tensor (not in weight matrix).
        Memory savings: ~25% less than RESU at 50% sparsity!

        Storage:
        - θ: (n_pruned,) FP32 tensor - trainable
        - W_A_quantized: Full matrix quantized (INT4/INT8)
        - Mask indices: int32

        Args:
            bits: Quantization bit-width (4 or 8)
            epsilon: Initialization scale for θ
            qscheme: 'per_tensor' or 'per_channel'
        """
        from ..utils.quantization import quantize_per_channel, quantize_per_tensor

        if self._mask is None:
            raise RuntimeError("Cannot enter QRESU mode without mask. Call set_mask() first.")

        if self._mode in [RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
            return  # Already in QRESU mode

        # Get masks
        active_mask = self._mask.mask.bool()
        pruned_mask = ~active_mask

        # 1. QUANTIZE ACTIVE WEIGHTS
        # Extract active weights (zero out pruned)
        W_active = self._mask.apply(self.weight.data)

        # Quantize using our custom implementation (PyTorch's qint4 not yet available)
        if qscheme == "per_channel":
            self._W_A_quantized, self._qscale, self._qzero = quantize_per_channel(
                W_active, bits=bits
            )
        else:
            self._W_A_quantized, self._qscale, self._qzero = quantize_per_tensor(
                W_active, bits=bits
            )

        self._qscheme = qscheme
        self._qbits = bits

        # 2. INITIALIZE θ AS FLAT 1D TENSOR (KEY OPTIMIZATION!)
        # Compute active weight statistics for initialization
        active_weights = self.weight.data[active_mask]
        active_std = active_weights.std().item() if len(active_weights) > 0 else 1.0

        # Store θ as flat 1D tensor - saves 0.25 MB at 50% sparsity!
        n_pruned = int(pruned_mask.sum().item())
        self._theta = torch.randn(n_pruned, device=self.device, dtype=self.dtype)
        self._theta *= epsilon * active_std
        self._theta.requires_grad_(True)

        # 3. WEIGHT TENSOR NO LONGER NEEDED FOR TRAINING
        # In QRESU mode, we reconstruct W_eff on-the-fly from θ + W_A_quantized
        self.weight.requires_grad_(False)

        # Remove any old hooks
        if hasattr(self, '_qresu_grad_hook_handle'):
            self._qresu_grad_hook_handle.remove()
            delattr(self, '_qresu_grad_hook_handle')

        # Clear old resurrection object
        self._resurrection = None

        self._mode = RESUMode.QRESU

    def enter_qresu_selective_mode(
        self,
        bits: int = 4,
        epsilon: float = 0.1,
        qscheme: Literal["per_channel", "per_tensor"] = "per_channel",
        selective_config: Optional[SimpleSelectionConfig] = None,
        selection_ratio: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        """Enter QRESU-Selective mode - quantized W_A with simple selective filtering.

        OPTIMIZED STORAGE: θ stored as flat 1D tensor with simple selection.

        Simple selective filtering (new approach):
        - Select random subset of θ coordinates at phase start
        - Only those coordinates receive gradient updates
        - Zero per-step overhead (no EMA, no TopK)

        Args:
            bits: Quantization bit-width (4 or 8)
            epsilon: Initialization scale for θ
            qscheme: 'per_tensor' or 'per_channel'
            selective_config: Configuration for selective updates (deprecated)
            selection_ratio: Fraction of θ to update (0.2 = 20%)
            lr: Learning rate for θ
            weight_decay: Weight decay for θ
        """
        # First, enter regular QRESU mode (sets up θ, quantized W_A, etc.)
        self.enter_qresu_mode(bits=bits, epsilon=epsilon, qscheme=qscheme)

        # Verify θ was created by enter_qresu_mode
        assert self._theta is not None, "θ should be initialized by enter_qresu_mode"
        n_pruned = self._theta.numel()

        # Simple selection: random subset at phase start (zero overhead!)
        k_select = max(1, int(n_pruned * selection_ratio))
        self._selected_theta_indices = torch.randperm(n_pruned, device=self.device)[:k_select]
        self._selective_lr = lr

        print(f"QRESU-Selective: Selected {k_select}/{n_pruned} θ positions ({selection_ratio:.0%})")

        # Register gradient hook for simple selective updates
        def simple_selective_grad_hook(grad):
            """Apply simple selective filtering to θ gradients."""
            if self._theta is None or self._theta.grad is None:
                return grad

            # Zero out non-selected gradients
            mask = torch.zeros(self._theta.numel(), device=self.device)
            mask[self._selected_theta_indices] = 1.0
            self._theta.grad.mul_(mask)

            return grad

        # Remove old hook if exists
        if hasattr(self, '_grad_hook_handle') and self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()

        # Attach hook to θ
        self._grad_hook_handle = self._theta.register_hook(simple_selective_grad_hook)

        self._mode = RESUMode.QRESU_SELECTIVE

    def exit_qresu_mode(self, commit: bool = True):
        """Exit QRESU mode - dequantize and merge with flat θ.

        OPTIMIZED MODE: θ stored as flat 1D tensor.
        Reconstructs full weight matrix from θ + dequantized W_A.

        Args:
            commit: If True, dequantize W_A and merge with θ into weight matrix
        """
        from ..utils.quantization import dequantize_per_channel

        if self._mode not in [RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
            return

        if commit and self._W_A_quantized is not None and self._mask is not None and self._theta is not None:
            # Dequantize active weights back to FP32
            if self._qscheme == "per_channel":
                if self._qscale is not None and self._qzero is not None:
                    W_active_dequant = dequantize_per_channel(
                        self._W_A_quantized, self._qscale, self._qzero
                    )
                else:
                    raise RuntimeError("Missing quantization parameters")
            else:
                # Per-tensor dequantization
                if self._qscale is not None and self._qzero is not None:
                    W_active_dequant = (self._W_A_quantized.float() - self._qzero) * self._qscale
                else:
                    raise RuntimeError("Missing quantization parameters")

            # Reconstruct full weight matrix: W_A + θ
            with torch.no_grad():
                active_mask = self._mask.mask.bool()
                # Place dequantized active weights
                self.weight.data[active_mask] = W_active_dequant[active_mask]
                # Place flat θ at pruned positions
                self.weight.data.view(-1)[self._mask.pruned_indices] = self._theta.data
        elif not commit and self._mask is not None:
            # Don't commit: zero out pruned positions (discard θ)
            with torch.no_grad():
                pruned_mask = ~self._mask.mask.bool()
                self.weight.data[pruned_mask] = 0.0

        # Re-enable full gradients on W
        self.weight.requires_grad_(True)

        # Free quantization storage
        self._W_A_quantized = None
        self._qscale = None
        self._qzero = None
        self._theta = None

        # Clean up selective filtering state
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        self._ema_m = None
        self._ema_v = None
        self._consistency = None
        self._selective_config = None
        self._selective_step_count = 0

        # Clear RESU state
        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.SPARSE

    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(self, x: torch.Tensor, freeze_active: bool = True, use_sparse: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., in_features)
            freeze_active: In RESU mode, don't backprop through active weights
            use_sparse: Whether to use sparse matrix multiplication when beneficial

        Returns:
            Output tensor (..., out_features)
        """
        # Capture activations if enabled
        if self._capture_activations:
            self._last_input = x.detach()
            # Compute L2 norm along batch dimensions
            self._activation_norms = x.detach().norm(p=2, dim=0)

        # Get weights based on mode
        if self._mode == RESUMode.DENSE:
            W = self.weight
            return F.linear(x, W, self.bias)

        elif self._mode == RESUMode.SPARSE:
            # Use tensor cores if converted to sparse (after to_sparse_inference())
            if self._use_tensor_cores and self._weight_sparse is not None:
                x_fp16 = x.half()
                bias_fp16 = self.bias.half() if self.bias is not None else None
                out = F.linear(x_fp16, self._weight_sparse, bias_fp16)
                return out.to(x.dtype)
            # Use sparse matmul if sparsity exceeds threshold
            # Note: torch.sparse has overhead on GPU; tune sparse_threshold based on hardware
            # Typical values: 0.7-0.9 for CPU, 0.95+ for GPU
            elif use_sparse and self.sparsity > self.sparse_threshold:
                return self._forward_sparse(x)
            else:
                # Dense matmul with masked weights (usually faster on GPU)
                W = self._mask.apply(self.weight)
                return F.linear(x, W, self.bias)

        elif self._mode == RESUMode.RESU:
            # IN-PLACE MODE: θ is already stored in W's pruned positions!
            # W now contains:
            #   - Active weights at active positions (frozen via grad hook)
            #   - θ values at pruned positions (trainable via grad hook)

            # Use tensor cores for 2:4 structured sparsity if enabled
            if self._use_tensor_cores and self._weight_sparse is not None:
                # Use cached sparse representation for tensor core forward
                # Sync values from weight to sparse tensor (structure unchanged)
                x_fp16 = x.half()
                bias_fp16 = self.bias.half() if self.bias is not None else None
                out = F.linear(x_fp16, self._weight_sparse, bias_fp16)
                return out.to(x.dtype)
            else:
                # Standard dense forward (no tensor core acceleration)
                return F.linear(x, self.weight, self.bias)

        elif self._mode in [RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
            # QRESU OPTIMIZED MODE:
            # Reconstruct W_eff on-the-fly from θ (flat 1D) + W_A_quantized
            from ..utils.quantization import dequantize_per_channel

            if self._W_A_quantized is None or self._mask is None or self._theta is None:
                raise RuntimeError("QRESU mode but missing quantized weights, mask, or theta")

            # Dequantize active weights
            if self._qscheme == "per_channel":
                if self._qscale is None or self._qzero is None:
                    raise RuntimeError("QRESU per_channel mode but no scale/zero_point found")
                W_active_dequant = dequantize_per_channel(
                    self._W_A_quantized, self._qscale, self._qzero
                )
            else:
                # Per-tensor dequantization
                if self._qscale is None or self._qzero is None:
                    raise RuntimeError("QRESU per_tensor mode but no scale/zero_point found")
                W_active_dequant = (self._W_A_quantized.float() - self._qzero) * self._qscale

            # Reconstruct W_eff: W_A (active) + θ (pruned)
            # This is the KEY OPTIMIZATION - reconstruct on-the-fly, minimal storage!
            W_eff = torch.zeros_like(self.weight)
            active_mask = self._mask.mask.bool()
            pruned_mask = ~active_mask

            # Place dequantized W_A at active positions
            W_eff[active_mask] = W_active_dequant[active_mask]
            # Place flat θ at pruned positions
            W_eff.view(-1)[self._mask.pruned_indices] = self._theta

            return F.linear(x, W_eff, self.bias)

        else:
            raise RuntimeError(f"Unknown mode: {self._mode}")

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using sparse matrix multiplication.

        Converts weight matrix to CSR format for efficient sparse-dense matmul.
        Only beneficial when sparsity > ~30%.
        """
        # Get sparse weight matrix (only active positions)
        W_sparse = self.weight * self._mask.mask

        # Convert to COO format (easier to construct)
        indices = torch.nonzero(W_sparse, as_tuple=False).t()
        values = W_sparse[indices[0], indices[1]]

        # Create sparse tensor in COO format
        # W_sparse_coo shape: (out_features, in_features)
        W_sparse_coo = torch.sparse_coo_tensor(
            indices,
            values,
            size=W_sparse.shape,
            device=W_sparse.device,
            dtype=W_sparse.dtype
        )

        # Reshape x for matrix multiplication
        # x: (..., in_features) -> (batch_size, in_features)
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])  # (batch_size, in_features)

        # Sparse-dense matmul: W @ x.T
        # W: (out_features, in_features), x.T: (in_features, batch_size)
        # Result: (out_features, batch_size)
        out = torch.sparse.mm(W_sparse_coo, x_2d.t())  # (out_features, batch_size)
        out = out.t()  # (batch_size, out_features)

        # Reshape back and add bias
        out = out.reshape(*x_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias

        return out
    
    # =========================================================================
    # Activation Capture (for Wanda)
    # =========================================================================
    
    def enable_activation_capture(self):
        """Enable activation capture for Wanda pruning."""
        self._capture_activations = True
    
    def disable_activation_capture(self):
        """Disable activation capture."""
        self._capture_activations = False
        self._last_input = None
    
    def clear_activation_cache(self):
        """Clear cached activations."""
        self._last_input = None
        self._activation_norms = None
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def get_effective_weight(self) -> torch.Tensor:
        """Get the effective weight matrix for current mode."""
        if self._mode == RESUMode.DENSE:
            return self.weight.data
        elif self._mode == RESUMode.SPARSE:
            return self._mask.apply(self.weight.data)
        elif self._mode == RESUMode.RESU:
            return self._mask.apply(self.weight.data) + self._resurrection.commit()
        else:
            raise RuntimeError(f"Unknown mode: {self._mode}")
    
    def state_dict_extended(self) -> Dict[str, Any]:
        """Extended state dict including RESU state."""
        state = {
            'weight': self.weight.data.cpu(),
            'bias': self.bias.data.cpu() if self.bias is not None else None,
            'mode': self._mode.name,
            'in_features': self.in_features,
            'out_features': self.out_features,
        }
        
        if self._mask is not None:
            state['mask'] = self._mask.state_dict()
        
        if self._resurrection is not None:
            state['resurrection'] = self._resurrection.state_dict()
        
        if self._selective is not None:
            state['selective'] = self._selective.state_dict()
        
        return state
    
    def load_state_dict_extended(self, state: Dict[str, Any]):
        """Load extended state dict."""
        self.weight.data = state['weight'].to(self.device, self.dtype)
        if state['bias'] is not None and self.bias is not None:
            self.bias.data = state['bias'].to(self.device, self.dtype)
        
        self._mode = RESUMode[state['mode']]
        
        if 'mask' in state:
            self._mask = SparseMask.from_state_dict(state['mask'], self.device)
        
        if 'resurrection' in state:
            self._resurrection = ResurrectionEmbedding(
                self._mask, self.storage_mode, self.device, self.dtype
            )
            self._resurrection.load_state_dict(state['resurrection'])
        
        if 'selective' in state and self._resurrection is not None:
            # Deprecated: old RESUSelective state loading
            # New simple selection doesn't need state (selection happens at phase start)
            print("Warning: Ignoring old 'selective' state (deprecated). Using simple selection now.")
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category."""
        total = self.weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        
        if self._mask is None:
            return {
                'total': total,
                'active': total,
                'pruned': 0,
                'resurrection': 0,
            }
        
        return {
            'total': total,
            'active': self._mask.n_active + (self.bias.numel() if self.bias is not None else 0),
            'pruned': self._mask.n_pruned,
            'resurrection': self._resurrection.p if self._resurrection is not None else 0,
        }
    
    def extra_repr(self) -> str:
        s = f'{self.in_features}, {self.out_features}'
        if self.bias is None:
            s += ', bias=False'
        s += f', mode={self._mode.name}'
        if self._mask is not None:
            s += f', sparsity={self.sparsity:.1%}'
        return s
    
    # =========================================================================
    # Class Methods
    # =========================================================================
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        storage_mode: StorageMode = StorageMode.COMPACT,
    ) -> "RESULinear":
        """Create RESULinear from existing nn.Linear.
        
        Args:
            linear: Source linear layer
            storage_mode: Storage mode for resurrection parameters
            
        Returns:
            RESULinear with copied weights
        """
        resu = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            storage_mode=storage_mode,
        )
        
        resu.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and resu.bias is not None:
            resu.bias.data.copy_(linear.bias.data)
        
        return resu


# =============================================================================
# Utility: Convert model
# =============================================================================

def convert_to_resu(
    model: nn.Module,
    target_modules: Optional[list] = None,
    storage_mode: StorageMode = StorageMode.COMPACT,
) -> nn.Module:
    """Convert nn.Linear layers in a model to RESULinear.
    
    Args:
        model: Model to convert
        target_modules: List of module names to convert (None = all Linear)
        storage_mode: Storage mode for resurrection parameters
        
    Returns:
        Model with RESULinear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is None or name in target_modules:
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Replace with RESULinear
                resu_linear = RESULinear.from_linear(module, storage_mode)
                setattr(parent, child_name, resu_linear)
    
    return model


def get_resu_modules(model: nn.Module) -> Dict[str, RESULinear]:
    """Get all RESULinear modules in a model."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RESULinear)
    }


def find_layers_qwen2(layer: nn.Module) -> Dict[str, nn.Linear]:
    """Find all Linear layers in a transformer block.
    
    Compatible with Qwen2.5 and similar architectures.
    """
    out = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, (nn.Linear, RESULinear)):
            out[name] = mod
    return out


# =============================================================================
# Testing
# =============================================================================

def _test_resu_linear():
    """Test RESULinear functionality."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on {device}...")
    
    in_features, out_features = 128, 64
    batch_size = 32
    
    # Create layer
    layer = RESULinear(in_features, out_features, device=device)
    print(f"Created: {layer}")
    
    # Test dense forward
    x = torch.randn(batch_size, in_features, device=device)
    y = layer(x)
    assert y.shape == (batch_size, out_features), f"Wrong output shape: {y.shape}"
    print("✓ Dense forward works")
    
    # Test pruning
    layer.prune_by_magnitude(0.5)
    print(f"After pruning: {layer}")
    assert abs(layer.sparsity - 0.5) < 0.01, f"Wrong sparsity: {layer.sparsity}"
    
    # Verify pruned weights are zero
    pruned_vals = layer.weight.data.view(-1)[layer.mask.pruned_indices]
    assert (pruned_vals == 0).all(), "Pruned weights should be zero"
    print("✓ Magnitude pruning works")
    
    # Test sparse forward
    y_sparse = layer(x)
    assert y_sparse.shape == (batch_size, out_features)
    print("✓ Sparse forward works")
    
    # Test RESU mode
    layer.enter_resu_mode(epsilon=0.1, use_selective=True)
    print(f"In RESU mode: {layer}")
    assert layer.is_resu_active
    
    # RESU forward
    y_resu = layer(x)
    assert y_resu.shape == (batch_size, out_features)
    
    # Backward should work
    loss = y_resu.sum()
    loss.backward()
    
    # Check that weight gradients exist (θ is stored in-place in W)
    # In-place mode: gradients flow to W, masked by _grad_mask
    assert layer.weight.grad is not None, "Weight should have gradients"
    print("✓ RESU forward/backward works")
    
    # Test manual RESU step
    grad = torch.randn_like(layer.weight)
    stats = layer.resu_step(grad)
    if stats:
        print(f"  RESU step stats: selected={stats['n_selected']}")
    print("✓ RESU step works")
    
    # Test commit
    # With in-place θ storage, θ is already in W - commit just removes hook
    old_weight = layer.weight.data.clone()
    layer.exit_resu_mode(commit=True)

    assert layer.mode == RESUMode.SPARSE
    # In-place mode: weights stay the same on commit (θ already in W)
    assert torch.equal(old_weight, layer.weight.data), "Weights should stay same on commit (in-place mode)"
    print("✓ RESU commit works (in-place mode)")
    
    # Test from_linear conversion
    linear = nn.Linear(256, 128, device=device)
    resu_from = RESULinear.from_linear(linear)
    assert torch.equal(resu_from.weight.data, linear.weight.data)
    print("✓ from_linear conversion works")
    
    # Test state dict - use fresh layer since committed layer has no sparsity
    layer_fresh = RESULinear(in_features, out_features, device=device)
    layer_fresh.prune_by_magnitude(0.5)
    layer_fresh.enter_resu_mode()
    state = layer_fresh.state_dict_extended()

    layer3 = RESULinear(in_features, out_features, device=device)
    layer3.load_state_dict_extended(state)

    assert layer3.mode == layer_fresh.mode
    assert torch.allclose(layer3.weight.data, layer_fresh.weight.data)
    print("✓ State dict round-trip works")
    
    print("\n✓ All RESULinear tests passed!")


if __name__ == "__main__":
    _test_resu_linear()
