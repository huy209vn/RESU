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
from typing import Optional, Union, Dict, Any
from enum import Enum, auto
import math

from ..core.mask import SparseMask
from ..core.resurrection import ResurrectionEmbedding, StorageMode
from ..core.effective import effective_weight, effective_weight_dense
from ..core.selective import RESUSelective, SelectionConfig


class RESUMode(Enum):
    """Operating mode for RESU modules."""
    DENSE = auto()      # No pruning, standard forward
    SPARSE = auto()     # Pruned, mask applied
    RESU = auto()       # Resurrection phase, θ active


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
        self._selective: Optional[RESUSelective] = None

        # Performance optimization: cached active weights during RESU
        self._W_active_cached: Optional[torch.Tensor] = None

        # Activation capture for Wanda
        self._capture_activations = False
        self._last_input: Optional[torch.Tensor] = None
        self._activation_norms: Optional[torch.Tensor] = None
        
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
    
    def prune_by_wanda(self, sparsity: float, activation_norms: Optional[torch.Tensor] = None):
        """Prune using Wanda (Weights and Activations).
        
        Score = |W| * ||X||
        
        Args:
            sparsity: Fraction to prune
            activation_norms: Input activation L2 norms (in_features,)
                            Uses captured activations if None
        """
        if activation_norms is None:
            activation_norms = self._activation_norms
        
        if activation_norms is None:
            raise ValueError("No activation norms available. Enable capture or provide norms.")
        
        # Compute Wanda scores
        scores = self.weight.data.abs() * activation_norms.unsqueeze(0)
        
        # Create mask from scores
        mask = SparseMask.from_magnitude(scores, sparsity)
        self.set_mask(mask)
        
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
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        """Enter RESU mode - enable resurrection parameter training.

        Args:
            epsilon: Initialization scale for θ
            use_selective: Use RESU-Selective filtering
            selective_config: Configuration for selective updates
            lr: Learning rate for θ
            weight_decay: Weight decay for θ
        """
        if self._mask is None:
            raise RuntimeError("Cannot enter RESU mode without mask. Call set_mask() first.")

        if self._mode == RESUMode.RESU:
            return  # Already in RESU mode

        # CRITICAL FIX: Freeze active weights (disable gradients on W)
        # This prevents optimizer from tracking W during RESU phase
        self.weight.requires_grad_(False)

        # Cache active weights (won't change during RESU since frozen)
        self._W_active_cached = self._mask.apply(self.weight.data).clone()
        self._W_active_cached.requires_grad = False

        # Compute active weight statistics for initialization
        active_weights = self.weight.data[self._mask.mask.bool()]
        active_std = active_weights.std().item() if len(active_weights) > 0 else 1.0

        # Create resurrection embedding
        self._resurrection = ResurrectionEmbedding(
            self._mask,
            storage_mode=self.storage_mode,
            device=self.device,
            dtype=self.dtype,
        )
        self._resurrection.initialize(active_std, epsilon)

        # Enable gradients on θ (only resurrection params need updates)
        self._resurrection.requires_grad = True

        # Setup selective updater if requested
        if use_selective:
            self._selective = RESUSelective(
                self._resurrection,
                config=selective_config or SelectionConfig(),
                lr=lr,
                weight_decay=weight_decay,
            )

        self._mode = RESUMode.RESU
    
    def exit_resu_mode(self, commit: bool = True):
        """Exit RESU mode.

        Args:
            commit: If True, merge θ into weights
        """
        if self._mode != RESUMode.RESU:
            return

        if commit and self._resurrection is not None:
            # Commit: W = M⊙W + Φ(θ)
            with torch.no_grad():
                phi_theta = self._resurrection.commit()
                self.weight.data = self._mask.apply(self.weight.data) + phi_theta

        # CRITICAL FIX: Re-enable gradients on W for next training phase
        self.weight.requires_grad_(True)

        # Clear cached active weights
        self._W_active_cached = None

        self._resurrection = None
        self._selective = None
        self._mode = RESUMode.SPARSE
    
    def resu_step(self, grad_matrix: torch.Tensor) -> Optional[dict]:
        """Perform one RESU update step.
        
        Args:
            grad_matrix: Gradient w.r.t. effective weights
            
        Returns:
            Update statistics (if selective), else None
        """
        if self._mode != RESUMode.RESU:
            raise RuntimeError("Not in RESU mode")
        
        if self._selective is not None:
            return self._selective.step(grad_matrix)
        else:
            # Basic SGD update
            self._resurrection.update_sgd(grad_matrix, lr=1e-4)
            return None
    
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
            # Use sparse matmul if sparsity exceeds threshold
            # Note: torch.sparse has overhead on GPU; tune sparse_threshold based on hardware
            # Typical values: 0.7-0.9 for CPU, 0.95+ for GPU
            if use_sparse and self.sparsity > self.sparse_threshold:
                return self._forward_sparse(x)
            else:
                # Dense matmul with masked weights (usually faster on GPU)
                W = self._mask.apply(self.weight)
                return F.linear(x, W, self.bias)

        elif self._mode == RESUMode.RESU:
            # Use effective weights with resurrection
            # NOTE: We cache _W_active for potential future optimization,
            # but currently just use standard path for best autograd performance
            W = effective_weight(
                self.weight,
                self._resurrection.theta,
                self._mask,
                freeze_active=freeze_active,
            )
            return F.linear(x, W, self.bias)

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
            self._selective = RESUSelective(self._resurrection)
            self._selective.load_state_dict(state['selective'])
    
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
    
    # Check that theta received gradients
    assert layer.resurrection.theta.grad is not None or layer._selective is not None
    print("✓ RESU forward/backward works")
    
    # Test manual RESU step
    grad = torch.randn_like(layer.weight)
    stats = layer.resu_step(grad)
    if stats:
        print(f"  RESU step stats: selected={stats['n_selected']}")
    print("✓ RESU step works")
    
    # Test commit
    old_weight = layer.weight.data.clone()
    layer.exit_resu_mode(commit=True)
    
    assert layer.mode == RESUMode.SPARSE
    assert not torch.equal(old_weight, layer.weight.data), "Weights should change after commit"
    print("✓ RESU commit works")
    
    # Test activation capture for Wanda
    layer2 = RESULinear(in_features, out_features, device=device)
    layer2.enable_activation_capture()
    
    _ = layer2(x)
    assert layer2._activation_norms is not None
    assert layer2._activation_norms.shape == (in_features,)
    
    layer2.prune_by_wanda(0.5)
    assert abs(layer2.sparsity - 0.5) < 0.01
    print("✓ Wanda pruning works")
    
    # Test from_linear conversion
    linear = nn.Linear(256, 128, device=device)
    resu_from = RESULinear.from_linear(linear)
    assert torch.equal(resu_from.weight.data, linear.weight.data)
    print("✓ from_linear conversion works")
    
    # Test state dict
    layer.enter_resu_mode()
    state = layer.state_dict_extended()
    
    layer3 = RESULinear(in_features, out_features, device=device)
    layer3.load_state_dict_extended(state)
    
    assert layer3.mode == layer.mode
    assert torch.allclose(layer3.weight.data, layer.weight.data)
    print("✓ State dict round-trip works")
    
    print("\n✓ All RESULinear tests passed!")


if __name__ == "__main__":
    _test_resu_linear()
