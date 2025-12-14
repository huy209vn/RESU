"""
ResurrectionEmbedding: The core abstraction for RESU.

Implements:
- Φ: ℝᵖ → S_P (embed θ into pruned subspace)
- Φ⁻¹: S_P → ℝᵖ (extract from pruned subspace)

The resurrection parameters θ are learnable during RESU phases,
allowing pruned weights to compete for reactivation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from enum import Enum, auto

from .mask import SparseMask
from ..kernels.embedding import (
    phi_scatter,
    phi_inverse_gather,
    resu_update_indexed,
    resu_update_with_momentum,
    DenseResurrectionBuffer,
)


class StorageMode(Enum):
    """How θ is stored in memory."""
    COMPACT = auto()    # θ ∈ ℝᵖ as separate vector
    DENSE = auto()      # θ stored in full matrix (pruned positions)


class ResurrectionEmbedding:
    """Resurrection Embedding for RESU.
    
    Manages the learnable resurrection parameters θ that allow
    pruned weights to be trained and potentially reactivated.
    
    During RESU phases:
    - θ receives gradient updates
    - Active weights (W_A) are frozen
    - Effective weights W_eff = M⊙W + (1-M)⊙Φ(θ)
    
    Two storage modes:
    - COMPACT: θ ∈ ℝᵖ stored as 1D tensor (needs scatter/gather)
    - DENSE: θ stored directly in W-shaped tensor (zero overhead)
    """
    
    def __init__(
        self,
        sparse_mask: SparseMask,
        storage_mode: StorageMode = StorageMode.COMPACT,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            sparse_mask: The SparseMask defining (A, P) partition
            storage_mode: How to store θ
            device: CUDA device
            dtype: Data type for θ
        """
        self.mask = sparse_mask
        self.storage_mode = storage_mode
        self.device = device or sparse_mask.device
        self.dtype = dtype
        
        # Resurrection parameters
        self._theta: Optional[torch.Tensor] = None
        self._dense_buffer: Optional[DenseResurrectionBuffer] = None
        
        # Optimizer states (for fused updates)
        self._momentum: Optional[torch.Tensor] = None
        self._variance: Optional[torch.Tensor] = None
        self._step_count: int = 0
        
        # State
        self._initialized = False
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def p(self) -> int:
        """Number of resurrection parameters (= number of pruned positions)."""
        return self.mask.n_pruned
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the weight matrix."""
        return self.mask.shape
    
    @property
    def theta(self) -> torch.Tensor:
        """The resurrection parameters θ.
        
        Returns compact form (p,) regardless of storage mode.
        """
        if not self._initialized:
            raise RuntimeError("ResurrectionEmbedding not initialized")
        
        if self.storage_mode == StorageMode.COMPACT:
            return self._theta
        else:
            return self._dense_buffer.get_theta_compact()
    
    @theta.setter
    def theta(self, value: torch.Tensor):
        """Set θ from compact vector."""
        if not self._initialized:
            raise RuntimeError("ResurrectionEmbedding not initialized")

        assert value.shape == (self.p,), f"Expected ({self.p},), got {value.shape}"

        if self.storage_mode == StorageMode.COMPACT:
            with torch.no_grad():
                self._theta.copy_(value)
        else:
            self._dense_buffer.set_theta_compact(value)
    
    @property
    def requires_grad(self) -> bool:
        """Whether θ has gradients enabled."""
        if not self._initialized:
            return False
        
        if self.storage_mode == StorageMode.COMPACT:
            return self._theta.requires_grad
        else:
            return self._dense_buffer.theta_dense.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool):
        if not self._initialized:
            raise RuntimeError("Not initialized")
        
        if self.storage_mode == StorageMode.COMPACT:
            self._theta.requires_grad_(value)
        else:
            self._dense_buffer.theta_dense.requires_grad_(value)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize(
        self,
        active_std: float,
        epsilon: float = 0.1,
        init_type: str = "normal",
    ):
        """Initialize resurrection parameters.
        
        θ_k ~ N(0, ε·σ_A) where σ_A is std of active weights.
        
        This scale-matched initialization ensures resurrection parameters
        begin at comparable magnitude to active weights.
        
        Args:
            active_std: Standard deviation of active weights
            epsilon: Scale factor (typically 0.05-0.2)
            init_type: 'normal', 'uniform', or 'zero'
        """
        p = self.p
        init_scale = epsilon * active_std
        
        if self.storage_mode == StorageMode.COMPACT:
            if init_type == "normal":
                self._theta = torch.randn(p, device=self.device, dtype=self.dtype) * init_scale
            elif init_type == "uniform":
                self._theta = (torch.rand(p, device=self.device, dtype=self.dtype) * 2 - 1) * init_scale
            elif init_type == "zero":
                self._theta = torch.zeros(p, device=self.device, dtype=self.dtype)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            self._theta.requires_grad_(True)
        
        else:  # DENSE mode
            self._dense_buffer = DenseResurrectionBuffer(
                self.shape, self.mask.mask, self.device
            )
            if init_type != "zero":
                self._dense_buffer.initialize(active_std, epsilon)
            self._dense_buffer.theta_dense.requires_grad_(True)
        
        self._initialized = True
    
    def initialize_from(self, source: torch.Tensor):
        """Initialize θ from existing values at pruned positions.
        
        Useful for warm-starting from previous weights.
        """
        if self.storage_mode == StorageMode.COMPACT:
            # Gather from pruned positions
            self._theta = phi_inverse_gather(
                source, self.mask.pruned_indices
            ).to(self.dtype)
            self._theta.requires_grad_(True)
        else:
            self._dense_buffer = DenseResurrectionBuffer(
                self.shape, self.mask.mask, self.device
            )
            # Copy values at pruned positions
            flat = source.view(-1)
            self._dense_buffer.theta_dense.view(-1)[self.mask.pruned_indices] = \
                flat[self.mask.pruned_indices].to(self.dtype)
            self._dense_buffer.theta_dense.requires_grad_(True)
        
        self._initialized = True
    
    # =========================================================================
    # Core Operations: Φ and Φ⁻¹
    # =========================================================================
    
    def phi(self) -> torch.Tensor:
        """Φ(θ): Embed θ into pruned subspace S_P.

        Returns matrix with θ values at pruned positions, zeros elsewhere.
        Shape matches the weight matrix.

        Note: Uses gradient-enabled scatter for autograd support.
        """
        if not self._initialized:
            raise RuntimeError("Not initialized")

        if self.storage_mode == StorageMode.COMPACT:
            # Use gradient-enabled version for proper backprop
            from ..kernels.embedding import phi_scatter_grad
            return phi_scatter_grad(
                self._theta,
                self.mask.pruned_indices,
                self.shape,
            )
        else:
            return self._dense_buffer.phi()
    
    def phi_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
        """Φ⁻¹(V): Extract values from pruned positions.
        
        Args:
            matrix: Full matrix (typically gradient)
            
        Returns:
            Compact vector (p,) of values at pruned positions
        """
        return phi_inverse_gather(matrix, self.mask.pruned_indices)
    
    # =========================================================================
    # Effective Weight Computation
    # =========================================================================
    
    def effective_weights(
        self,
        W: torch.Tensor,
        detach_active: bool = True,
    ) -> torch.Tensor:
        """Compute W_eff = M⊙W + (1-M)⊙Φ(θ)
        
        Args:
            W: Original weight matrix
            detach_active: If True, active weights don't receive gradients
            
        Returns:
            Effective weights for forward pass
        """
        if not self._initialized:
            raise RuntimeError("Not initialized")
        
        # Active component
        if detach_active:
            W_A = self.mask.apply(W.detach())
        else:
            W_A = self.mask.apply(W)
        
        # Resurrection component
        phi_theta = self.phi()
        
        # Combine: pruned positions get θ, active get W
        W_eff = W_A + phi_theta
        
        return W_eff
    
    # =========================================================================
    # RESU Updates
    # =========================================================================
    
    def init_optimizer_state(self, with_variance: bool = True):
        """Initialize optimizer state for fused updates."""
        p = self.p
        self._momentum = torch.zeros(p, device=self.device, dtype=self.dtype)
        if with_variance:
            self._variance = torch.zeros(p, device=self.device, dtype=self.dtype)
        self._step_count = 0
    
    def update_sgd(
        self,
        grad_matrix: torch.Tensor,
        lr: float,
    ):
        """SGD update: θ ← θ - η·Φ⁻¹(G)
        
        Fused gather and update.
        """
        if not self._initialized:
            raise RuntimeError("Not initialized")
        
        if self.storage_mode == StorageMode.COMPACT:
            resu_update_indexed(
                self._theta,
                grad_matrix,
                self.mask.pruned_indices,
                lr,
            )
        else:
            # For dense mode, need to update in-place
            grad_theta = self.phi_inverse(grad_matrix)
            with torch.no_grad():
                self._dense_buffer.theta_dense.view(-1)[self.mask.pruned_indices] -= lr * grad_theta
    
    def update_momentum(
        self,
        grad_matrix: torch.Tensor,
        lr: float,
        beta: float = 0.9,
    ):
        """Momentum SGD update.
        
        m ← β·m + (1-β)·g
        θ ← θ - η·m
        """
        if self._momentum is None:
            self.init_optimizer_state(with_variance=False)
        
        if self.storage_mode == StorageMode.COMPACT:
            resu_update_with_momentum(
                self._theta,
                grad_matrix,
                self.mask.pruned_indices,
                self._momentum,
                lr,
                beta,
            )
        else:
            grad_theta = self.phi_inverse(grad_matrix)
            self._momentum = beta * self._momentum + (1 - beta) * grad_theta
            self._dense_buffer.theta_dense.view(-1)[self.mask.pruned_indices] -= lr * self._momentum
    
    def update_adam(
        self,
        grad_matrix: torch.Tensor,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """AdamW update for θ.
        
        Full Adam optimizer step with bias correction.
        """
        if self._momentum is None or self._variance is None:
            self.init_optimizer_state(with_variance=True)
        
        self._step_count += 1
        
        # Get gradient
        grad_theta = self.phi_inverse(grad_matrix)
        
        # Update biased moments
        self._momentum = beta1 * self._momentum + (1 - beta1) * grad_theta
        self._variance = beta2 * self._variance + (1 - beta2) * grad_theta.square()
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** self._step_count
        bias_correction2 = 1 - beta2 ** self._step_count
        
        m_hat = self._momentum / bias_correction1
        v_hat = self._variance / bias_correction2
        
        # Update
        update = m_hat / (v_hat.sqrt() + eps)
        
        if self.storage_mode == StorageMode.COMPACT:
            # Weight decay
            if weight_decay > 0:
                self._theta.data.mul_(1 - lr * weight_decay)
            # Adam update
            self._theta.data.sub_(lr * update)
        else:
            theta_flat = self._dense_buffer.theta_dense.view(-1)
            if weight_decay > 0:
                theta_flat[self.mask.pruned_indices].mul_(1 - lr * weight_decay)
            theta_flat[self.mask.pruned_indices] -= lr * update
    
    # =========================================================================
    # Commit: Transfer θ back to weights
    # =========================================================================
    
    def commit(self) -> torch.Tensor:
        """Get final Φ(θ) for committing to weight matrix.
        
        Called at end of RESU phase to merge θ into weights.
        """
        return self.phi()
    
    def commit_to(self, W: torch.Tensor) -> torch.Tensor:
        """Commit θ to weight matrix: W ← M⊙W + Φ(θ)
        
        Returns modified weight matrix.
        """
        return self.effective_weights(W, detach_active=False)
    
    # =========================================================================
    # Mask Updates
    # =========================================================================
    
    def update_mask(self, new_mask: SparseMask, preserve_matching: bool = True):
        """Update the underlying mask (e.g., after amnesty pruning).
        
        Args:
            new_mask: New SparseMask
            preserve_matching: Keep θ values for positions that remain pruned
        """
        if preserve_matching and self._initialized:
            # Find positions that were pruned before and are still pruned
            old_pruned = set(self.mask.pruned_indices.cpu().numpy())
            new_pruned = set(new_mask.pruned_indices.cpu().numpy())
            
            # Get old θ values
            old_theta = self.theta.clone()
            old_indices = self.mask.pruned_indices
            
            # Create mapping
            old_to_theta = {int(idx): i for i, idx in enumerate(old_indices.cpu().numpy())}
            
            # Update mask
            self.mask = new_mask
            
            # Reinitialize with zeros
            self._theta = torch.zeros(new_mask.n_pruned, device=self.device, dtype=self.dtype)
            self._theta.requires_grad_(True)
            
            # Copy matching values
            for i, idx in enumerate(new_mask.pruned_indices.cpu().numpy()):
                if idx in old_to_theta:
                    self._theta.data[i] = old_theta[old_to_theta[idx]]
            
            # Reset optimizer state
            self._momentum = None
            self._variance = None
            self._step_count = 0
        else:
            self.mask = new_mask
            self._initialized = False
    
    # =========================================================================
    # State dict
    # =========================================================================
    
    def state_dict(self) -> dict:
        """Serialize state."""
        state = {
            "storage_mode": self.storage_mode.name,
            "initialized": self._initialized,
            "step_count": self._step_count,
        }
        
        if self._initialized:
            state["theta"] = self.theta.cpu()
        if self._momentum is not None:
            state["momentum"] = self._momentum.cpu()
        if self._variance is not None:
            state["variance"] = self._variance.cpu()
        
        return state
    
    def load_state_dict(self, state: dict):
        """Load state."""
        if state["initialized"]:
            p = self.p
            self._theta = state["theta"].to(self.device, self.dtype)
            self._theta.requires_grad_(True)
            
            if "momentum" in state:
                self._momentum = state["momentum"].to(self.device, self.dtype)
            if "variance" in state:
                self._variance = state["variance"].to(self.device, self.dtype)
            
            self._step_count = state.get("step_count", 0)
            self._initialized = True
    
    # =========================================================================
    # Repr
    # =========================================================================
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "uninitialized"
        return (f"ResurrectionEmbedding(p={self.p}, shape={list(self.shape)}, "
                f"mode={self.storage_mode.name}, {status})")


# =============================================================================
# Testing
# =============================================================================

def _test_resurrection_embedding():
    """Test ResurrectionEmbedding."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on {device}...")
    
    # Create mask
    shape = (256, 128)
    mask_tensor = (torch.rand(shape, device=device) > 0.5).float()
    sparse_mask = SparseMask(mask_tensor)
    
    # Test COMPACT mode
    print("\nTesting COMPACT mode...")
    embed = ResurrectionEmbedding(sparse_mask, StorageMode.COMPACT, device)
    print(f"Created: {embed}")
    
    # Initialize
    active_std = 0.5
    embed.initialize(active_std, epsilon=0.1)
    print(f"After init: {embed}")
    
    # Check theta shape
    theta = embed.theta
    assert theta.shape == (embed.p,), f"Wrong theta shape: {theta.shape}"
    assert theta.requires_grad, "theta should require grad"
    
    # Test phi
    phi_theta = embed.phi()
    assert phi_theta.shape == shape, f"Wrong phi shape: {phi_theta.shape}"
    
    # Verify: active positions should be 0
    active_vals = phi_theta.view(-1)[sparse_mask.active_indices]
    assert (active_vals == 0).all(), "Phi should be 0 at active positions"
    
    # Verify: pruned positions should have theta values
    pruned_vals = phi_theta.view(-1)[sparse_mask.pruned_indices]
    assert torch.allclose(pruned_vals, theta), "Phi mismatch at pruned positions"
    
    # Test phi_inverse
    matrix = torch.randn(shape, device=device)
    gathered = embed.phi_inverse(matrix)
    expected = matrix.view(-1)[sparse_mask.pruned_indices]
    assert torch.allclose(gathered, expected), "phi_inverse failed"
    
    # Test effective weights
    W = torch.randn(shape, device=device)
    W_eff = embed.effective_weights(W)
    
    # Verify: active positions have W, pruned have theta
    W_active = W_eff.view(-1)[sparse_mask.active_indices]
    W_expected_active = W.view(-1)[sparse_mask.active_indices]
    assert torch.allclose(W_active, W_expected_active), "Active weights mismatch"
    
    W_pruned = W_eff.view(-1)[sparse_mask.pruned_indices]
    assert torch.allclose(W_pruned, theta), "Pruned weights should be theta"
    
    # Test updates
    grad = torch.randn(shape, device=device)
    old_theta = embed.theta.clone()
    
    embed.update_sgd(grad, lr=0.01)
    new_theta = embed.theta
    
    # Verify update happened
    assert not torch.allclose(old_theta, new_theta), "SGD update didn't change theta"
    
    # Test Adam update
    embed.initialize(active_std, epsilon=0.1)
    for _ in range(10):
        grad = torch.randn(shape, device=device)
        embed.update_adam(grad, lr=0.001)
    
    assert embed._step_count == 10, "Step count wrong"
    
    # Test state dict
    state = embed.state_dict()
    
    embed2 = ResurrectionEmbedding(sparse_mask, StorageMode.COMPACT, device)
    embed2.load_state_dict(state)
    
    assert torch.allclose(embed.theta, embed2.theta), "State load failed"
    
    print("✓ COMPACT mode tests passed!")
    
    # Test DENSE mode
    print("\nTesting DENSE mode...")
    embed_dense = ResurrectionEmbedding(sparse_mask, StorageMode.DENSE, device)
    embed_dense.initialize(active_std, epsilon=0.1)
    
    phi_dense = embed_dense.phi()
    assert phi_dense.shape == shape
    
    # Should match compact mode behavior
    pruned_vals_dense = phi_dense.view(-1)[sparse_mask.pruned_indices]
    assert len(pruned_vals_dense) == embed_dense.p
    
    print("✓ DENSE mode tests passed!")
    
    print("\n✓ All ResurrectionEmbedding tests passed!")


if __name__ == "__main__":
    _test_resurrection_embedding()
