"""
EffectiveWeight: Custom autograd for RESU forward/backward.

Computes W_eff = M⊙W + (1-M)⊙Φ(θ) with proper gradient routing:
- ∂L/∂W_A flows to active weights (when not frozen)
- ∂L/∂θ flows to resurrection parameters

The key insight: during RESU, we want gradients for θ but not for W.
This autograd function handles that cleanly.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple, Any

from .mask import SparseMask
from ..kernels.masked_ops import (
    fused_effective_weight,
    split_gradient,
    extract_pruned_gradient,
)
from ..kernels.embedding import phi_scatter, phi_inverse_gather


class EffectiveWeightFunction(Function):
    """Autograd function for W_eff = M⊙W + (1-M)⊙Φ(θ)
    
    Forward:
        Computes effective weights combining active W and resurrection θ.
        
    Backward:
        Splits gradient by mask:
        - G_A = M⊙G → gradient for W (if needs_grad)
        - G_P = (1-M)⊙G → Φ⁻¹ → gradient for θ
    """
    
    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,              # Original weights
        theta: torch.Tensor,          # Resurrection params (compact, p-dim)
        mask: torch.Tensor,           # Binary mask (float)
        pruned_indices: torch.Tensor, # Flat indices of pruned positions
        freeze_active: bool = True,   # Whether to freeze active weights
    ) -> torch.Tensor:
        """
        Args:
            W: Weight matrix
            theta: Resurrection parameters (p,)
            mask: Binary mask (1=active, 0=pruned)
            pruned_indices: Indices for Φ/Φ⁻¹
            freeze_active: If True, don't compute grad for W
            
        Returns:
            W_eff = M⊙W + (1-M)⊙Φ(θ)
        """
        # Embed θ into full matrix
        phi_theta = phi_scatter(theta, pruned_indices, W.shape)
        
        # Compute effective weights
        W_eff = fused_effective_weight(W, phi_theta, mask)
        
        # Save for backward
        ctx.save_for_backward(mask, pruned_indices)
        ctx.freeze_active = freeze_active
        ctx.W_shape = W.shape
        
        return W_eff
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Args:
            grad_output: ∂L/∂W_eff
            
        Returns:
            grad_W: ∂L/∂W = M⊙(∂L/∂W_eff) if not frozen, else None
            grad_theta: ∂L/∂θ = Φ⁻¹((1-M)⊙(∂L/∂W_eff))
            None, None, None for mask, indices, freeze_active
        """
        mask, pruned_indices = ctx.saved_tensors
        
        # Gradient for theta: gather from pruned positions
        grad_theta = phi_inverse_gather(grad_output, pruned_indices)
        
        # Gradient for W (only if not frozen)
        if ctx.freeze_active:
            grad_W = None
        else:
            # Only active positions get gradient
            grad_W = mask * grad_output
        
        return grad_W, grad_theta, None, None, None


class EffectiveWeightDense(Function):
    """Variant when θ is stored in dense form (same shape as W).
    
    More memory efficient when we already have Φ(θ) computed.
    """
    
    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,         # Original weights
        phi_theta: torch.Tensor, # Pre-computed Φ(θ)
        mask: torch.Tensor,      # Binary mask
        freeze_active: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            W: Weight matrix
            phi_theta: Φ(θ) already in dense form
            mask: Binary mask
            freeze_active: Freeze active weights
            
        Returns:
            W_eff = M⊙W + (1-M)⊙Φ(θ)
        """
        W_eff = fused_effective_weight(W, phi_theta, mask)
        
        ctx.save_for_backward(mask)
        ctx.freeze_active = freeze_active
        
        return W_eff
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        mask, = ctx.saved_tensors
        
        # Gradient for phi_theta (pruned positions)
        grad_phi_theta = (1 - mask) * grad_output
        
        # Gradient for W
        if ctx.freeze_active:
            grad_W = None
        else:
            grad_W = mask * grad_output
        
        return grad_W, grad_phi_theta, None, None


# =============================================================================
# High-level wrappers
# =============================================================================

def effective_weight(
    W: torch.Tensor,
    theta: torch.Tensor,
    sparse_mask: SparseMask,
    freeze_active: bool = True,
) -> torch.Tensor:
    """Compute W_eff = M⊙W + (1-M)⊙Φ(θ) with autograd support.
    
    Args:
        W: Original weight matrix
        theta: Resurrection parameters (compact, p-dim)
        sparse_mask: The SparseMask
        freeze_active: Don't backprop through active weights
        
    Returns:
        Effective weights with proper gradient routing
    """
    return EffectiveWeightFunction.apply(
        W,
        theta,
        sparse_mask.mask,
        sparse_mask.pruned_indices,
        freeze_active,
    )


def effective_weight_dense(
    W: torch.Tensor,
    phi_theta: torch.Tensor,
    sparse_mask: SparseMask,
    freeze_active: bool = True,
) -> torch.Tensor:
    """Variant when Φ(θ) is already computed in dense form.
    
    Args:
        W: Original weight matrix
        phi_theta: Pre-computed Φ(θ)
        sparse_mask: The SparseMask
        freeze_active: Don't backprop through active weights
        
    Returns:
        Effective weights with proper gradient routing
    """
    return EffectiveWeightDense.apply(
        W,
        phi_theta,
        sparse_mask.mask,
        freeze_active,
    )


# =============================================================================
# RESU Forward Module
# =============================================================================

class RESUForward(nn.Module):
    """Module wrapper for RESU forward pass.
    
    Handles:
    - Effective weight computation
    - Gradient routing
    - Optional activation capture for Wanda
    """
    
    def __init__(
        self,
        weight: nn.Parameter,
        sparse_mask: SparseMask,
        resurrection_embedding,  # ResurrectionEmbedding
        capture_activations: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.sparse_mask = sparse_mask
        self.resurrection = resurrection_embedding
        self.capture_activations = capture_activations
        
        # For Wanda pruning
        self._last_input_norm: Optional[torch.Tensor] = None
    
    def get_effective_weight(self, freeze_active: bool = True) -> torch.Tensor:
        """Get W_eff for forward pass."""
        return effective_weight(
            self.weight,
            self.resurrection.theta,
            self.sparse_mask,
            freeze_active=freeze_active,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with effective weights."""
        raise NotImplementedError("Subclass should implement forward")
    
    @property
    def last_input_norm(self) -> Optional[torch.Tensor]:
        """Get last captured input norm (for Wanda)."""
        return self._last_input_norm
    
    def _capture_activation(self, x: torch.Tensor):
        """Capture activation statistics."""
        if self.capture_activations:
            # Compute L2 norm along feature dimension
            self._last_input_norm = x.norm(p=2, dim=0, keepdim=True)


class RESULinearForward(RESUForward):
    """RESU forward for linear layers."""
    
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        sparse_mask: SparseMask,
        resurrection_embedding,
        capture_activations: bool = False,
    ):
        super().__init__(weight, sparse_mask, resurrection_embedding, capture_activations)
        self.bias = bias
    
    def forward(self, x: torch.Tensor, freeze_active: bool = True) -> torch.Tensor:
        self._capture_activation(x)
        
        W_eff = self.get_effective_weight(freeze_active)
        return torch.nn.functional.linear(x, W_eff, self.bias)


# =============================================================================
# Gradient Analysis Utilities
# =============================================================================

def analyze_gradient_flow(
    grad_output: torch.Tensor,
    sparse_mask: SparseMask,
) -> dict:
    """Analyze gradient distribution between active and pruned subspaces.
    
    Useful for debugging and understanding RESU dynamics.
    """
    G_A, G_P = split_gradient(grad_output, sparse_mask.mask)
    
    return {
        "grad_active_norm": G_A.norm().item(),
        "grad_pruned_norm": G_P.norm().item(),
        "grad_active_mean": G_A[sparse_mask.mask.bool()].mean().item() if sparse_mask.n_active > 0 else 0,
        "grad_pruned_mean": G_P[~sparse_mask.mask.bool()].mean().item() if sparse_mask.n_pruned > 0 else 0,
        "grad_active_std": G_A[sparse_mask.mask.bool()].std().item() if sparse_mask.n_active > 0 else 0,
        "grad_pruned_std": G_P[~sparse_mask.mask.bool()].std().item() if sparse_mask.n_pruned > 0 else 0,
        "active_nonzero": (G_A.abs() > 1e-8).sum().item(),
        "pruned_nonzero": (G_P.abs() > 1e-8).sum().item(),
    }


def verify_gradient_isolation(
    grad_W: Optional[torch.Tensor],
    grad_theta: torch.Tensor,
    sparse_mask: SparseMask,
    tol: float = 1e-6,
) -> bool:
    """Verify that gradients are properly isolated.
    
    Proposition 1 from paper: ⟨G_A, G_P⟩_F = 0
    """
    if grad_W is None:
        return True  # Active frozen, nothing to check
    
    # Gradient for W should only be at active positions
    grad_W_at_pruned = grad_W.view(-1)[sparse_mask.pruned_indices]
    if grad_W_at_pruned.abs().max() > tol:
        return False
    
    return True


# =============================================================================
# Testing
# =============================================================================

def _test_effective_weight():
    """Test effective weight computation and gradients."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on {device}...")
    
    shape = (128, 64)
    
    # Create mask (~50% sparse)
    mask_tensor = (torch.rand(shape, device=device) > 0.5).float()
    sparse_mask = SparseMask(mask_tensor)
    
    # Create W and theta
    W = torch.randn(shape, device=device, requires_grad=True)
    theta = torch.randn(sparse_mask.n_pruned, device=device, requires_grad=True)
    
    print(f"W shape: {W.shape}, theta shape: {theta.shape}")
    print(f"Active: {sparse_mask.n_active}, Pruned: {sparse_mask.n_pruned}")
    
    # Test forward (freeze_active=True)
    W_eff = effective_weight(W, theta, sparse_mask, freeze_active=True)
    
    # Verify shape
    assert W_eff.shape == shape, f"Wrong shape: {W_eff.shape}"
    
    # Verify values: active should match W, pruned should match theta
    W_eff_flat = W_eff.view(-1)
    W_flat = W.view(-1)
    
    for idx in sparse_mask.active_indices[:10]:  # Check some active
        assert torch.allclose(W_eff_flat[idx], W_flat[idx]), "Active mismatch"
    
    for i, idx in enumerate(sparse_mask.pruned_indices[:10]):  # Check some pruned
        assert torch.allclose(W_eff_flat[idx], theta[i]), "Pruned mismatch"
    
    print("✓ Forward pass correct")
    
    # Test backward
    loss = W_eff.sum()
    loss.backward()
    
    # W should not have gradient (frozen)
    assert W.grad is None, "W should not have grad when frozen"
    
    # theta should have gradient
    assert theta.grad is not None, "theta should have grad"
    assert theta.grad.shape == theta.shape, "theta grad shape wrong"
    
    # theta grad should be all 1s (since loss = sum)
    assert torch.allclose(theta.grad, torch.ones_like(theta)), "theta grad should be 1s"
    
    print("✓ Backward pass with freeze_active=True correct")
    
    # Test with freeze_active=False
    W2 = torch.randn(shape, device=device, requires_grad=True)
    theta2 = torch.randn(sparse_mask.n_pruned, device=device, requires_grad=True)
    
    W_eff2 = effective_weight(W2, theta2, sparse_mask, freeze_active=False)
    loss2 = W_eff2.sum()
    loss2.backward()
    
    # Now W should have gradient
    assert W2.grad is not None, "W should have grad when not frozen"
    
    # W grad should be 1 at active positions, 0 at pruned
    W_grad_flat = W2.grad.view(-1)
    for idx in sparse_mask.active_indices[:10]:
        assert torch.allclose(W_grad_flat[idx], torch.tensor(1.0, device=device)), "Active grad should be 1"
    for idx in sparse_mask.pruned_indices[:10]:
        assert torch.allclose(W_grad_flat[idx], torch.tensor(0.0, device=device)), "Pruned grad should be 0"
    
    print("✓ Backward pass with freeze_active=False correct")
    
    # Test gradient isolation
    W3 = torch.randn(shape, device=device, requires_grad=True)
    theta3 = torch.randn(sparse_mask.n_pruned, device=device, requires_grad=True)
    
    W_eff3 = effective_weight(W3, theta3, sparse_mask, freeze_active=False)
    
    # More complex loss
    target = torch.randn(shape, device=device)
    loss3 = ((W_eff3 - target) ** 2).mean()
    loss3.backward()
    
    is_isolated = verify_gradient_isolation(W3.grad, theta3.grad, sparse_mask)
    assert is_isolated, "Gradient isolation violated"
    
    print("✓ Gradient isolation verified")
    
    # Test gradient analysis
    grad_out = torch.randn(shape, device=device)
    analysis = analyze_gradient_flow(grad_out, sparse_mask)
    print(f"  Gradient analysis: active_norm={analysis['grad_active_norm']:.4f}, "
          f"pruned_norm={analysis['grad_pruned_norm']:.4f}")
    
    print("\n✓ All effective_weight tests passed!")


if __name__ == "__main__":
    _test_effective_weight()
