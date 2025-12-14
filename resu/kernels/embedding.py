"""
Triton kernels for Resurrection Embedding operations.

Φ: ℝᵖ → S_P     (embed θ into pruned subspace)
Φ⁻¹: S_P → ℝᵖ   (extract from pruned subspace)

The key insight: instead of storing θ as a separate p-dimensional vector
and doing scatter/gather, we can work directly with the full matrix
where pruned positions hold θ values.

Two modes:
1. Index-based: θ ∈ ℝᵖ is a compact vector, scatter/gather via indices
2. Dense-mode: θ lives directly in pruned positions of W-shaped tensor
"""

import torch
import triton
import triton.language as tl
from torch.autograd import Function
from typing import Optional, Tuple


# =============================================================================
# Index-based Φ and Φ⁻¹ 
# =============================================================================

@triton.jit
def phi_scatter_kernel(
    theta_ptr,          # Source: θ ∈ ℝᵖ (compact)
    indices_ptr,        # Pruned position indices (int64)
    out_ptr,            # Output: full matrix (zeros except at indices)
    p,                  # Number of pruned positions (len(theta))
    BLOCK_SIZE: tl.constexpr,
):
    """Φ(θ): Scatter θ values to pruned positions.
    
    out[indices[k]] = theta[k] for k in [0, p)
    
    This creates a sparse matrix with θ values at pruned coordinates.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    # Load theta values and their target indices
    theta_vals = tl.load(theta_ptr + offsets, mask=mask, other=0.0)
    target_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Scatter to output positions
    tl.store(out_ptr + target_indices, theta_vals, mask=mask)


@triton.jit
def phi_inverse_gather_kernel(
    matrix_ptr,         # Source: full gradient matrix
    indices_ptr,        # Pruned position indices
    out_ptr,            # Output: θ gradient ∈ ℝᵖ
    p,                  # Number of pruned positions
    BLOCK_SIZE: tl.constexpr,
):
    """Φ⁻¹(G): Gather gradient values from pruned positions.
    
    out[k] = matrix[indices[k]] for k in [0, p)
    
    Extracts the p-dimensional gradient for θ from full gradient matrix.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    # Load source indices
    source_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Gather from matrix
    values = tl.load(matrix_ptr + source_indices, mask=mask, other=0.0)
    
    # Store to compact output
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def phi_scatter_add_kernel(
    theta_ptr,          # Source: θ ∈ ℝᵖ
    indices_ptr,        # Pruned position indices
    out_ptr,            # Output: add to existing matrix
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """Add θ values to existing matrix at pruned positions.
    
    out[indices[k]] += theta[k]
    
    Useful for accumulating updates.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    theta_vals = tl.load(theta_ptr + offsets, mask=mask, other=0.0)
    target_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Atomic add for safety (though usually not needed if indices are unique)
    tl.atomic_add(out_ptr + target_indices, theta_vals, mask=mask)


# =============================================================================
# Fused Operations
# =============================================================================

@triton.jit
def fused_effective_weight_indexed_kernel(
    W_ptr,              # Original weights (flat)
    theta_ptr,          # Resurrection params θ ∈ ℝᵖ
    indices_ptr,        # Pruned position indices
    mask_ptr,           # Binary mask (float, flat)
    out_ptr,            # Output: effective weights
    n_total,            # Total elements in W
    p,                  # Number of pruned positions
    BLOCK_W: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Compute W_eff = M⊙W + (1-M)⊙Φ(θ) using indexed θ.
    
    Two-phase kernel:
    1. Copy M⊙W to output
    2. Scatter θ to pruned positions
    
    More memory efficient than pre-computing full Φ(θ).
    """
    pid = tl.program_id(0)
    
    # Phase 1: Copy active weights (M⊙W)
    # Process in blocks
    for block_start in range(0, n_total, BLOCK_W):
        offsets = block_start + tl.arange(0, BLOCK_W)
        valid = offsets < n_total
        
        w = tl.load(W_ptr + offsets, mask=valid, other=0.0)
        m = tl.load(mask_ptr + offsets, mask=valid, other=0.0)
        
        # Store masked weights
        tl.store(out_ptr + offsets, w * m, mask=valid)
    
    # Sync point (conceptual - each block handles this separately)
    
    # Phase 2: Scatter θ to pruned positions
    # Each program handles a subset of θ
    block_start = pid * BLOCK_P
    offsets = block_start + tl.arange(0, BLOCK_P)
    valid = offsets < p
    
    theta_vals = tl.load(theta_ptr + offsets, mask=valid, other=0.0)
    target_indices = tl.load(indices_ptr + offsets, mask=valid, other=0)
    
    # Write θ values (these positions have 0 from phase 1)
    tl.store(out_ptr + target_indices, theta_vals, mask=valid)


@triton.jit
def fused_backward_gather_kernel(
    grad_out_ptr,       # Full gradient (flat)
    indices_ptr,        # Pruned position indices
    grad_theta_ptr,     # Output: gradient for θ
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass: extract ∂L/∂θ from full gradient.
    
    Since W_eff[pruned] = θ, we have:
    ∂L/∂θ[k] = ∂L/∂W_eff[indices[k]]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    source_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    grad_vals = tl.load(grad_out_ptr + source_indices, mask=mask, other=0.0)
    
    tl.store(grad_theta_ptr + offsets, grad_vals, mask=mask)


# =============================================================================
# RESU Update Kernels (Fused with embedding)
# =============================================================================

@triton.jit
def resu_update_indexed_kernel(
    theta_ptr,          # θ parameters (in/out)
    grad_matrix_ptr,    # Full gradient matrix
    indices_ptr,        # Pruned position indices
    lr,                 # Learning rate
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """RESU update: θ ← θ - η·Φ⁻¹(G_P)
    
    Fuses gather and update.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    # Load current θ
    theta = tl.load(theta_ptr + offsets, mask=mask, other=0.0)
    
    # Gather gradient from matrix
    source_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    grad = tl.load(grad_matrix_ptr + source_indices, mask=mask, other=0.0)
    
    # Update
    theta_new = theta - lr * grad
    
    tl.store(theta_ptr + offsets, theta_new, mask=mask)


@triton.jit
def resu_update_momentum_indexed_kernel(
    theta_ptr,          # θ parameters (in/out)
    grad_matrix_ptr,    # Full gradient matrix
    indices_ptr,        # Pruned position indices
    momentum_ptr,       # Momentum buffer (in/out)
    lr,                 # Learning rate
    beta,               # Momentum coefficient
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """RESU update with momentum.
    
    m ← β·m + (1-β)·g
    θ ← θ - η·m
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    # Load
    theta = tl.load(theta_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(momentum_ptr + offsets, mask=mask, other=0.0)
    source_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    grad = tl.load(grad_matrix_ptr + source_indices, mask=mask, other=0.0)
    
    # Update momentum
    m_new = beta * m + (1.0 - beta) * grad
    
    # Update θ
    theta_new = theta - lr * m_new
    
    # Store
    tl.store(theta_ptr + offsets, theta_new, mask=mask)
    tl.store(momentum_ptr + offsets, m_new, mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _get_block_size(n: int) -> int:
    if n >= 1024 * 1024:
        return 1024
    elif n >= 64 * 1024:
        return 512
    elif n >= 4 * 1024:
        return 256
    else:
        return 128


def phi_scatter(
    theta: torch.Tensor,
    indices: torch.Tensor,
    shape: Tuple[int, ...],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Φ(θ): Scatter θ to create matrix with values at pruned positions.
    
    Args:
        theta: Compact θ vector (p,)
        indices: Flat indices of pruned positions (p,)
        shape: Target output shape
        out: Optional pre-allocated output (will be zeroed)
        
    Returns:
        Matrix of shape `shape` with θ at pruned positions, 0 elsewhere
    """
    assert theta.dim() == 1
    assert indices.dim() == 1
    assert len(theta) == len(indices)
    assert theta.is_cuda and indices.is_cuda
    
    p = len(theta)
    
    if out is None:
        out = torch.zeros(shape, dtype=theta.dtype, device=theta.device)
    else:
        out.zero_()
    
    out_flat = out.view(-1)
    assert out_flat.numel() >= indices.max().item() + 1, "Index out of bounds"
    
    BLOCK_SIZE = _get_block_size(p)
    grid = (triton.cdiv(p, BLOCK_SIZE),)
    
    phi_scatter_kernel[grid](
        theta, indices.to(torch.int64), out_flat,
        p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def phi_inverse_gather(
    matrix: torch.Tensor,
    indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Φ⁻¹(G): Gather values from matrix at pruned positions.
    
    Args:
        matrix: Full matrix (any shape, will be flattened)
        indices: Flat indices of pruned positions (p,)
        out: Optional pre-allocated output of shape (p,)
        
    Returns:
        Compact vector (p,) with values from pruned positions
    """
    assert indices.dim() == 1
    assert matrix.is_cuda and indices.is_cuda
    
    p = len(indices)
    matrix_flat = matrix.contiguous().view(-1)
    
    if out is None:
        out = torch.empty(p, dtype=matrix.dtype, device=matrix.device)
    
    BLOCK_SIZE = _get_block_size(p)
    grid = (triton.cdiv(p, BLOCK_SIZE),)
    
    phi_inverse_gather_kernel[grid](
        matrix_flat, indices.to(torch.int64), out,
        p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# =============================================================================
# Autograd-enabled Φ and Φ⁻¹
# =============================================================================

class PhiScatterFunction(Function):
    """Autograd function for Φ(θ): scatter with proper gradient support.

    Forward: θ (compact) → matrix (sparse)
    Backward: grad_matrix → grad_theta (via gather)
    """

    @staticmethod
    def forward(ctx, theta: torch.Tensor, indices: torch.Tensor, shape: Tuple[int, ...]):
        """
        Args:
            theta: Compact resurrection parameters (p,)
            indices: Pruned position indices (p,)
            shape: Target output shape

        Returns:
            Matrix with θ at pruned positions, zeros elsewhere
        """
        ctx.save_for_backward(indices)
        ctx.shape = shape
        return phi_scatter(theta, indices, shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Args:
            grad_output: Gradient w.r.t. output matrix

        Returns:
            grad_theta: Gradient w.r.t. theta (via gather)
            None, None: No gradients for indices and shape
        """
        indices, = ctx.saved_tensors
        # Backward of scatter is gather
        grad_theta = phi_inverse_gather(grad_output, indices)
        return grad_theta, None, None


def phi_scatter_grad(
    theta: torch.Tensor,
    indices: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """Autograd-enabled Φ(θ): scatter θ to pruned positions.

    This version supports gradient backpropagation to theta.
    Use this instead of phi_scatter() when you need gradients.

    Args:
        theta: Compact θ vector (p,)
        indices: Flat indices of pruned positions (p,)
        shape: Target output shape

    Returns:
        Matrix of shape `shape` with θ at pruned positions, 0 elsewhere
        Gradients flow: grad_output → grad_theta via gather
    """
    return PhiScatterFunction.apply(theta, indices, shape)


def resu_update_indexed(
    theta: torch.Tensor,
    grad_matrix: torch.Tensor,
    indices: torch.Tensor,
    lr: float,
) -> None:
    """In-place RESU update: θ ← θ - η·Φ⁻¹(G)
    
    Args:
        theta: Resurrection parameters (p,) - modified in-place
        grad_matrix: Full gradient matrix
        indices: Pruned position indices
        lr: Learning rate
    """
    assert theta.dim() == 1
    assert len(theta) == len(indices)
    
    p = len(theta)
    grad_flat = grad_matrix.contiguous().view(-1)
    
    BLOCK_SIZE = _get_block_size(p)
    grid = (triton.cdiv(p, BLOCK_SIZE),)
    
    resu_update_indexed_kernel[grid](
        theta, grad_flat, indices.to(torch.int64),
        lr,
        p,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def resu_update_with_momentum(
    theta: torch.Tensor,
    grad_matrix: torch.Tensor,
    indices: torch.Tensor,
    momentum: torch.Tensor,
    lr: float,
    beta: float = 0.9,
) -> None:
    """RESU update with momentum.
    
    Args:
        theta: Resurrection parameters (modified in-place)
        grad_matrix: Full gradient matrix
        indices: Pruned position indices
        momentum: Momentum buffer (modified in-place)
        lr: Learning rate
        beta: Momentum coefficient
    """
    assert theta.shape == momentum.shape
    assert len(theta) == len(indices)
    
    p = len(theta)
    grad_flat = grad_matrix.contiguous().view(-1)
    
    BLOCK_SIZE = _get_block_size(p)
    grid = (triton.cdiv(p, BLOCK_SIZE),)
    
    resu_update_momentum_indexed_kernel[grid](
        theta, grad_flat, indices.to(torch.int64), momentum,
        lr, beta,
        p,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# =============================================================================
# Utility: Compute indices from mask
# =============================================================================

def compute_pruned_indices(mask: torch.Tensor) -> torch.Tensor:
    """Compute flat indices of pruned positions from binary mask.
    
    Args:
        mask: Binary mask (1=active, 0=pruned)
        
    Returns:
        1D tensor of flat indices where mask == 0
    """
    mask_flat = mask.view(-1)
    pruned_indices = torch.nonzero(mask_flat == 0, as_tuple=True)[0]
    return pruned_indices.to(torch.int64)


def compute_active_indices(mask: torch.Tensor) -> torch.Tensor:
    """Compute flat indices of active positions from binary mask."""
    mask_flat = mask.view(-1)
    active_indices = torch.nonzero(mask_flat == 1, as_tuple=True)[0]
    return active_indices.to(torch.int64)


# =============================================================================
# Dense-mode Embedding (θ stored in-place)
# =============================================================================

class DenseResurrectionBuffer:
    """Manage θ stored directly in W-shaped tensor.
    
    Instead of compact θ ∈ ℝᵖ with scatter/gather,
    we store a full matrix where:
    - Active positions: don't care (will be masked)
    - Pruned positions: hold θ values
    
    This matches the paper's claim of zero memory overhead:
    θ lives in the memory already allocated for pruned weights.
    """
    
    def __init__(self, shape: Tuple[int, ...], mask: torch.Tensor, device: torch.device):
        """
        Args:
            shape: Shape of weight matrix
            mask: Binary mask (1=active, 0=pruned)
            device: CUDA device
        """
        self.shape = shape
        self.mask = mask.to(device)
        self.device = device
        
        # θ stored in full matrix form
        # Pruned positions will hold resurrection parameters
        self.theta_dense = torch.zeros(shape, device=device)
        
        # Precompute indices for fast access
        self._indices = compute_pruned_indices(self.mask)
    
    @property
    def n_pruned(self) -> int:
        return len(self._indices)
    
    def initialize(self, active_std: float, epsilon: float = 0.1):
        """Initialize θ ~ N(0, ε·σ_A) at pruned positions."""
        # Generate random values for pruned positions only
        p = self.n_pruned
        init_values = torch.randn(p, device=self.device) * (epsilon * active_std)
        
        # Scatter to dense form
        self.theta_dense.view(-1)[self._indices] = init_values
    
    def phi(self) -> torch.Tensor:
        """Return Φ(θ) - the dense embedding.
        
        Since θ is already in dense form, just return it.
        The mask will handle which positions matter.
        """
        return self.theta_dense
    
    def phi_inverse(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        """Φ⁻¹(G): Extract gradient for θ (compact form)."""
        return phi_inverse_gather(grad_matrix, self._indices)
    
    def get_theta_compact(self) -> torch.Tensor:
        """Get θ as compact vector (p,)."""
        return self.theta_dense.view(-1)[self._indices].clone()
    
    def set_theta_compact(self, theta: torch.Tensor):
        """Set θ from compact vector."""
        assert len(theta) == self.n_pruned
        if theta.device != self.device:
            theta = theta.to(self.device)
        with torch.no_grad():
            self.theta_dense.view(-1)[self._indices] = theta
    
    def update_mask(self, new_mask: torch.Tensor):
        """Update mask (e.g., after amnesty pruning)."""
        self.mask = new_mask.to(self.device)
        self._indices = compute_pruned_indices(new_mask)
        
        # Reset θ for new pruned positions
        self.theta_dense.zero_()


# =============================================================================
# Testing
# =============================================================================

def _test_embedding():
    """Verify embedding operations."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    shapes = [(128, 64), (512, 256), (1024, 1024)]
    
    for shape in shapes:
        print(f"Testing shape {shape}...")
        
        # Create mask with ~50% sparsity
        mask = (torch.rand(shape, device=device) > 0.5).float()
        indices = compute_pruned_indices(mask)
        p = len(indices)
        
        # Create θ
        theta = torch.randn(p, device=device)
        
        # Test phi_scatter
        phi_theta = phi_scatter(theta, indices, shape)
        # Verify: values at pruned positions should match theta
        gathered = phi_theta.view(-1)[indices]
        assert torch.allclose(theta, gathered), "phi_scatter failed"
        # Verify: active positions should be 0
        active_indices = compute_active_indices(mask)
        assert (phi_theta.view(-1)[active_indices] == 0).all(), "phi_scatter wrote to active positions"
        
        # Test phi_inverse_gather
        matrix = torch.randn(shape, device=device)
        gathered = phi_inverse_gather(matrix, indices)
        ref = matrix.view(-1)[indices]
        assert torch.allclose(ref, gathered), "phi_inverse_gather failed"
        
        # Test resu_update
        theta_test = theta.clone()
        grad = torch.randn(shape, device=device)
        lr = 0.01
        
        # Reference update
        grad_theta_ref = grad.view(-1)[indices]
        theta_ref = theta - lr * grad_theta_ref
        
        # Kernel update
        resu_update_indexed(theta_test, grad, indices, lr)
        
        assert torch.allclose(theta_ref, theta_test, atol=1e-5), "resu_update_indexed failed"
        
        # Test DenseResurrectionBuffer
        buffer = DenseResurrectionBuffer(shape, mask, device)
        buffer.initialize(active_std=1.0, epsilon=0.1)
        
        # Verify initialization at pruned positions only
        phi = buffer.phi()
        assert (phi.view(-1)[active_indices] == 0).all(), "Buffer init wrote to active"
        assert (phi.view(-1)[indices] != 0).any(), "Buffer init didn't write to pruned"
        
        # Test compact round-trip
        theta_compact = buffer.get_theta_compact()
        buffer.set_theta_compact(theta_compact * 2)
        theta_new = buffer.get_theta_compact()
        assert torch.allclose(theta_compact * 2, theta_new), "Compact round-trip failed"
        
        print(f"  ✓ All tests passed for shape {shape}")
    
    print("\n✓ All embedding tests passed!")


if __name__ == "__main__":
    _test_embedding()
