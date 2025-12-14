"""
Triton kernels for masked tensor operations.

These are the foundational operations for RESU:
- M ⊙ X (masked multiply)
- (1-M) ⊙ X (inverse masked multiply)
- M⊙W + (1-M)⊙V (fused effective weight)
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Core Masked Operations
# =============================================================================

@triton.jit
def masked_mul_kernel(
    X_ptr,          # Input tensor
    M_ptr,          # Binary mask (same shape as X)
    Out_ptr,        # Output tensor
    n_elements,     # Total elements
    BLOCK_SIZE: tl.constexpr,
):
    """Out = M ⊙ X
    
    Element-wise multiply with binary mask.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    out = x * m
    tl.store(Out_ptr + offsets, out, mask=mask)


@triton.jit
def inv_masked_mul_kernel(
    X_ptr,          # Input tensor
    M_ptr,          # Binary mask
    Out_ptr,        # Output tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Out = (1 - M) ⊙ X
    
    Element-wise multiply with inverted mask.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    out = x * (1.0 - m)
    tl.store(Out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_effective_weight_kernel(
    W_ptr,          # Original weights
    V_ptr,          # Resurrection values (Φ(θ) already embedded)
    M_ptr,          # Binary mask
    Out_ptr,        # Output: effective weights
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Out = M⊙W + (1-M)⊙V
    
    Compute effective weights in single pass.
    W: original weight matrix
    V: resurrection embedding Φ(θ) 
    M: binary mask (1 = active, 0 = pruned)
    
    For active positions (M=1): use W
    For pruned positions (M=0): use V (resurrected)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    w = tl.load(W_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    # Branchless: out = m * w + (1 - m) * v
    out = m * w + (1.0 - m) * v
    tl.store(Out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_effective_weight_inplace_kernel(
    W_ptr,          # Original weights (also output)
    V_ptr,          # Resurrection values
    M_ptr,          # Binary mask
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """W = M⊙W + (1-M)⊙V (in-place)
    
    Memory-efficient version that overwrites W.
    Use during commit phase.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    w = tl.load(W_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    out = m * w + (1.0 - m) * v
    tl.store(W_ptr + offsets, out, mask=mask)


# =============================================================================
# Gradient Splitting
# =============================================================================

@triton.jit
def split_gradient_kernel(
    G_ptr,          # Full gradient
    M_ptr,          # Binary mask
    G_A_ptr,        # Gradient for active (output)
    G_P_ptr,        # Gradient for pruned (output)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Split gradient by mask: G_A = M⊙G, G_P = (1-M)⊙G
    
    Used in backward pass to separate gradients for
    active weights and resurrection parameters.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    g = tl.load(G_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    g_a = g * m
    g_p = g * (1.0 - m)
    
    tl.store(G_A_ptr + offsets, g_a, mask=mask)
    tl.store(G_P_ptr + offsets, g_p, mask=mask)


@triton.jit 
def extract_pruned_gradient_kernel(
    G_ptr,          # Full gradient
    M_ptr,          # Binary mask
    G_P_ptr,        # Gradient for pruned only (output)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """G_P = (1-M)⊙G
    
    More efficient when we only need pruned gradients
    (which is the common case during RESU).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    g = tl.load(G_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    g_p = g * (1.0 - m)
    tl.store(G_P_ptr + offsets, g_p, mask=mask)


# =============================================================================
# Mask Application Utilities
# =============================================================================

@triton.jit
def apply_mask_kernel(
    X_ptr,          # Input/Output tensor (in-place)
    M_ptr,          # Binary mask
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """X = M ⊙ X (in-place)
    
    Zero out pruned positions.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    tl.store(X_ptr + offsets, x * m, mask=mask)


@triton.jit
def mask_where_kernel(
    X_ptr,          # Values where M=1
    Y_ptr,          # Values where M=0
    M_ptr,          # Binary mask
    Out_ptr,        # Output
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Out = where(M, X, Y)
    
    Select X where mask is 1, Y where mask is 0.
    Generalization of effective weight computation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + offsets, mask=mask, other=0.0)
    
    # Branchless select
    out = m * x + (1.0 - m) * y
    tl.store(Out_ptr + offsets, out, mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def _get_block_size(n: int) -> int:
    """Choose block size based on tensor size."""
    if n >= 1024 * 1024:
        return 1024
    elif n >= 64 * 1024:
        return 512
    elif n >= 4 * 1024:
        return 256
    else:
        return 128


def masked_mul(x: torch.Tensor, mask: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute M ⊙ X using Triton kernel.
    
    Args:
        x: Input tensor
        mask: Binary mask (same shape as x, float32)
        out: Optional output tensor
        
    Returns:
        Result tensor M ⊙ X
    """
    assert x.shape == mask.shape, f"Shape mismatch: {x.shape} vs {mask.shape}"
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    
    if out is None:
        out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Ensure contiguous
    x_flat = x.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(x.dtype)
    out_flat = out.view(-1)
    
    masked_mul_kernel[grid](
        x_flat, mask_flat, out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def inv_masked_mul(x: torch.Tensor, mask: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute (1-M) ⊙ X using Triton kernel."""
    assert x.shape == mask.shape
    assert x.is_cuda and mask.is_cuda
    
    if out is None:
        out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    x_flat = x.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(x.dtype)
    out_flat = out.view(-1)
    
    inv_masked_mul_kernel[grid](
        x_flat, mask_flat, out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def fused_effective_weight(
    W: torch.Tensor,
    V: torch.Tensor, 
    mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Compute W_eff = M⊙W + (1-M)⊙V
    
    Args:
        W: Original weights
        V: Resurrection embedding Φ(θ)
        mask: Binary mask (1=active, 0=pruned)
        out: Optional output tensor
        inplace: If True, write result to W
        
    Returns:
        Effective weights
    """
    assert W.shape == V.shape == mask.shape
    assert W.is_cuda and V.is_cuda and mask.is_cuda
    
    n_elements = W.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    W_flat = W.contiguous().view(-1)
    V_flat = V.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(W.dtype)
    
    if inplace:
        fused_effective_weight_inplace_kernel[grid](
            W_flat, V_flat, mask_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return W
    else:
        if out is None:
            out = torch.empty_like(W)
        out_flat = out.view(-1)
        
        fused_effective_weight_kernel[grid](
            W_flat, V_flat, mask_flat, out_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


def split_gradient(
    G: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split gradient by mask.
    
    Args:
        G: Full gradient tensor
        mask: Binary mask
        
    Returns:
        (G_A, G_P) where G_A = M⊙G and G_P = (1-M)⊙G
    """
    assert G.shape == mask.shape
    assert G.is_cuda and mask.is_cuda
    
    G_A = torch.empty_like(G)
    G_P = torch.empty_like(G)
    
    n_elements = G.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    G_flat = G.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(G.dtype)
    G_A_flat = G_A.view(-1)
    G_P_flat = G_P.view(-1)
    
    split_gradient_kernel[grid](
        G_flat, mask_flat, G_A_flat, G_P_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return G_A, G_P


def extract_pruned_gradient(
    G: torch.Tensor,
    mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract gradient for pruned positions: G_P = (1-M)⊙G"""
    assert G.shape == mask.shape
    assert G.is_cuda and mask.is_cuda
    
    if out is None:
        out = torch.empty_like(G)
    
    n_elements = G.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    G_flat = G.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(G.dtype)
    out_flat = out.view(-1)
    
    extract_pruned_gradient_kernel[grid](
        G_flat, mask_flat, out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def apply_mask_inplace(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask in-place: X = M ⊙ X"""
    assert x.shape == mask.shape
    assert x.is_cuda and mask.is_cuda
    
    n_elements = x.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    x_flat = x.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(x.dtype)
    
    apply_mask_kernel[grid](
        x_flat, mask_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x


def mask_where(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select x where mask=1, y where mask=0."""
    assert x.shape == y.shape == mask.shape
    assert x.is_cuda and y.is_cuda and mask.is_cuda
    
    if out is None:
        out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = _get_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1).to(x.dtype)
    out_flat = out.view(-1)
    
    mask_where_kernel[grid](
        x_flat, y_flat, mask_flat, out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# =============================================================================
# Testing / Verification
# =============================================================================

def _test_masked_ops():
    """Verify kernels against PyTorch reference."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    # Test shapes
    shapes = [(128, 128), (1024, 512), (4096, 4096), (768, 3072)]
    
    for shape in shapes:
        print(f"Testing shape {shape}...")
        
        W = torch.randn(shape, device=device)
        V = torch.randn(shape, device=device)
        M = (torch.rand(shape, device=device) > 0.5).float()
        G = torch.randn(shape, device=device)
        
        # Test masked_mul
        ref = M * W
        out = masked_mul(W, M)
        assert torch.allclose(ref, out, atol=1e-5), "masked_mul failed"
        
        # Test inv_masked_mul
        ref = (1 - M) * W
        out = inv_masked_mul(W, M)
        assert torch.allclose(ref, out, atol=1e-5), "inv_masked_mul failed"
        
        # Test fused_effective_weight
        ref = M * W + (1 - M) * V
        out = fused_effective_weight(W, V, M)
        assert torch.allclose(ref, out, atol=1e-5), "fused_effective_weight failed"
        
        # Test split_gradient
        ref_A = M * G
        ref_P = (1 - M) * G
        out_A, out_P = split_gradient(G, M)
        assert torch.allclose(ref_A, out_A, atol=1e-5), "split_gradient G_A failed"
        assert torch.allclose(ref_P, out_P, atol=1e-5), "split_gradient G_P failed"
        
        # Test extract_pruned_gradient
        ref = (1 - M) * G
        out = extract_pruned_gradient(G, M)
        assert torch.allclose(ref, out, atol=1e-5), "extract_pruned_gradient failed"
        
        # Test mask_where
        X = torch.randn(shape, device=device)
        Y = torch.randn(shape, device=device)
        ref = torch.where(M.bool(), X, Y)
        out = mask_where(X, Y, M)
        assert torch.allclose(ref, out, atol=1e-5), "mask_where failed"
        
        print(f"  ✓ All tests passed for shape {shape}")
    
    print("\n✓ All masked_ops tests passed!")


if __name__ == "__main__":
    _test_masked_ops()
