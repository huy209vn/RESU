"""
RESU-Selective: Intelligent update filtering for resurrection parameters.

Filters updates to coordinates with high signal quality using:
- Directional consistency: C_t = |m_t| / (v_t + δ)
- Magnitude-based screening: TopK by gradient magnitude
- Intersection-based selection: Only update high-quality candidates

When C_t ≈ 1, gradients consistently push in one direction (coherent signal).
When C_t ≈ 0, gradients oscillate (noise).
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto

from .mask import SparseMask
from .resurrection import ResurrectionEmbedding, StorageMode
from ..kernels.embedding import phi_inverse_gather


# =============================================================================
# Triton Kernels for Consistency Tracking
# =============================================================================

@triton.jit
def ema_update_kernel(
    m_ptr,          # Momentum EMA (in/out)
    v_ptr,          # Magnitude EMA (in/out)
    g_ptr,          # Current gradient
    beta,           # EMA coefficient
    n,              # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Update EMAs:
    m = β·m + (1-β)·g
    v = β·v + (1-β)·|g|
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    
    m_new = beta * m + (1.0 - beta) * g
    v_new = beta * v + (1.0 - beta) * tl.abs(g)
    
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)


@triton.jit
def compute_consistency_kernel(
    m_ptr,          # Momentum EMA
    v_ptr,          # Magnitude EMA
    c_ptr,          # Output: consistency
    delta,          # Stability constant
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """C = |m| / (v + δ)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    
    c = tl.abs(m) / (v + delta)
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def fused_ema_consistency_kernel(
    m_ptr, v_ptr, g_ptr, c_ptr,
    beta, delta,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """All-in-one: update EMAs and compute consistency."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    
    # Update EMAs
    m_new = beta * m + (1.0 - beta) * g
    v_new = beta * v + (1.0 - beta) * tl.abs(g)
    
    # Compute consistency
    c = tl.abs(m_new) / (v_new + delta)
    
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def selective_update_kernel(
    theta_ptr,          # θ parameters (in/out)
    grad_ptr,           # Gradient
    selection_ptr,      # Selection mask (0 or 1)
    consistency_ptr,    # Consistency weights
    lr,                 # Learning rate
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Selective update: θ ← θ - η · selection · C · grad"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    theta = tl.load(theta_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    sel = tl.load(selection_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(consistency_ptr + offsets, mask=mask, other=0.0)
    
    # Weighted update
    update = lr * sel * c * grad
    theta_new = theta - update
    
    tl.store(theta_ptr + offsets, theta_new, mask=mask)


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


def update_ema(
    m: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: float,
) -> None:
    """Update EMAs in-place using Triton kernel."""
    assert m.shape == v.shape == g.shape
    n = m.numel()
    
    BLOCK_SIZE = _get_block_size(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    ema_update_kernel[grid](
        m, v, g,
        beta,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def compute_consistency(
    m: torch.Tensor,
    v: torch.Tensor,
    delta: float = 1e-8,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute consistency C = |m| / (v + δ)."""
    if out is None:
        out = torch.empty_like(m)
    
    n = m.numel()
    BLOCK_SIZE = _get_block_size(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    compute_consistency_kernel[grid](
        m, v, out,
        delta,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def update_ema_and_consistency(
    m: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: float,
    delta: float = 1e-8,
    consistency_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused EMA update and consistency computation."""
    if consistency_out is None:
        consistency_out = torch.empty_like(m)

    # CPU fallback
    if not m.is_cuda:
        m.mul_(beta).add_(g, alpha=(1.0 - beta))
        v.mul_(beta).add_(g.abs(), alpha=(1.0 - beta))
        consistency_out.copy_(m.abs() / (v + delta))
        return consistency_out

    n = m.numel()
    BLOCK_SIZE = _get_block_size(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    fused_ema_consistency_kernel[grid](
        m, v, g, consistency_out,
        beta, delta,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return consistency_out


def selective_update(
    theta: torch.Tensor,
    grad: torch.Tensor,
    selection: torch.Tensor,
    consistency: torch.Tensor,
    lr: float,
) -> None:
    """Apply selective update with consistency weighting."""
    # CPU fallback
    if not theta.is_cuda:
        update = lr * selection * consistency * grad
        theta.sub_(update)
        return

    n = theta.numel()
    BLOCK_SIZE = _get_block_size(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    selective_update_kernel[grid](
        theta, grad, selection, consistency,
        lr,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# =============================================================================
# Selection Logic
# =============================================================================

@dataclass
class SelectionConfig:
    """Configuration for RESU-Selective."""
    beta: float = 0.9           # EMA coefficient
    delta: float = 1e-8         # Stability constant
    tau_stable: float = 0.5     # Consistency threshold
    k_screen_ratio: float = 0.5 # Fraction for magnitude screening
    k_select_ratio: float = 0.2 # Fraction for final selection
    
    def k_screen(self, p: int) -> int:
        return max(1, int(self.k_screen_ratio * p))
    
    def k_select(self, p: int) -> int:
        return max(1, int(self.k_select_ratio * p))


class SelectionResult(NamedTuple):
    """Result of coordinate selection."""
    mask: torch.Tensor          # Binary selection mask
    n_selected: int             # Number selected
    consistency: torch.Tensor   # Consistency scores
    p_mag: torch.Tensor         # Magnitude-screened mask
    p_con: torch.Tensor         # Consistency-filtered mask


def select_coordinates(
    grad_theta: torch.Tensor,
    consistency: torch.Tensor,
    config: SelectionConfig,
) -> SelectionResult:
    """Select coordinates for update using RESU-Selective algorithm.
    
    P_mag = TopK by |grad| (magnitude screening)
    P_con = {i : C[i] > τ} (consistency filtering)
    P_select = TopK of (P_mag ∩ P_con) by |grad|
    
    Args:
        grad_theta: Gradient for θ (p,)
        consistency: Consistency scores C (p,)
        config: Selection configuration
        
    Returns:
        SelectionResult with binary selection mask and diagnostics
    """
    p = len(grad_theta)
    device = grad_theta.device
    
    k_screen = config.k_screen(p)
    k_select = config.k_select(p)
    
    grad_abs = grad_theta.abs()
    
    # P_mag: TopK by gradient magnitude
    if k_screen >= p:
        p_mag = torch.ones(p, device=device, dtype=torch.bool)
    else:
        _, topk_indices = torch.topk(grad_abs, k_screen)
        p_mag = torch.zeros(p, device=device, dtype=torch.bool)
        p_mag[topk_indices] = True
    
    # P_con: Above consistency threshold
    p_con = consistency > config.tau_stable
    
    # Intersection
    intersection = p_mag & p_con
    n_intersection = intersection.sum().item()
    
    # P_select: TopK of intersection
    if n_intersection <= k_select:
        # Take all in intersection
        selection = intersection.float()
    else:
        # TopK within intersection
        intersection_indices = torch.nonzero(intersection, as_tuple=True)[0]
        intersection_grads = grad_abs[intersection_indices]
        _, topk_local = torch.topk(intersection_grads, k_select)
        selected_indices = intersection_indices[topk_local]
        
        selection = torch.zeros(p, device=device)
        selection[selected_indices] = 1.0
    
    return SelectionResult(
        mask=selection,
        n_selected=int(selection.sum().item()),
        consistency=consistency,
        p_mag=p_mag.float(),
        p_con=p_con.float(),
    )


# =============================================================================
# RESU-Selective Updater
# =============================================================================

class RESUSelective:
    """RESU-Selective: Intelligent update filtering for resurrection.
    
    Tracks gradient statistics and filters updates to coordinates
    with high signal quality (directional consistency).
    """
    
    def __init__(
        self,
        resurrection: ResurrectionEmbedding,
        config: Optional[SelectionConfig] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            resurrection: The ResurrectionEmbedding to update
            config: Selection configuration
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.resurrection = resurrection
        self.config = config or SelectionConfig()
        self.lr = lr
        self.weight_decay = weight_decay
        
        p = resurrection.p
        device = resurrection.device
        dtype = resurrection.dtype
        
        # EMA state
        self.m = torch.zeros(p, device=device, dtype=dtype)
        self.v = torch.zeros(p, device=device, dtype=dtype)
        self.consistency = torch.zeros(p, device=device, dtype=dtype)
        
        # Statistics
        self.step_count = 0
        self._last_selection: Optional[SelectionResult] = None
    
    @property
    def last_selection(self) -> Optional[SelectionResult]:
        """Last selection result (for debugging/analysis)."""
        return self._last_selection
    
    def step(self, grad_matrix: torch.Tensor) -> dict:
        """Perform one selective RESU update step.
        
        Args:
            grad_matrix: Full gradient matrix ∂L/∂W_eff
            
        Returns:
            Dictionary with step statistics
        """
        self.step_count += 1
        
        # Get gradient for θ
        grad_theta = self.resurrection.phi_inverse(grad_matrix)
        
        # Update EMAs and compute consistency
        self.consistency = update_ema_and_consistency(
            self.m, self.v, grad_theta,
            self.config.beta,
            self.config.delta,
            self.consistency,
        )
        
        # Select coordinates
        selection = select_coordinates(
            grad_theta,
            self.consistency,
            self.config,
        )
        self._last_selection = selection
        
        # Get current theta
        if self.resurrection.storage_mode == StorageMode.COMPACT:
            theta = self.resurrection._theta
        else:
            theta = self.resurrection._dense_buffer.get_theta_compact()
        
        # Apply weight decay
        if self.weight_decay > 0:
            theta.data.mul_(1 - self.lr * self.weight_decay)
        
        # Apply selective update
        selective_update(
            theta,
            grad_theta,
            selection.mask,
            self.consistency,
            self.lr,
        )
        
        # If dense mode, write back
        if self.resurrection.storage_mode == StorageMode.DENSE:
            self.resurrection._dense_buffer.set_theta_compact(theta)
        
        return {
            "n_selected": selection.n_selected,
            "selection_ratio": selection.n_selected / self.resurrection.p,
            "mean_consistency": self.consistency.mean().item(),
            "max_consistency": self.consistency.max().item(),
            "min_consistency": self.consistency.min().item(),
            "grad_norm": grad_theta.norm().item(),
            "p_mag_count": selection.p_mag.sum().item(),
            "p_con_count": selection.p_con.sum().item(),
        }
    
    def reset_state(self):
        """Reset EMA state (e.g., at start of new RESU phase)."""
        self.m.zero_()
        self.v.zero_()
        self.consistency.zero_()
        self.step_count = 0
        self._last_selection = None
    
    def state_dict(self) -> dict:
        """Serialize state."""
        return {
            "m": self.m.cpu(),
            "v": self.v.cpu(),
            "consistency": self.consistency.cpu(),
            "step_count": self.step_count,
            "config": {
                "beta": self.config.beta,
                "delta": self.config.delta,
                "tau_stable": self.config.tau_stable,
                "k_screen_ratio": self.config.k_screen_ratio,
                "k_select_ratio": self.config.k_select_ratio,
            },
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
    
    def load_state_dict(self, state: dict):
        """Load state."""
        device = self.resurrection.device
        dtype = self.resurrection.dtype
        
        self.m = state["m"].to(device, dtype)
        self.v = state["v"].to(device, dtype)
        self.consistency = state["consistency"].to(device, dtype)
        self.step_count = state["step_count"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        
        if "config" in state:
            cfg = state["config"]
            self.config = SelectionConfig(**cfg)


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_selection_quality(
    selection: SelectionResult,
    grad_theta: torch.Tensor,
) -> dict:
    """Analyze the quality of coordinate selection.
    
    Returns:
        Dictionary with selection quality metrics
    """
    selected_mask = selection.mask.bool()
    
    if selection.n_selected == 0:
        return {
            "coverage": 0.0,
            "mean_selected_grad": 0.0,
            "mean_unselected_grad": grad_theta.abs().mean().item(),
            "selection_grad_ratio": 0.0,
            "mean_selected_consistency": 0.0,
        }
    
    selected_grads = grad_theta.abs()[selected_mask]
    unselected_grads = grad_theta.abs()[~selected_mask]
    
    mean_sel = selected_grads.mean().item()
    mean_unsel = unselected_grads.mean().item() if len(unselected_grads) > 0 else 0
    
    return {
        "coverage": selection.n_selected / len(grad_theta),
        "mean_selected_grad": mean_sel,
        "mean_unselected_grad": mean_unsel,
        "selection_grad_ratio": mean_sel / (mean_unsel + 1e-8),
        "mean_selected_consistency": selection.consistency[selected_mask].mean().item(),
        "mean_unselected_consistency": selection.consistency[~selected_mask].mean().item() if (~selected_mask).any() else 0,
    }


# =============================================================================
# Testing
# =============================================================================

def _test_resu_selective():
    """Test RESU-Selective functionality."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on {device}...")
    
    # Create mask and resurrection embedding
    shape = (256, 128)
    mask_tensor = (torch.rand(shape, device=device) > 0.5).float()
    sparse_mask = SparseMask(mask_tensor)
    
    resurrection = ResurrectionEmbedding(sparse_mask, StorageMode.COMPACT, device)
    resurrection.initialize(active_std=1.0, epsilon=0.1)
    
    p = resurrection.p
    print(f"Created resurrection embedding with p={p}")
    
    # Test EMA update
    m = torch.zeros(p, device=device)
    v = torch.zeros(p, device=device)
    g = torch.randn(p, device=device)
    
    update_ema(m, v, g, beta=0.9)
    
    # Verify EMA update
    expected_m = 0.1 * g
    expected_v = 0.1 * g.abs()
    assert torch.allclose(m, expected_m, atol=1e-5), "EMA m update failed"
    assert torch.allclose(v, expected_v, atol=1e-5), "EMA v update failed"
    
    print("✓ EMA update correct")
    
    # Test consistency computation
    m2 = torch.randn(p, device=device)
    v2 = torch.rand(p, device=device) + 0.1
    
    c = compute_consistency(m2, v2, delta=1e-8)
    expected_c = m2.abs() / (v2 + 1e-8)
    assert torch.allclose(c, expected_c, atol=1e-5), "Consistency computation failed"
    
    print("✓ Consistency computation correct")
    
    # Test fused operation
    m3 = torch.zeros(p, device=device)
    v3 = torch.zeros(p, device=device)
    g3 = torch.randn(p, device=device)
    
    c3 = update_ema_and_consistency(m3, v3, g3, beta=0.9, delta=1e-8)
    
    expected_m3 = 0.1 * g3
    expected_v3 = 0.1 * g3.abs()
    expected_c3 = expected_m3.abs() / (expected_v3 + 1e-8)
    
    assert torch.allclose(m3, expected_m3, atol=1e-5), "Fused m failed"
    assert torch.allclose(v3, expected_v3, atol=1e-5), "Fused v failed"
    assert torch.allclose(c3, expected_c3, atol=1e-5), "Fused c failed"
    
    print("✓ Fused EMA+consistency correct")
    
    # Test selection
    config = SelectionConfig(
        beta=0.9,
        tau_stable=0.3,
        k_screen_ratio=0.5,
        k_select_ratio=0.2,
    )
    
    # Create gradient with varying consistency
    grad = torch.randn(p, device=device)
    
    # Simulate some steps to build up EMA
    m_sel = torch.zeros(p, device=device)
    v_sel = torch.zeros(p, device=device)
    
    for _ in range(10):
        # Consistent direction for first half
        g = torch.randn(p, device=device)
        g[:p//2] = g[:p//2].abs()  # Always positive
        update_ema(m_sel, v_sel, g, beta=0.9)
    
    c_sel = compute_consistency(m_sel, v_sel)
    
    selection = select_coordinates(grad, c_sel, config)
    
    print(f"  Selected: {selection.n_selected}/{p} ({100*selection.n_selected/p:.1f}%)")
    print(f"  P_mag: {int(selection.p_mag.sum())}, P_con: {int(selection.p_con.sum())}")
    
    assert selection.n_selected <= config.k_select(p) + 1, "Too many selected"
    assert selection.n_selected >= 0, "Negative selection"
    
    print("✓ Selection algorithm correct")
    
    # Test full RESUSelective
    selective = RESUSelective(
        resurrection,
        config=config,
        lr=0.01,
    )
    
    # Simulate training
    old_theta = resurrection.theta.clone()
    
    for step in range(20):
        grad_matrix = torch.randn(shape, device=device)
        stats = selective.step(grad_matrix)
        
        if step % 5 == 0:
            print(f"  Step {step}: selected={stats['n_selected']}, "
                  f"mean_C={stats['mean_consistency']:.3f}")
    
    new_theta = resurrection.theta
    assert not torch.allclose(old_theta, new_theta), "Theta should have changed"
    
    print("✓ RESUSelective training loop works")
    
    # Test state dict
    state = selective.state_dict()
    
    selective2 = RESUSelective(resurrection, config, lr=0.01)
    selective2.load_state_dict(state)
    
    assert selective.step_count == selective2.step_count, "Step count mismatch"
    assert torch.allclose(selective.m, selective2.m), "m mismatch"
    
    print("✓ State dict round-trip works")
    
    # Test analysis
    if selective.last_selection is not None:
        quality = analyze_selection_quality(
            selective.last_selection,
            selective.resurrection.phi_inverse(grad_matrix),
        )
        print(f"  Selection quality: coverage={quality['coverage']:.2%}, "
              f"grad_ratio={quality['selection_grad_ratio']:.2f}")
    
    print("\n✓ All RESU-Selective tests passed!")


if __name__ == "__main__":
    _test_resu_selective()
