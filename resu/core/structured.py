"""
Structured Sparsity: N:M patterns for hardware acceleration.

N:M sparsity (e.g., 2:4) maintains N non-zero weights per M consecutive elements.
This structure enables hardware acceleration (tensor cores on modern GPUs).

Key insight: 2:4 (50% sparse) can be FASTER than dense on A100/H100!
"""

import torch
import torch.nn as nn
from typing import Tuple, Literal
from enum import Enum, auto


class StructuredPattern(Enum):
    """Supported structured sparsity patterns."""
    PATTERN_2_4 = (2, 4)      # 50% sparse: keep 2 per 4
    PATTERN_4_8 = (4, 8)      # 50% sparse: keep 4 per 8
    PATTERN_1_4 = (1, 4)      # 75% sparse: keep 1 per 4


# =============================================================================
# N:M Projection
# =============================================================================

def project_to_nm_structured(
    W: torch.Tensor,
    n: int,
    m: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project weight matrix to N:M structured sparsity.

    For each group of M consecutive elements along dimension dim,
    keeps the top-N by magnitude and zeros the rest.

    Args:
        W: Weight matrix (out_features, in_features)
        n: Number to keep per group
        m: Group size
        dim: Dimension to apply pattern (0=rows, 1=columns)

    Returns:
        W_structured: Structured sparse weights
        mask: Binary mask (1=kept, 0=pruned)

    Example:
        >>> W = torch.randn(512, 256)
        >>> W_24, mask = project_to_nm_structured(W, n=2, m=4, dim=1)
        >>> # Each row has exactly 2 non-zeros per 4 consecutive columns
    """
    assert n < m, f"N must be less than M (got n={n}, m={m})"
    assert n > 0, f"N must be positive (got n={n})"

    shape = W.shape
    device = W.device

    if dim == 1:
        # Apply along columns (in_features)
        assert shape[1] % m == 0, f"in_features={shape[1]} must be divisible by m={m}"
        W_reshaped = W.view(shape[0], shape[1] // m, m)
    else:
        # Apply along rows (out_features)
        assert shape[0] % m == 0, f"out_features={shape[0]} must be divisible by m={m}"
        W_reshaped = W.view(shape[0] // m, m, shape[1])

    # Find top-n per group
    if dim == 1:
        # Shape: (out_features, n_groups, m)
        topk_vals, topk_idx = torch.topk(W_reshaped.abs(), n, dim=2)
        # Create mask
        mask_reshaped = torch.zeros_like(W_reshaped)
        mask_reshaped.scatter_(2, topk_idx, 1.0)
    else:
        # Shape: (n_groups, m, in_features)
        topk_vals, topk_idx = torch.topk(W_reshaped.abs(), n, dim=1)
        mask_reshaped = torch.zeros_like(W_reshaped)
        mask_reshaped.scatter_(1, topk_idx, 1.0)

    # Reshape back
    mask = mask_reshaped.view(shape)
    W_structured = W * mask

    return W_structured, mask


def score_to_partial_nm_structured(
    scores: torch.Tensor,
    sparsity: float,
    n: int,
    m: int,
    dim: int = 1,
) -> torch.Tensor:
    """Convert Wanda++ scores to PARTIAL N:M structured mask (≤N per M).

    Uses importance scores to create partial structured sparsity for
    densification pipeline: prune aggressively, then RESU fills to exactly N:M.

    For each group of M:
    - Target: k = sparsity * numel pruned weights total
    - Keep top scoring weights, but enforce ≤N per group
    - Result: some groups have N active, some have <N (underfilled)

    Args:
        scores: Wanda++ importance scores (higher = more important)
        sparsity: Target overall sparsity (e.g., 0.7 for 70% sparse)
        n: Max number to keep per group
        m: Group size
        dim: Dimension to apply pattern (0=rows, 1=columns)

    Returns:
        mask: Binary mask (1=kept, 0=pruned) with partial N:M structure

    Example:
        >>> # After Wanda++ calibration
        >>> scores = pruner.stats[layer_idx]["weight"]["score"]
        >>> mask = score_to_partial_nm_structured(scores, sparsity=0.7, n=2, m=4)
        >>> # Result: ~70% sparse with ≤2 active per 4
        >>> # Groups: (x,x,0,0), (x,0,0,0), (0,0,0,0), etc.
    """
    assert n < m, f"N must be less than M (got n={n}, m={m})"
    assert n > 0, f"N must be positive (got n={n})"
    assert 0 <= sparsity <= 1, f"Sparsity must be in [0,1] (got {sparsity})"

    shape = scores.shape
    device = scores.device

    # Step 1: Global pruning to target sparsity
    flat_scores = scores.view(-1)
    k_prune = int(sparsity * flat_scores.numel())

    if k_prune > 0:
        threshold = torch.kthvalue(flat_scores, k_prune).values
        global_mask = (scores > threshold).float()
    else:
        global_mask = torch.ones_like(scores)

    # Step 2: Enforce ≤N per M groups
    if dim == 1:
        # Apply along columns (in_features)
        assert shape[1] % m == 0, f"in_features={shape[1]} must be divisible by m={m}"
        global_mask_reshaped = global_mask.view(shape[0], shape[1] // m, m)
        scores_reshaped = scores.view(shape[0], shape[1] // m, m)
    else:
        # Apply along rows (out_features)
        assert shape[0] % m == 0, f"out_features={shape[0]} must be divisible by m={m}"
        global_mask_reshaped = global_mask.view(shape[0] // m, m, shape[1])
        scores_reshaped = scores.view(shape[0] // m, m, shape[1])

    # VECTORIZED: Prune overfilled groups (no Python loops!)
    # Strategy: For each group, keep top-n by score among globally active positions

    if dim == 1:
        # Shape: (out_features, n_groups, m)
        group_dim = 2
    else:
        # Shape: (n_groups, m, in_features)
        group_dim = 1

    # Count active per group
    n_active_per_group = global_mask_reshaped.sum(dim=group_dim, keepdim=True)

    # Mask scores: -inf where globally pruned (so they won't be selected)
    scores_masked = torch.where(
        global_mask_reshaped.bool(),
        scores_reshaped,
        torch.tensor(float('-inf'), device=device, dtype=scores.dtype)
    )

    # Get top-n scores per group
    _, topk_idx = torch.topk(scores_masked, n, dim=group_dim)

    # Create mask from topk indices
    mask_reshaped = torch.zeros_like(global_mask_reshaped)
    mask_reshaped.scatter_(group_dim, topk_idx, 1.0)

    # For underfilled groups (n_active <= n), use original global_mask instead
    # (topk on underfilled groups may include -inf positions)
    underfilled = (n_active_per_group <= n)
    mask_reshaped = torch.where(underfilled, global_mask_reshaped, mask_reshaped)

    # Reshape back
    mask = mask_reshaped.view(shape)
    return mask


def score_to_nm_structured(
    scores: torch.Tensor,
    n: int,
    m: int,
    dim: int = 1,
) -> torch.Tensor:
    """Convert importance scores to N:M structured mask.

    Like project_to_nm_structured but uses importance scores instead of magnitudes.
    Useful for Wanda++, where scores indicate importance.

    Args:
        scores: Importance scores (higher = more important)
        n: Number to keep per group
        m: Group size
        dim: Dimension to apply pattern (0=rows, 1=columns)

    Returns:
        mask: Binary mask (1=kept, 0=pruned) with N:M structure

    Example:
        >>> # Wanda++ scores
        >>> scores = get_wanda_scores(model, dataloader)
        >>> mask = score_to_nm_structured(scores, n=2, m=4, dim=1)
        >>> # Apply to weights
        >>> W_structured = W * mask
    """
    assert n < m, f"N must be less than M (got n={n}, m={m})"
    assert n > 0, f"N must be positive (got n={n})"

    shape = scores.shape
    device = scores.device

    if dim == 1:
        # Apply along columns (in_features)
        assert shape[1] % m == 0, f"in_features={shape[1]} must be divisible by m={m}"
        scores_reshaped = scores.view(shape[0], shape[1] // m, m)
    else:
        # Apply along rows (out_features)
        assert shape[0] % m == 0, f"out_features={shape[0]} must be divisible by m={m}"
        scores_reshaped = scores.view(shape[0] // m, m, shape[1])

    # Find top-n per group BY SCORE (not magnitude)
    if dim == 1:
        # Shape: (out_features, n_groups, m)
        topk_vals, topk_idx = torch.topk(scores_reshaped, n, dim=2)
        # Create mask
        mask_reshaped = torch.zeros_like(scores_reshaped)
        mask_reshaped.scatter_(2, topk_idx, 1.0)
    else:
        # Shape: (n_groups, m, in_features)
        topk_vals, topk_idx = torch.topk(scores_reshaped, n, dim=1)
        mask_reshaped = torch.zeros_like(scores_reshaped)
        mask_reshaped.scatter_(1, topk_idx, 1.0)

    # Reshape back
    mask = mask_reshaped.view(shape)

    return mask


def compute_nm_fill_positions(
    mask: torch.Tensor,
    n: int,
    m: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute which positions need θ to reach EXACTLY N:M structure.

    Given a partial N:M mask (≤N per M), determines:
    - Which pruned positions should get θ to fill underfilled groups
    - The resulting exact N:M mask

    For each group of M:
    - Count active: k = sum(mask in group)
    - If k < N: need (N-k) θ positions in this group
    - Pick positions for θ (first available pruned slots)

    VECTORIZED: O(1) GPU operations, no Python loops!

    Args:
        mask: Current mask (1=active, 0=pruned), partial N:M structure
        n: Target number per group
        m: Group size
        dim: Dimension for N:M pattern

    Returns:
        theta_mask: Binary mask where θ should be initialized (1=θ position)
        final_mask: The exact N:M mask after filling (active + θ positions)

    Example:
        >>> # After partial 2:4 projection
        >>> theta_mask, final_mask = compute_nm_fill_positions(mask, n=2, m=4)
        >>> # theta_mask: where to put θ (fills underfilled groups)
        >>> # final_mask: exact 2:4 structure (original active + θ positions)
    """
    assert n < m, f"N must be less than M (got n={n}, m={m})"

    shape = mask.shape
    device = mask.device

    if dim == 1:
        assert shape[1] % m == 0, f"in_features={shape[1]} must be divisible by m={m}"
        mask_reshaped = mask.view(shape[0], shape[1] // m, m)
        group_dim = 2  # Groups along last dimension
    else:
        assert shape[0] % m == 0, f"out_features={shape[0]} must be divisible by m={m}"
        mask_reshaped = mask.view(shape[0] // m, m, shape[1])
        group_dim = 1  # Groups along middle dimension

    # VECTORIZED: Count active per group
    n_active_per_group = mask_reshaped.sum(dim=group_dim, keepdim=True)  # (..., 1, ...) or (..., ..., 1)

    # How many positions need filling per group
    n_needed_per_group = torch.clamp(n - n_active_per_group, min=0)  # (n - k) where k < n, else 0

    # Pruned positions (mask == 0)
    pruned_mask = (mask_reshaped == 0).float()

    # Create priority scores for pruned positions (cumsum gives position order within group)
    # We want to pick the FIRST n_needed pruned positions per group
    if dim == 1:
        # Shape: (out_features, n_groups, m) - cumsum along m dimension
        cumsum_pruned = pruned_mask.cumsum(dim=2)
    else:
        # Shape: (n_groups, m, in_features) - cumsum along m dimension
        cumsum_pruned = pruned_mask.cumsum(dim=1)

    # Select positions: pruned AND within needed count
    # cumsum_pruned[i,j,k] = how many pruned positions seen so far (including k)
    # Select if: position is pruned AND cumsum <= n_needed
    theta_mask_reshaped = (pruned_mask * (cumsum_pruned <= n_needed_per_group)).float()

    # Final mask = original active + theta positions
    final_mask_reshaped = mask_reshaped + theta_mask_reshaped

    # Clamp to handle any floating point issues
    final_mask_reshaped = torch.clamp(final_mask_reshaped, 0, 1)

    theta_mask = theta_mask_reshaped.view(shape)
    final_mask = final_mask_reshaped.view(shape)

    return theta_mask, final_mask


def commit_structured_nm(
    W_active: torch.Tensor,
    theta: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    m: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Commit RESU to EXACT N:M structure (fills underfilled groups).

    This is the final step of densification pipeline:
    1. Wanda++ → partial 2:4 (≤2 per 4, some underfilled)
    2. RESU → trains θ for pruned positions
    3. THIS → picks best N per M from {W_active ∪ θ}

    For each group of M:
    - Merge candidates: active weights + trained θ
    - Pick top-N by magnitude
    - Result: EXACTLY N:M structure!

    Args:
        W_active: Active weights (M⊙W) - sparse, active positions only
        theta: Trained θ values - in pruned positions (can overlap storage)
        mask: Current binary mask (1=active in W, 0=has θ)
        n: Number to keep per group
        m: Group size
        dim: Dimension for N:M pattern

    Returns:
        W_committed: Weights with EXACT N:M structure
        mask_new: New mask reflecting N:M structure

    Example:
        >>> # After RESU training on partial 2:4
        >>> W_committed, mask_new = commit_structured_nm(
        ...     W_active=layer.weight * mask,
        ...     theta=layer.theta_buffer,
        ...     mask=mask,
        ...     n=2, m=4
        ... )
        >>> # W_committed has EXACTLY 2:4 structure!
    """
    assert n < m, f"N must be less than M (got n={n}, m={m})"
    assert W_active.shape == theta.shape == mask.shape

    shape = W_active.shape
    device = W_active.device

    # Merge: use W_active where mask=1, theta where mask=0
    candidates = torch.where(mask.bool(), W_active, theta)

    if dim == 1:
        assert shape[1] % m == 0, f"in_features={shape[1]} must be divisible by m={m}"
        candidates_reshaped = candidates.view(shape[0], shape[1] // m, m)
    else:
        assert shape[0] % m == 0, f"out_features={shape[0]} must be divisible by m={m}"
        candidates_reshaped = candidates.view(shape[0] // m, m, shape[1])

    # Pick top-N per group by magnitude
    if dim == 1:
        _, topk_idx = torch.topk(candidates_reshaped.abs(), n, dim=2)
        mask_new_reshaped = torch.zeros_like(candidates_reshaped)
        mask_new_reshaped.scatter_(2, topk_idx, 1.0)
    else:
        _, topk_idx = torch.topk(candidates_reshaped.abs(), n, dim=1)
        mask_new_reshaped = torch.zeros_like(candidates_reshaped)
        mask_new_reshaped.scatter_(1, topk_idx, 1.0)

    # Apply mask to get final weights
    W_committed_reshaped = candidates_reshaped * mask_new_reshaped

    # Reshape back
    W_committed = W_committed_reshaped.view(shape)
    mask_new = mask_new_reshaped.view(shape)

    return W_committed, mask_new


def verify_nm_structure(
    W: torch.Tensor,
    n: int,
    m: int,
    dim: int = 1,
) -> Tuple[bool, float]:
    """Verify that weight matrix follows N:M structure.

    Args:
        W: Weight matrix
        n: Expected non-zeros per group
        m: Group size
        dim: Dimension to check

    Returns:
        is_valid: True if all groups have exactly n non-zeros
        actual_sparsity: Actual sparsity ratio
    """
    shape = W.shape

    if dim == 1:
        W_reshaped = W.view(shape[0], shape[1] // m, m)
        nnz_per_group = (W_reshaped != 0).sum(dim=2)
    else:
        W_reshaped = W.view(shape[0] // m, m, shape[1])
        nnz_per_group = (W_reshaped != 0).sum(dim=1)

    # Check if all groups have exactly n non-zeros
    is_valid = (nnz_per_group == n).all().item()

    # Compute actual sparsity
    total_nnz = (W != 0).sum().item()
    total_params = W.numel()
    actual_sparsity = 1.0 - (total_nnz / total_params)

    return is_valid, actual_sparsity


# =============================================================================
# RESU → 2:4 Pipeline
# =============================================================================

def resu_to_structured_pipeline(
    layer,
    initial_sparsity: float = 0.7,
    target_sparsity: float = 0.5,
    n: int = 2,
    m: int = 4,
    resu_steps: int = 100,
) -> dict:
    """Convert layer from unstructured to structured sparsity via RESU.

    Pipeline:
    1. Prune to initial_sparsity (e.g., 70%) unstructured
    2. RESU resurrection to target_sparsity (e.g., 50%)
    3. Project to N:M structured pattern (e.g., 2:4)

    Args:
        layer: RESULinear layer
        initial_sparsity: Starting sparsity (aggressive pruning)
        target_sparsity: Target after RESU (should match N/M ratio)
        n: N in N:M pattern
        m: M in N:M pattern
        resu_steps: Number of RESU training steps

    Returns:
        Stats dict with results
    """
    expected_sparsity = 1.0 - (n / m)
    assert abs(target_sparsity - expected_sparsity) < 0.01, \
        f"target_sparsity={target_sparsity} should match N:M ratio ({n}:{m} = {expected_sparsity:.1%})"

    print(f"RESU → {n}:{m} Structured Pipeline")
    print("=" * 60)
    print(f"Phase 1: Prune to {initial_sparsity:.0%} (unstructured)")
    print(f"Phase 2: RESU resurrection to {target_sparsity:.0%}")
    print(f"Phase 3: Project to {n}:{m} structured")
    print()

    # Phase 1: Aggressive unstructured pruning
    layer.prune_by_magnitude(initial_sparsity)
    phase1_sparsity = layer.sparsity
    print(f"✓ Phase 1: {phase1_sparsity:.1%} sparse")

    # Phase 2: RESU resurrection
    # (User should run actual training here)
    print(f"  Phase 2: Run RESU training for {resu_steps} steps")
    print(f"           (Revive important weights back to ~{target_sparsity:.0%})")

    # Phase 3: Project to structured
    W = layer.weight.data
    W_structured, mask = project_to_nm_structured(W, n, m, dim=1)

    # Update layer weights
    layer.weight.data.copy_(W_structured)

    # Verify structure
    is_valid, actual_sparsity = verify_nm_structure(W_structured, n, m, dim=1)

    print(f"✓ Phase 3: Projected to {n}:{m} structured")
    print(f"           Valid: {is_valid}, Sparsity: {actual_sparsity:.1%}")
    print()

    return {
        "initial_sparsity": phase1_sparsity,
        "target_sparsity": target_sparsity,
        "final_sparsity": actual_sparsity,
        "is_valid_structure": is_valid,
        "pattern": f"{n}:{m}",
    }


# =============================================================================
# Conversion for Inference
# =============================================================================

def convert_to_structured_inference(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
):
    """Convert all RESULinear layers to N:M structured for inference.

    Modifies model in-place.

    Args:
        model: Model with RESULinear layers
        n: N in N:M pattern
        m: M in N:M pattern
    """
    from ..modules.linear import RESULinear

    converted = 0
    for name, module in model.named_modules():
        if isinstance(module, RESULinear):
            # Exit RESU mode (merge θ → W)
            if module.mode.name != "DENSE":
                module.exit_resu_mode()

            # Project to structured
            W_structured, mask = project_to_nm_structured(
                module.weight.data, n, m, dim=1
            )
            module.weight.data.copy_(W_structured)

            # Verify
            is_valid, sparsity = verify_nm_structure(W_structured, n, m, dim=1)
            print(f"  {name}: {sparsity:.1%} sparse, valid={is_valid}")

            converted += 1

    print(f"\n✓ Converted {converted} layers to {n}:{m} structured")


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_structured_vs_dense(
    out_features: int = 4096,
    in_features: int = 4096,
    batch_size: int = 32,
    n: int = 2,
    m: int = 4,
    num_steps: int = 100,
):
    """Benchmark structured sparse vs dense matmul.

    Requires GPU with tensor core support (A100/H100) for speedup.
    """
    import time

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = "cuda"

    # Dense baseline
    W_dense = torch.randn(out_features, in_features, device=device)
    x = torch.randn(batch_size, in_features, device=device)

    # Warmup
    for _ in range(10):
        y = torch.nn.functional.linear(x, W_dense)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_steps):
        y = torch.nn.functional.linear(x, W_dense)
    torch.cuda.synchronize()
    end = time.perf_counter()

    dense_time_ms = (end - start) * 1000 / num_steps

    # Structured sparse
    W_structured, _ = project_to_nm_structured(W_dense, n, m, dim=1)

    # Warmup
    for _ in range(10):
        y = torch.nn.functional.linear(x, W_structured)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_steps):
        y = torch.nn.functional.linear(x, W_structured)
    torch.cuda.synchronize()
    end = time.perf_counter()

    structured_time_ms = (end - start) * 1000 / num_steps

    speedup = dense_time_ms / structured_time_ms

    print(f"{n}:{m} Structured vs Dense Benchmark")
    print("=" * 60)
    print(f"Layer: ({out_features}, {in_features})")
    print(f"Batch: {batch_size}")
    print()
    print(f"Dense:      {dense_time_ms:.4f} ms")
    print(f"Structured: {structured_time_ms:.4f} ms")
    print(f"Speedup:    {speedup:.2f}x")
    print()

    if speedup > 1.0:
        print(f"✓ Structured is {speedup:.2f}x FASTER than dense!")
    else:
        print(f"  Structured is {1/speedup:.2f}x slower than dense")
        print(f"  (Tensor core acceleration may not be enabled)")


# =============================================================================
# Testing
# =============================================================================

def _test_nm_projection():
    """Test N:M projection correctness."""
    print("Testing N:M projection...")

    # Test 2:4
    W = torch.randn(512, 256)
    W_24, mask = project_to_nm_structured(W, n=2, m=4, dim=1)

    # Verify structure
    is_valid, sparsity = verify_nm_structure(W_24, n=2, m=4, dim=1)
    assert is_valid, "2:4 structure invalid!"
    assert abs(sparsity - 0.5) < 0.01, f"Expected 50% sparsity, got {sparsity:.1%}"

    print(f"✓ 2:4 projection: {sparsity:.1%} sparse, valid={is_valid}")

    # Test 4:8
    W_48, mask = project_to_nm_structured(W, n=4, m=8, dim=1)
    is_valid, sparsity = verify_nm_structure(W_48, n=4, m=8, dim=1)
    assert is_valid, "4:8 structure invalid!"
    assert abs(sparsity - 0.5) < 0.01, f"Expected 50% sparsity, got {sparsity:.1%}"

    print(f"✓ 4:8 projection: {sparsity:.1%} sparse, valid={is_valid}")

    # Test 1:4
    W_14, mask = project_to_nm_structured(W, n=1, m=4, dim=1)
    is_valid, sparsity = verify_nm_structure(W_14, n=1, m=4, dim=1)
    assert is_valid, "1:4 structure invalid!"
    assert abs(sparsity - 0.75) < 0.01, f"Expected 75% sparsity, got {sparsity:.1%}"

    print(f"✓ 1:4 projection: {sparsity:.1%} sparse, valid={is_valid}")

    print("\n✓ All N:M projection tests passed!")


if __name__ == "__main__":
    _test_nm_projection()
    print()
    benchmark_structured_vs_dense()
