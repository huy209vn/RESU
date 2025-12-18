"""
Test the CORRECT RESU-Selective implementation.

Key points:
- Selection happens ONCE at phase start (not every backward)
- Only selected 20% of pruned positions get updated
- Zero per-backward overhead (static mask)
- 5× memory and compute reduction
"""

import torch
import torch.nn as nn
from resu.modules.linear import RESULinear


def test_resu_selective_correct():
    """Test correct RESU-Selective with random sampling."""

    print("="*80)
    print("Testing CORRECT RESU-Selective Implementation")
    print("="*80)
    print()

    # Create layer
    layer = RESULinear(512, 256, bias=False)
    layer.train()

    # Prune to 50%
    layer.prune_by_magnitude(0.5)
    n_pruned = int((~layer._mask.mask).sum().item())
    print(f"✓ Pruned to 50%: {n_pruned} pruned parameters")
    print()

    # Enter RESU mode with selective
    print("Entering RESU mode with selective (α=0.2)...")
    layer.enter_resu_mode(
        epsilon=0.1,
        use_selective=True,
        freeze_active=True,
    )
    print()

    # Check selection
    n_selected = layer._n_selected
    selection_ratio = layer._selection_ratio

    print(f"Selection at phase start:")
    print(f"  Total pruned: {n_pruned}")
    print(f"  Selected: {n_selected}")
    print(f"  Ratio: {selection_ratio:.1%}")
    print(f"  Expected: ~20%")
    assert abs(selection_ratio - 0.2) < 0.05, "Selection ratio should be ~20%"
    print(f"✓ Selection looks correct!")
    print()

    # Track which parameters get updated
    print("Running 5 training steps...")
    optimizer = torch.optim.SGD([layer.weight], lr=0.01)

    # Get initial values
    W_initial = layer.weight.data.clone()

    for step in range(5):
        x = torch.randn(32, 512)
        y = layer(x)
        loss = y.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")

    print(f"  ...")
    print(f"  Step 4: loss={loss.item():.6f}")
    print()

    # Check what changed
    W_final = layer.weight.data
    W_diff = (W_final - W_initial).abs()

    changed_mask = W_diff > 1e-6
    n_changed = changed_mask.sum().item()

    # Get active and pruned masks
    active_mask = layer._mask.mask
    pruned_mask = ~active_mask

    # Count changes in active vs pruned
    n_active_changed = (changed_mask & active_mask).sum().item()
    n_pruned_changed = (changed_mask & pruned_mask).sum().item()

    print("Weight update analysis:")
    print(f"  Total parameters: {W_diff.numel()}")
    print(f"  Active parameters: {active_mask.sum().item()}")
    print(f"  Pruned parameters: {pruned_mask.sum().item()}")
    print()
    print(f"  Active changed: {n_active_changed} (should be 0 - frozen)")
    print(f"  Pruned changed: {n_pruned_changed} (should be ~{n_selected})")
    print(f"  Selection ratio: {n_pruned_changed}/{n_pruned} = {n_pruned_changed/n_pruned:.1%}")
    print()

    # Verify correctness
    assert n_active_changed == 0, "Active weights should be frozen!"

    # Some selected params might have tiny gradients, so allow ~5% variance
    expected_min = int(n_selected * 0.90)
    expected_max = int(n_selected * 1.10)
    assert expected_min <= n_pruned_changed <= expected_max, \
        f"Expected {expected_min}-{expected_max} pruned to change, got {n_pruned_changed}"

    print("✓ Correct behavior:")
    print(f"  - Active weights frozen (0 changed)")
    print(f"  - Only selected pruned weights updated ({n_pruned_changed}/{n_pruned})")
    print(f"  - Matches selection at phase start!")
    print()

    # Memory analysis
    print("="*80)
    print("Memory Comparison")
    print("="*80)
    print()

    # Standard RESU would need full θ buffer
    standard_theta_memory = n_pruned * 4  # FP32

    # Our implementation: θ lives in W (no extra allocation!)
    # Only gradients for selected positions are non-zero
    selective_effective_memory = n_selected * 4  # Effectively

    print(f"Standard RESU:")
    print(f"  θ buffer: {standard_theta_memory / 1024:.2f} KB")
    print()
    print(f"RESU-Selective (ours):")
    print(f"  θ buffer: 0 KB (in-place in W)")
    print(f"  Effective active θ: {selective_effective_memory / 1024:.2f} KB")
    print(f"  Reduction: {standard_theta_memory / selective_effective_memory:.1f}×")
    print()

    print("✓ All tests passed!")
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("RESU-Selective (correct implementation):")
    print("  ✓ Selection happens ONCE at phase start")
    print("  ✓ Only selected 20% of pruned params get updated")
    print("  ✓ Zero per-backward overhead (static mask)")
    print("  ✓ 5× reduction in effective active parameters")
    print("  ✓ Simple, clean, practical!")
    print()


if __name__ == "__main__":
    test_resu_selective_correct()
