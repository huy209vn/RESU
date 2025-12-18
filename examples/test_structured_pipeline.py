"""
Test: Wanda++ → Partial 2:4 → RESU-Structured → 2:4 Training

This tests the CORRECT structured densification flow:
1. Wanda++ → 70% sparse (unstructured)
2. Partial 2:4 projection (≤2 per 4, some underfilled)
3. RESU-Structured: fills ONLY underfilled groups to exactly 2:4
4. Training happens WITH exact 2:4 structure (tensor core accelerated!)
5. Result: 2:4 throughout, not just at the end
"""

import torch
import torch.nn as nn
from resu.modules.linear import RESULinear
from resu.core.mask import SparseMask
from resu.core.structured import (
    score_to_partial_nm_structured,
    compute_nm_fill_positions,
    verify_nm_structure,
)


def test_structured_pipeline():
    """Test structured pipeline with 2:4 throughout training."""
    print("=" * 70)
    print("Test: RESU-Structured (2:4 throughout training)")
    print("=" * 70)
    print()

    # Create layer
    torch.manual_seed(42)
    layer = RESULinear(512, 256, bias=False)
    print(f"Layer: {layer.in_features} → {layer.out_features}")
    print()

    # Step 1: Simulate Wanda++ scores
    print("Step 1: Generate importance scores (simulating Wanda++)...")
    scores = layer.weight.data.abs()
    print()

    # Step 2: Partial 2:4 projection (70% sparse, ≤2 per 4)
    print("Step 2: Partial 2:4 projection (70% sparse, ≤2 per 4)...")
    mask = score_to_partial_nm_structured(scores, sparsity=0.7, n=2, m=4, dim=1)

    # Apply mask to layer
    with torch.no_grad():
        layer.weight.data *= mask

    # Create SparseMask
    pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
    sparse_mask = SparseMask(pruned_indices, mask.shape, device=layer.weight.device)
    layer._mask = sparse_mask
    layer._mode = layer._mode  # Keep current mode

    actual_sparsity = 1.0 - mask.sum().item() / mask.numel()
    print(f"  Actual sparsity: {actual_sparsity:.1%}")

    # Analyze group distribution BEFORE RESU
    mask_reshaped = mask.view(mask.shape[0], mask.shape[1] // 4, 4)
    nnz_per_group = mask_reshaped.sum(dim=2)
    print("  Group distribution (BEFORE RESU):")
    for k in range(3):
        count = (nnz_per_group == k).sum().item()
        pct = count / nnz_per_group.numel() * 100
        print(f"    {k} active: {count} ({pct:.1f}%)")
    print()

    # Step 3: Enter RESU-Structured mode (fills to exact 2:4)
    print("Step 3: Enter RESU-Structured mode...")
    layer.enter_resu_mode_structured(n=2, m=4, dim=1, epsilon=0.1)
    print()

    # Verify EXACT 2:4 structure NOW (before training!)
    is_valid_before, sparsity_before = verify_nm_structure(layer.weight.data, n=2, m=4, dim=1)
    print(f"  BEFORE training:")
    print(f"    Valid 2:4 structure: {is_valid_before}")
    print(f"    Sparsity: {sparsity_before:.1%}")
    assert is_valid_before, "Should be exact 2:4 BEFORE training!"
    print()

    # Step 4: Train with exact 2:4 structure
    print("Step 4: Training WITH exact 2:4 structure...")
    optimizer = torch.optim.SGD([layer.weight], lr=0.01)

    for step in range(100):
        x = torch.randn(32, 512)
        y = layer(x)
        loss = y.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify 2:4 maintained during training
        if step % 25 == 0:
            is_valid, sparsity = verify_nm_structure(layer.weight.data, n=2, m=4, dim=1)
            print(f"  Step {step}: loss={loss.item():.6f}, 2:4 valid={is_valid}, sparsity={sparsity:.1%}")
            assert is_valid, f"2:4 structure lost at step {step}!"
    print()

    # Step 5: Exit RESU (just cleanup, already 2:4)
    print("Step 5: Exit RESU mode (already exact 2:4, just cleanup)...")
    layer.exit_resu_mode(commit=True)

    # Final verification
    is_valid_final, sparsity_final = verify_nm_structure(layer.weight.data, n=2, m=4, dim=1)
    print(f"  Final: valid={is_valid_final}, sparsity={sparsity_final:.1%}")
    print()

    # Final group distribution
    mask_final = (layer.weight.data != 0).float()
    mask_final_reshaped = mask_final.view(mask_final.shape[0], mask_final.shape[1] // 4, 4)
    nnz_final = mask_final_reshaped.sum(dim=2)
    print("  Final group distribution:")
    for k in range(5):
        count = (nnz_final == k).sum().item()
        if count > 0:
            pct = count / nnz_final.numel() * 100
            print(f"    {k} active: {count} ({pct:.1f}%)")
    print()

    print("=" * 70)
    print("SUCCESS! 2:4 structure maintained throughout training!")
    print("=" * 70)
    print()
    print("Key difference from old approach:")
    print("  OLD: Dense training → commit to 2:4 at end")
    print("  NEW: Fill to 2:4 → 2:4 training throughout → 2:4 inference")
    print()
    print("This enables tensor core acceleration during TRAINING!")


if __name__ == "__main__":
    test_structured_pipeline()
