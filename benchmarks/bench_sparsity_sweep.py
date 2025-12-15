"""
Benchmark RESU at different sparsity levels.

Tests the adaptive storage optimization:
- Low sparsity (10%): Store pruned indices
- Medium sparsity (50%): Store pruned indices
- High sparsity (90%): Store active indices (adaptive kicks in)
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear


def get_tensor_memory(tensor):
    """Get memory used by a tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def test_sparsity(in_dim, out_dim, sparsity):
    """Test memory usage at a specific sparsity level."""

    # Create layer
    layer = RESULinear(in_dim, out_dim, bias=False)

    # Prune to target sparsity
    layer.prune_by_magnitude(sparsity)

    # Enter RESU mode (this creates the mask with adaptive storage)
    layer.enter_resu_mode(epsilon=0.1)

    # Get mask stats
    mask = layer._mask
    weight_mem = get_tensor_memory(layer.weight)
    indices_mem = get_tensor_memory(mask._indices)
    total_mem = weight_mem + indices_mem

    # Expected memory with int64 (old)
    n_params = in_dim * out_dim
    n_pruned = int(sparsity * n_params)
    n_active = n_params - n_pruned
    n_stored = n_active if (n_active < n_pruned) else n_pruned
    int64_mem = n_stored * 8 / 1024 / 1024
    int32_mem = n_stored * 4 / 1024 / 1024

    return {
        'sparsity': sparsity,
        'n_params': n_params,
        'n_active': n_active,
        'n_pruned': n_pruned,
        'stores_active': mask._stores_active,
        'n_stored': len(mask._indices),
        'weight_mem': weight_mem,
        'indices_mem': indices_mem,
        'total_mem': total_mem,
        'int64_expected': int64_mem,
        'int32_expected': int32_mem,
    }


def main():
    in_dim = 512
    out_dim = 256
    sparsities = [0.10, 0.30, 0.50, 0.70, 0.90]

    print("="*80)
    print("ADAPTIVE STORAGE TEST: Different Sparsity Levels")
    print("="*80)
    print(f"Layer: {in_dim} → {out_dim}")
    print(f"Total parameters: {in_dim * out_dim:,}")
    print("="*80)
    print()

    results = []
    for sparsity in sparsities:
        result = test_sparsity(in_dim, out_dim, sparsity)
        results.append(result)

    # Print results
    print(f"{'Sparsity':>10} {'Active':>10} {'Pruned':>10} {'Stores':>10} {'Stored':>10} "
          f"{'Int32 (MB)':>12} {'Expected (MB)':>15} {'Savings':>10}")
    print("-"*80)

    for r in results:
        storage_type = "active" if r['stores_active'] else "pruned"
        savings = (r['int64_expected'] - r['indices_mem']) / r['int64_expected'] * 100 if r['int64_expected'] > 0 else 0
        print(f"{r['sparsity']:>9.0%} {r['n_active']:>10,} {r['n_pruned']:>10,} "
              f"{storage_type:>10} {r['n_stored']:>10,} "
              f"{r['indices_mem']:>12.4f} {r['int32_expected']:>15.4f} {savings:>9.0f}%")

    print()
    print("="*80)
    print("MEMORY BREAKDOWN BY SPARSITY")
    print("="*80)

    for r in results:
        storage_type = "active" if r['stores_active'] else "pruned"
        overhead = r['total_mem'] - r['weight_mem']
        ratio = r['total_mem'] / r['weight_mem']

        print(f"\nSparsity: {r['sparsity']:.0%} (stores {storage_type})")
        print(f"  Weight:       {r['weight_mem']:.4f} MB")
        print(f"  Indices:      {r['indices_mem']:.4f} MB")
        print(f"  Total:        {r['total_mem']:.4f} MB")
        print(f"  Overhead:     {overhead:.4f} MB ({ratio:.2f}×)")

        # Check if meeting paper's claim
        if overhead < 0.10:  # Allow 0.1 MB tolerance
            print(f"  ✓ MEETS paper's 'no additional memory' claim!")
        else:
            print(f"  ❌ Still {overhead:.2f} MB over dense baseline")


if __name__ == "__main__":
    main()
