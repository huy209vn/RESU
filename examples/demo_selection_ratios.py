"""
Demo: Using different selection ratios with RESU-Selective.

Shows how to configure the selection ratio from 1% to 100%.
"""

import torch
import torch.nn as nn
from resu.modules.linear import RESULinear


def demo_selection_ratios():
    """Demonstrate different selection ratios."""

    print("=" * 80)
    print("RESU-Selective: Custom Selection Ratios")
    print("=" * 80)
    print()

    # Create layer
    layer = RESULinear(512, 256, bias=False)
    layer.train()

    # Prune to 50%
    layer.prune_by_magnitude(0.5)
    n_pruned = int((~layer._mask.mask).sum().item())
    print(f"Pruned to 50%: {n_pruned} pruned parameters")
    print()

    # Test different selection ratios
    test_ratios = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

    print("Testing different selection ratios:")
    print("-" * 80)

    for ratio in test_ratios:
        # Create fresh layer for each test
        test_layer = RESULinear(512, 256, bias=False)
        test_layer.prune_by_magnitude(0.5)

        # Enter RESU mode with custom selection ratio
        test_layer.enter_resu_mode(
            epsilon=0.1,
            use_selective=True,
            selection_ratio=ratio,  # <-- KEY: Set custom ratio here!
            freeze_active=True,
        )

        n_selected = test_layer._n_selected
        actual_ratio = test_layer._selection_ratio

        print(f"  ratio={ratio:5.1%}  →  {n_selected:6d} params selected  ({actual_ratio:5.1%} actual)")

    print()
    print("=" * 80)
    print("Usage Examples:")
    print("=" * 80)
    print()

    print("1. Ultra-low memory (1% selection):")
    print("   layer.enter_resu_mode(selection_ratio=0.01)")
    print()

    print("2. Aggressive efficiency (5% selection):")
    print("   layer.enter_resu_mode(selection_ratio=0.05)")
    print()

    print("3. Balanced (20% selection, default):")
    print("   layer.enter_resu_mode(selection_ratio=0.20)")
    print("   # Or just: layer.enter_resu_mode()  (default is 0.2)")
    print()

    print("4. Conservative (50% selection):")
    print("   layer.enter_resu_mode(selection_ratio=0.50)")
    print()

    print("5. Full RESU (100% selection, no filtering):")
    print("   layer.enter_resu_mode(selection_ratio=1.0)")
    print("   # Or: layer.enter_resu_mode(use_selective=False)")
    print()

    print("=" * 80)
    print("Expected Speedups vs Standard RESU:")
    print("=" * 80)
    print()
    print("  selection_ratio | Memory Reduction | Compute Reduction")
    print("  ----------------+------------------+------------------")
    print("       1%         |      100×        |      100×")
    print("       5%         |       20×        |       20×")
    print("      10%         |       10×        |       10×")
    print("      20%         |        5×        |        5×")
    print("      50%         |        2×        |        2×")
    print("     100%         |        1×        |        1×  (standard RESU)")
    print()

    print("✓ Selection ratio is now fully configurable!")
    print()


if __name__ == "__main__":
    demo_selection_ratios()
