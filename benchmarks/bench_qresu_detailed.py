"""
Detailed QRESU memory comparison at different sparsity levels.

Compares:
- Dense FP32
- RESU FP32
- QRESU 4-bit
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear, RESUMode


def get_tensor_memory(tensor):
    """Get memory used by a tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def measure_detailed_memory(layer: RESULinear, mode_name: str):
    """Detailed memory measurement."""

    breakdown = {}

    # Weight (FP32) - only used in Dense/RESU, not in QRESU
    if layer._mode not in [RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
        breakdown['weight_fp32'] = get_tensor_memory(layer.weight)

    # Mask indices (int32)
    if layer._mask is not None:
        breakdown['mask_indices_int32'] = get_tensor_memory(layer._mask._indices)

    # QRESU: flat θ tensor
    if layer._theta is not None:
        breakdown['theta_flat_fp32'] = get_tensor_memory(layer._theta)

    # QRESU quantized W_A
    if layer._W_A_quantized is not None:
        breakdown['W_A_quantized_uint8'] = get_tensor_memory(layer._W_A_quantized)

        if layer._qscale is not None:
            breakdown['qscale'] = get_tensor_memory(layer._qscale)
        if layer._qzero is not None:
            breakdown['qzero'] = get_tensor_memory(layer._qzero)

    breakdown['total'] = sum(breakdown.values())

    return breakdown


def compare_modes(in_dim=512, out_dim=256, sparsity=0.5):
    """Compare memory usage across modes."""

    print("="*80)
    print(f"DETAILED QRESU MEMORY COMPARISON")
    print("="*80)
    print(f"Layer: {in_dim} → {out_dim}")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Total params: {in_dim * out_dim:,}")
    print("="*80)
    print()

    # 1. DENSE BASELINE
    layer_dense = RESULinear(in_dim, out_dim, bias=False)
    mem_dense = measure_detailed_memory(layer_dense, "Dense")

    print("[1] DENSE (FP32)")
    print(f"  Weight (FP32):     {mem_dense['weight_fp32']:.4f} MB")
    print(f"  Total:             {mem_dense['total']:.4f} MB")
    print()

    # 2. RESU (FP32)
    layer_resu = RESULinear(in_dim, out_dim, bias=False)
    layer_resu.prune_by_magnitude(sparsity)
    layer_resu.enter_resu_mode(epsilon=0.1)
    mem_resu = measure_detailed_memory(layer_resu, "RESU")

    print("[2] RESU (FP32 - Optimized)")
    print(f"  Weight (FP32):     {mem_resu['weight_fp32']:.4f} MB  [contains θ at pruned positions]")
    print(f"  Mask (int32):      {mem_resu.get('mask_indices_int32', 0):.4f} MB")
    print(f"  Total:             {mem_resu['total']:.4f} MB")
    print()

    # 3. QRESU (4-bit)
    layer_qresu = RESULinear(in_dim, out_dim, bias=False)
    layer_qresu.prune_by_magnitude(sparsity)
    layer_qresu.enter_qresu_mode(bits=4, epsilon=0.1, qscheme="per_channel")
    mem_qresu = measure_detailed_memory(layer_qresu, "QRESU")

    print("[3] QRESU (4-bit Quantized - OPTIMIZED)")
    print(f"  θ (flat FP32):     {mem_qresu.get('theta_flat_fp32', 0):.4f} MB  [resurrection params]")
    print(f"  W_A (uint8):       {mem_qresu.get('W_A_quantized_uint8', 0):.4f} MB  [4-bit quantized active weights]")
    print(f"  Mask (int32):      {mem_qresu.get('mask_indices_int32', 0):.4f} MB")
    print(f"  QParams:           {mem_qresu.get('qscale', 0) + mem_qresu.get('qzero', 0):.4f} MB")
    print(f"  Total:             {mem_qresu['total']:.4f} MB")
    print(f"  [No weight matrix stored - reconstructed on-the-fly!]")
    print()

    # COMPARISON
    print("="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\n{'Mode':<20} {'Memory (MB)':<15} {'vs Dense':<15} {'vs RESU':<15}")
    print("-"*65)

    dense_mem = mem_dense['total']
    resu_mem = mem_resu['total']
    qresu_mem = mem_qresu['total']

    print(f"{'Dense (FP32)':<20} {dense_mem:>8.4f}       {1.0:>8.2f}×")
    print(f"{'RESU (FP32)':<20} {resu_mem:>8.4f}       {resu_mem/dense_mem:>8.2f}×       {1.0:>8.2f}×")
    print(f"{'QRESU (4-bit)':<20} {qresu_mem:>8.4f}       {qresu_mem/dense_mem:>8.2f}×       {qresu_mem/resu_mem:>8.2f}×")

    print()
    print("SAVINGS:")
    savings_vs_dense = (1 - qresu_mem / dense_mem) * 100
    savings_vs_resu = (1 - qresu_mem / resu_mem) * 100
    print(f"  QRESU vs Dense:  {savings_vs_dense:>6.1f}% {'savings' if savings_vs_dense > 0 else 'overhead'}")
    print(f"  QRESU vs RESU:   {savings_vs_resu:>6.1f}% {'savings' if savings_vs_resu > 0 else 'overhead'}")

    print()
    print("BREAKDOWN:")
    n_params = in_dim * out_dim
    n_active = int((1 - sparsity) * n_params)
    n_pruned = n_params - n_active

    fp32_per_param = 4  # bytes
    int32_per_index = 4  # bytes
    uint8_per_param = 1  # byte (stores 4-bit in uint8)

    print(f"  Active params:   {n_active:>8,}  ({(1-sparsity)*100:.0f}%)")
    print(f"  Pruned params:   {n_pruned:>8,}  ({sparsity*100:.0f}%)")
    print()
    print(f"  θ (FP32 at pruned):        {n_pruned * fp32_per_param / 1024 / 1024:.4f} MB")
    print(f"  W_A (4-bit quantized):     {n_active * uint8_per_param / 1024 / 1024:.4f} MB  [stored as uint8]")
    print(f"  Mask indices (int32):      {n_pruned * int32_per_index / 1024 / 1024:.4f} MB")
    print()
    print("ANALYSIS:")
    print(f"  At {sparsity*100:.0f}% sparsity, QRESU stores:")
    print(f"    - θ parameters in FP32 at pruned positions")
    print(f"    - W_A quantized to 4-bit (87.5% smaller than FP32)")
    print(f"    - Mask indices in int32 (50% smaller than int64)")


if __name__ == "__main__":
    compare_modes(in_dim=512, out_dim=256, sparsity=0.5)
