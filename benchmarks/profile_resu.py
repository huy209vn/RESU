"""
Deep profiling of RESU implementation.

Identifies:
1. Memory allocations at each step
2. Timing bottlenecks
3. Kernel launch overhead
4. Where we violate the paper's "zero overhead" claim
"""

import torch
import torch.nn as nn
import sys
import os
import time
from typing import Dict, List
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_tensor_memory(tensor):
    """Get memory used by a tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def profile_allocations(layer: RESULinear, verbose=True):
    """Profile all memory allocations in a RESU layer."""

    allocations = {}

    # 1. Weight parameter
    allocations['weight'] = get_tensor_memory(layer.weight)

    # 2. Mask (minimal storage: only indices - int32)
    if layer._mask is not None:
        # Measure the ACTUAL stored tensor (int32 indices)
        allocations['mask_indices_int32'] = get_tensor_memory(layer._mask._indices)
        if verbose:
            storage_type = "active" if layer._mask._stores_active else "pruned"
            print(f"\n  [Mask stores {storage_type} indices as int32]")
        # Don't access .mask or .active_indices as they're computed on-demand!

    # 3. Resurrection storage
    if layer._resurrection is not None:
        allocations['theta'] = get_tensor_memory(layer._resurrection.theta)

        # Check if using dense buffer (DENSE mode)
        if hasattr(layer._resurrection, '_dense_buffer') and layer._resurrection._dense_buffer is not None:
            allocations['theta_dense_buffer'] = get_tensor_memory(layer._resurrection._dense_buffer._data)

    # 4. Selective states
    if layer._selective is not None:
        allocations['selective_m'] = get_tensor_memory(layer._selective._m) if hasattr(layer._selective, '_m') else 0.0
        allocations['selective_v'] = get_tensor_memory(layer._selective._v) if hasattr(layer._selective, '_v') else 0.0
        allocations['selective_C'] = get_tensor_memory(layer._selective._C) if hasattr(layer._selective, '_C') else 0.0

    # 5. Cached active weights (our optimization attempt)
    if layer._W_active_cached is not None:
        allocations['W_active_cached'] = get_tensor_memory(layer._W_active_cached)

    # Total
    allocations['total'] = sum(allocations.values())

    if verbose:
        print("\n" + "="*80)
        print("MEMORY ALLOCATIONS")
        print("="*80)
        for name, mb in allocations.items():
            if mb > 0:
                print(f"  {name:30s}: {mb:8.2f} MB")
        print("="*80)

    return allocations


def profile_forward_pass(layer: RESULinear, x: torch.Tensor, num_warmup=10, num_trials=100):
    """Profile forward pass timing with detailed breakdown."""

    print("\n" + "="*80)
    print("FORWARD PASS PROFILING")
    print("="*80)

    # Warmup
    for _ in range(num_warmup):
        _ = layer(x)

    torch.cuda.synchronize()

    # Time full forward
    start = time.perf_counter()
    for _ in range(num_trials):
        out = layer(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_trials * 1000
    print(f"Full forward pass: {avg_time_ms:.4f} ms")

    # Now break down components (RESU mode only)
    if layer._mode.name in ['RESU', 'QRESU', 'QRESU_SELECTIVE']:
        print("\nComponent breakdown:")

        # Check if using in-place mode (no resurrection object)
        if layer._resurrection is None:
            print("  IN-PLACE MODE: θ stored directly in W")
            print("  No scatter overhead - forward pass is just F.linear(x, W)")
        else:
            # OLD MODE: separate resurrection object
            # 1. Mask apply (extract W_active)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_trials):
                W_active = layer._mask.apply(layer.weight)
            torch.cuda.synchronize()
            end = time.perf_counter()
            mask_time = (end - start) / num_trials * 1000
            print(f"  Mask apply (W_active extraction): {mask_time:.4f} ms")

            # 2. Phi(theta) scatter
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_trials):
                phi_theta = layer._resurrection.phi()
            torch.cuda.synchronize()
            end = time.perf_counter()
            phi_time = (end - start) / num_trials * 1000
            print(f"  Φ(θ) scatter:                   {phi_time:.4f} ms")

            # 3. W_eff = W_active + phi_theta (MATERIALIZATION)
            W_active = layer._mask.apply(layer.weight)
            phi_theta = layer._resurrection.phi()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_trials):
                W_eff = W_active + phi_theta  # Dense addition
            torch.cuda.synchronize()
            end = time.perf_counter()
            add_time = (end - start) / num_trials * 1000
            print(f"  W_eff = W_active + Φ(θ):         {add_time:.4f} ms")

            # 4. Linear layer
            W_eff = W_active + phi_theta
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_trials):
                out = torch.nn.functional.linear(x, W_eff, layer.bias)
            torch.cuda.synchronize()
            end = time.perf_counter()
            linear_time = (end - start) / num_trials * 1000
            print(f"  F.linear(x, W_eff):              {linear_time:.4f} ms")

            overhead = avg_time_ms - linear_time
            print(f"\n  RESU overhead: {overhead:.4f} ms ({overhead/avg_time_ms*100:.1f}% of total)")
            print(f"  Breakdown: mask={mask_time:.4f} + phi={phi_time:.4f} + add={add_time:.4f} = {mask_time+phi_time+add_time:.4f} ms")

    print("="*80)
    return avg_time_ms


def profile_backward_pass(layer: RESULinear, x: torch.Tensor, num_warmup=10, num_trials=100):
    """Profile backward pass timing."""

    print("\n" + "="*80)
    print("BACKWARD PASS PROFILING")
    print("="*80)

    # Warmup
    for _ in range(num_warmup):
        out = layer(x)
        loss = out.sum()
        loss.backward()
        if layer.weight.grad is not None:
            layer.weight.grad = None

    torch.cuda.synchronize()

    # Time backward
    start = time.perf_counter()
    for _ in range(num_trials):
        out = layer(x)
        loss = out.sum()
        loss.backward()
        if layer.weight.grad is not None:
            layer.weight.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_trials * 1000
    print(f"Backward pass: {avg_time_ms:.4f} ms")
    print("="*80)

    return avg_time_ms


def profile_w_eff_materialization(layer: RESULinear):
    """Check if W_eff is materialized and retained."""

    print("\n" + "="*80)
    print("W_EFF MATERIALIZATION CHECK")
    print("="*80)

    if layer._mode.name not in ['RESU', 'QRESU', 'QRESU_SELECTIVE']:
        print("Not in RESU mode, skipping")
        return

    x = torch.randn(32, layer.in_features, device=layer.weight.device)

    # Check initial memory
    torch.cuda.synchronize()
    mem_before = get_gpu_memory()

    # Forward pass
    out = layer(x)

    torch.cuda.synchronize()
    mem_after_forward = get_gpu_memory()

    # Backward pass
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    mem_after_backward = get_gpu_memory()

    weight_mb = get_tensor_memory(layer.weight)

    forward_alloc = mem_after_forward - mem_before
    backward_alloc = mem_after_backward - mem_after_forward

    print(f"Weight size:               {weight_mb:.2f} MB")
    print(f"Memory allocated (forward):  {forward_alloc:.2f} MB")
    print(f"Memory allocated (backward): {backward_alloc:.2f} MB")

    # Check if W_eff is materialized
    expected_w_eff_size = weight_mb
    if forward_alloc > expected_w_eff_size * 0.5:
        print(f"\n⚠️  W_eff likely MATERIALIZED ({forward_alloc:.2f} MB allocated)")
        print(f"    Expected: ~0 MB (in-place computation)")
        print(f"    Actual: {forward_alloc:.2f} MB")
    else:
        print(f"\n✓ W_eff appears to be computed in-place")

    print("="*80)

    # Cleanup
    if layer.weight.grad is not None:
        layer.weight.grad = None


def compare_dense_vs_resu(
    in_features=512,
    out_features=256,
    batch_size=32,
    sparsity=0.5,
    device=torch.device("cuda"),
):
    """Compare dense vs RESU memory and speed."""

    print("\n" + "="*80)
    print("DENSE VS RESU COMPARISON")
    print("="*80)
    print(f"Layer: {in_features} → {out_features}")
    print(f"Batch: {batch_size}")
    print(f"Sparsity: {sparsity:.0%}")
    print("="*80)

    x = torch.randn(batch_size, in_features, device=device)

    # =========================================================================
    # DENSE BASELINE
    # =========================================================================
    print("\n[1] DENSE BASELINE")
    layer_dense = RESULinear(in_features, out_features, device=device)

    mem_dense = profile_allocations(layer_dense, verbose=False)
    fwd_dense = profile_forward_pass(layer_dense, x, num_trials=1000)

    print(f"\nDense:")
    print(f"  Memory: {mem_dense['total']:.2f} MB")
    print(f"  Forward: {fwd_dense:.4f} ms")

    # =========================================================================
    # SPARSE (no RESU)
    # =========================================================================
    print("\n[2] SPARSE (no RESU)")
    layer_sparse = RESULinear(in_features, out_features, device=device)
    layer_sparse.prune_by_magnitude(sparsity)

    mem_sparse = profile_allocations(layer_sparse, verbose=False)
    fwd_sparse = profile_forward_pass(layer_sparse, x, num_trials=1000)

    print(f"\nSparse:")
    print(f"  Memory: {mem_sparse['total']:.2f} MB")
    print(f"  Forward: {fwd_sparse:.4f} ms")

    # =========================================================================
    # RESU
    # =========================================================================
    print("\n[3] RESU")
    layer_resu = RESULinear(in_features, out_features, device=device)
    layer_resu.prune_by_magnitude(sparsity)
    layer_resu.enter_resu_mode(epsilon=0.1, use_selective=False)

    mem_resu = profile_allocations(layer_resu, verbose=True)
    fwd_resu = profile_forward_pass(layer_resu, x, num_trials=1000)
    profile_w_eff_materialization(layer_resu)

    print(f"\nRESU:")
    print(f"  Memory: {mem_resu['total']:.2f} MB")
    print(f"  Forward: {fwd_resu:.4f} ms")

    # =========================================================================
    # RESU-Selective
    # =========================================================================
    print("\n[4] RESU-SELECTIVE")
    layer_resu_sel = RESULinear(in_features, out_features, device=device)
    layer_resu_sel.prune_by_magnitude(sparsity)
    layer_resu_sel.enter_resu_mode(epsilon=0.1, use_selective=True)

    mem_resu_sel = profile_allocations(layer_resu_sel, verbose=True)
    fwd_resu_sel = profile_forward_pass(layer_resu_sel, x, num_trials=1000)

    print(f"\nRESU-Selective:")
    print(f"  Memory: {mem_resu_sel['total']:.2f} MB")
    print(f"  Forward: {fwd_resu_sel:.4f} ms")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Memory comparison
    print("\nMemory Usage:")
    print(f"  Dense:            {mem_dense['total']:8.2f} MB  (1.00x)")
    print(f"  Sparse:           {mem_sparse['total']:8.2f} MB  ({mem_sparse['total']/mem_dense['total']:.2f}x)")
    print(f"  RESU:             {mem_resu['total']:8.2f} MB  ({mem_resu['total']/mem_dense['total']:.2f}x)")
    print(f"  RESU-Selective:   {mem_resu_sel['total']:8.2f} MB  ({mem_resu_sel['total']/mem_dense['total']:.2f}x)")

    # What paper expects
    expected_resu_mem = mem_dense['total']  # Should be same as dense!
    actual_overhead = mem_resu['total'] - expected_resu_mem
    print(f"\n  Paper expects RESU: {expected_resu_mem:.2f} MB (same as dense)")
    print(f"  Actual overhead:    {actual_overhead:+.2f} MB")

    # Speed comparison
    print("\nForward Pass Speed:")
    print(f"  Dense:            {fwd_dense:8.4f} ms  (1.00x)")
    print(f"  Sparse:           {fwd_sparse:8.4f} ms  ({fwd_sparse/fwd_dense:.2f}x)")
    print(f"  RESU:             {fwd_resu:8.4f} ms  ({fwd_resu/fwd_dense:.2f}x)")
    print(f"  RESU-Selective:   {fwd_resu_sel:8.4f} ms  ({fwd_resu_sel/fwd_dense:.2f}x)")

    print("\n" + "="*80)

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\nANALYSIS")
    print("="*80)

    print("\n1. MEMORY VIOLATIONS:")
    if mem_resu['total'] > mem_dense['total'] * 1.1:
        print(f"   ❌ RESU uses {mem_resu['total'] - mem_dense['total']:.2f} MB MORE than dense")
        print(f"      Paper claims: 'no additional memory'")
       

    if 'theta' in mem_resu and mem_resu['theta'] > 0:
        print(f"\n   Found separate θ tensor: {mem_resu['theta']:.2f} MB")
        print(f"   This should be stored IN pruned positions of W")

    if any(k.startswith('selective_') for k in mem_resu_sel):
        selective_mem = sum(v for k, v in mem_resu_sel.items() if k.startswith('selective_'))
        print(f"\n   Selective states (m, v, C): {selective_mem:.2f} MB")
        print(f"   These should reuse optimizer state slots")

    print("\n2. SPEED VIOLATIONS:")
    if fwd_resu > fwd_dense * 1.5:
        print(f"   ❌ RESU is {fwd_resu/fwd_dense:.2f}x slower than dense")
        print(f"      Expected: ~1.0-1.2x (just scatter/gather overhead)")
        print(f"      Likely causes:")
        print(f"        - W_eff materialization")
        print(f"        - Extra kernel launches")
        print(f"        - Autograd graph overhead")

    print("\n" + "="*80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU (will be slower).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    compare_dense_vs_resu(
        in_features=512,
        out_features=256,
        batch_size=32,
        sparsity=0.5,
        device=device,
    )
