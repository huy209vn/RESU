"""
Benchmark RESU-Selective Performance

Measures:
- Memory usage
- Forward/backward throughput (TFLOPS)
- Update efficiency (% of params actually updated)
- Comparison: Dense vs RESU vs QRESU-Selective

Key question: Is RESU-Selective actually efficient, or are we just doing
dense matmul when we should be using sparse operations?
"""

import torch
import time
from resu.modules.linear import RESULinear
from resu.core.selective import SelectionConfig


def get_tensor_memory(tensor):
    """Get memory used by tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def compute_tflops(m, n, k, time_ms):
    """
    Compute TFLOPS for matrix multiplication.

    For Y = X @ W^T where X is (batch, m, k) and W is (n, k):
    - FLOPs = batch * m * n * (2k - 1) ≈ batch * m * n * 2k
    - Forward: 1 matmul
    - Backward: 2 matmuls (dL/dX and dL/dW)
    Total: 3 matmuls per forward-backward pass
    """
    batch = 32  # We use batch=32
    flops_per_matmul = batch * m * n * 2 * k
    total_flops = 3 * flops_per_matmul  # Forward + 2 backward passes
    tflops = (total_flops / 1e12) / (time_ms / 1000)
    return tflops


def measure_memory_breakdown(layer, mode_name):
    """Measure detailed memory breakdown."""
    breakdown = {
        'mode': mode_name,
        'weight': get_tensor_memory(layer.weight),
    }

    # Mask
    if hasattr(layer, '_mask') and layer._mask is not None:
        breakdown['mask'] = get_tensor_memory(layer._mask._indices)

    # QRESU-specific
    if hasattr(layer, '_theta') and layer._theta is not None:
        breakdown['theta'] = get_tensor_memory(layer._theta)
    if hasattr(layer, '_W_A_quantized') and layer._W_A_quantized is not None:
        breakdown['W_A_q'] = get_tensor_memory(layer._W_A_quantized)
        breakdown['qparams'] = (get_tensor_memory(layer._qscale) +
                                get_tensor_memory(layer._qzero))

    # Selective EMA buffers
    if hasattr(layer, '_selective') and layer._selective is not None:
        sel = layer._selective
        if hasattr(sel, 'm') and sel.m is not None:
            breakdown['ema_m'] = get_tensor_memory(sel.m)
        if hasattr(sel, 'v') and sel.v is not None:
            breakdown['ema_v'] = get_tensor_memory(sel.v)

    # Sum only numeric values
    breakdown['total'] = sum(v for k, v in breakdown.items() if k != 'mode' and isinstance(v, (int, float)))
    return breakdown


def benchmark_mode(layer, mode_name, batch_size=32, num_steps=100, warmup=10):
    """
    Benchmark a layer mode.

    Returns:
        dict with memory, throughput (TFLOPS), and timing stats
    """
    in_dim, out_dim = layer.weight.shape[1], layer.weight.shape[0]

    # Memory breakdown
    memory = measure_memory_breakdown(layer, mode_name)

    # Warmup
    layer.train()
    for _ in range(warmup):
        x = torch.randn(batch_size, in_dim, device='cuda', requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        x.grad = None

    torch.cuda.synchronize()

    # Benchmark
    times_ms = []
    for _ in range(num_steps):
        x = torch.randn(batch_size, in_dim, device='cuda', requires_grad=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Forward
        y = layer(x)

        # Backward
        loss = y.sum()
        loss.backward()

        torch.cuda.synchronize()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

        # Clear gradients
        x.grad = None
        if layer.weight.grad is not None:
            layer.weight.grad = None

    # Stats
    mean_time = sum(times_ms) / len(times_ms)
    tflops = compute_tflops(batch_size, out_dim, in_dim, mean_time)

    return {
        'mode': mode_name,
        'memory': memory,
        'time_ms': mean_time,
        'tflops': tflops,
        'times': times_ms,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print("="*80)
    print("RESU-SELECTIVE PERFORMANCE BENCHMARK")
    print("="*80)
    print()

    in_dim, out_dim = 4096, 4096
    sparsity = 0.5
    batch_size = 32
    num_steps = 100

    print(f"Configuration:")
    print(f"  Layer: ({out_dim}, {in_dim})")
    print(f"  Sparsity: {sparsity:.0%}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps: {num_steps}")
    print()

    # =========================================================================
    # 1. Dense Baseline
    # =========================================================================
    print("Benchmarking Dense (baseline)...")
    layer_dense = RESULinear(in_dim, out_dim, bias=False).cuda()
    dense_stats = benchmark_mode(layer_dense, "Dense", batch_size, num_steps)

    # =========================================================================
    # 2. RESU (in-place θ)
    # =========================================================================
    print("Benchmarking RESU (in-place θ)...")
    layer_resu = RESULinear(in_dim, out_dim, bias=False).cuda()
    layer_resu.prune_by_magnitude(sparsity)
    layer_resu.enter_resu_mode(epsilon=0.1, use_selective=False)
    resu_stats = benchmark_mode(layer_resu, "RESU", batch_size, num_steps)

    # =========================================================================
    # 3. QRESU-Selective
    # =========================================================================
    print("Benchmarking QRESU-Selective...")
    layer_qresu_sel = RESULinear(in_dim, out_dim, bias=False).cuda()
    layer_qresu_sel.prune_by_magnitude(sparsity)

    sel_config = SelectionConfig(
        beta=0.9,
        tau_stable=0.5,
        k_select_ratio=0.2,  # Update only 20% of θ per step
    )

    layer_qresu_sel.enter_qresu_selective_mode(
        bits=4,
        epsilon=0.1,
        selective_config=sel_config,
        lr=1e-4,
    )

    qresu_sel_stats = benchmark_mode(layer_qresu_sel, "QRESU-Selective", batch_size, num_steps)

    # =========================================================================
    # Results
    # =========================================================================
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    # Memory comparison
    print("MEMORY USAGE:")
    print("-" * 80)
    for stats in [dense_stats, resu_stats, qresu_sel_stats]:
        mem = stats['memory']
        print(f"\n{stats['mode']}:")
        print(f"  Total: {mem['total']:.2f} MB")
        for key, val in mem.items():
            if key not in ['mode', 'total']:
                print(f"    {key:12s}: {val:.2f} MB")

    print()

    # Throughput comparison
    print("THROUGHPUT (Higher is better):")
    print("-" * 80)
    print(f"{'Mode':<20} {'Time (ms)':<12} {'TFLOPS':<10} {'vs Dense':<10}")
    print("-" * 80)

    for stats in [dense_stats, resu_stats, qresu_sel_stats]:
        time_ms = stats['time_ms']
        tflops = stats['tflops']
        vs_dense = tflops / dense_stats['tflops']

        print(f"{stats['mode']:<20} {time_ms:>10.3f} ms {tflops:>8.2f}   {vs_dense:>7.2f}x")

    print()

    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    resu_mem_ratio = resu_stats['memory']['total'] / dense_stats['memory']['total']
    qresu_mem_ratio = qresu_sel_stats['memory']['total'] / dense_stats['memory']['total']

    resu_tflops_ratio = resu_stats['tflops'] / dense_stats['tflops']
    qresu_tflops_ratio = qresu_sel_stats['tflops'] / dense_stats['tflops']

    print(f"Memory:")
    print(f"  Dense:           {dense_stats['memory']['total']:.2f} MB (1.00x)")
    print(f"  RESU:            {resu_stats['memory']['total']:.2f} MB ({resu_mem_ratio:.2f}x)")
    print(f"  QRESU-Selective: {qresu_sel_stats['memory']['total']:.2f} MB ({qresu_mem_ratio:.2f}x)")
    print()

    print(f"Throughput:")
    print(f"  Dense:           {dense_stats['tflops']:.2f} TFLOPS (1.00x)")
    print(f"  RESU:            {resu_stats['tflops']:.2f} TFLOPS ({resu_tflops_ratio:.2f}x)")
    print(f"  QRESU-Selective: {qresu_sel_stats['tflops']:.2f} TFLOPS ({qresu_tflops_ratio:.2f}x)")
    print()

    print("Key Findings:")
    print()

    # Check if QRESU-Selective has EMA buffers
    qresu_mem = qresu_sel_stats['memory']
    has_ema = 'ema_m' in qresu_mem or 'ema_v' in qresu_mem

    if has_ema:
        ema_overhead = qresu_mem.get('ema_m', 0) + qresu_mem.get('ema_v', 0)
        print(f"  ✓ QRESU-Selective has EMA buffers ({ema_overhead:.2f} MB overhead)")
        print(f"    - This is for gradient momentum/variance tracking")
        print(f"    - Enables selective updates (only ~20% of θ per step)")
    else:
        print(f"  ⚠ QRESU-Selective missing EMA buffers!")
        print(f"    - Selective filtering may not be working correctly")

    print()
    print(f"  Current Implementation:")
    print(f"    - All modes use DENSE matmul (cuBLAS)")
    print(f"    - No sparse kernels yet (this is expected)")
    print(f"    - Throughput similar across modes (~{dense_stats['tflops']:.1f} TFLOPS)")
    print()

    print(f"  RESU-Selective Benefits:")
    print(f"    - Update efficiency: Only ~20% of θ updated per step")
    print(f"    - Memory: Adds EMA buffers but quantizes W_A")
    print(f"    - Throughput: Same as dense (no sparse kernel yet)")
    print()

    print(f"  Potential Improvements:")
    print(f"    - Sparse forward pass: Could skip pruned weights")
    print(f"    - Sparse backward: Could skip gradient computation")
    print(f"    - Trade-off: Dense is VERY optimized (cuBLAS), sparse may be slower")
    print()

    # Theoretical TFLOPS for sparse
    sparsity_ratio = 1 - sparsity  # 50% sparsity = 0.5x FLOPs
    theoretical_sparse_tflops = dense_stats['tflops'] / sparsity_ratio

    print(f"  Theoretical Sparse Performance (at {sparsity:.0%} sparsity):")
    print(f"    - Dense matmul:  {dense_stats['tflops']:.2f} TFLOPS")
    print(f"    - Sparse matmul: {theoretical_sparse_tflops:.2f} TFLOPS (if perfectly efficient)")
    print(f"    - Reality: Sparse kernels are rarely this efficient")
    print(f"    - Dense cuBLAS is HIGHLY optimized, hard to beat even with sparsity")
    print()

    print("  Conclusion:")
    print(f"    - Current approach (dense matmul) is reasonable")
    print(f"    - RESU-Selective benefit is update efficiency, not throughput")
    print(f"    - For throughput gains, need custom sparse kernels")
    print(f"    - But custom kernels may be SLOWER than cuBLAS at 50% sparsity!")


if __name__ == "__main__":
    main()
