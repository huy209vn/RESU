"""
Real-world throughput benchmark: Tests actual training speed.

Compares:
1. Dense training (baseline)
2. Sparse training (masked weights)
3. RESU training (effective weights with cached optimization)
"""

import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear


def benchmark_forward_backward(
    layer,
    x,
    num_iters=100,
    warmup=10,
):
    """Time forward + backward passes."""

    # Warmup
    for _ in range(warmup):
        y = layer(x)
        loss = y.sum()
        loss.backward()
        if layer.weight.grad is not None:
            layer.weight.grad.zero_()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        y = layer(x)
        loss = y.sum()
        loss.backward()
        if layer.weight.grad is not None:
            layer.weight.grad.zero_()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / num_iters) * 1000
    throughput = num_iters / elapsed

    return avg_time_ms, throughput


def benchmark_real_throughput(
    in_features=4096,
    out_features=4096,
    batch_size=32,
    sparsity=0.5,
    num_iters=100,
    device=torch.device("cuda"),
):
    """Benchmark REAL training throughput."""

    print(f"\n{'='*70}")
    print(f"Real-World Throughput Benchmark")
    print(f"{'='*70}")
    print(f"Shape: ({in_features}, {out_features})")
    print(f"Batch size: {batch_size}")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Iterations: {num_iters}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Create input
    x = torch.randn(batch_size, in_features, device=device)

    # =========================================================================
    # 1. DENSE BASELINE
    # =========================================================================
    print("1. DENSE MODE (baseline)")
    layer_dense = RESULinear(in_features, out_features, device=device)

    time_dense, tput_dense = benchmark_forward_backward(layer_dense, x, num_iters)

    print(f"   Time per iter: {time_dense:.3f} ms")
    print(f"   Throughput: {tput_dense:.1f} iter/s")
    print(f"   Relative: 1.00x (baseline)\n")

    # =========================================================================
    # 2. SPARSE MODE (masked weights, no RESU)
    # =========================================================================
    print("2. SPARSE MODE (masked weights)")
    layer_sparse = RESULinear(in_features, out_features, device=device)
    layer_sparse.prune_by_magnitude(sparsity)

    time_sparse, tput_sparse = benchmark_forward_backward(layer_sparse, x, num_iters)
    speedup_sparse = time_dense / time_sparse

    print(f"   Time per iter: {time_sparse:.3f} ms")
    print(f"   Throughput: {tput_sparse:.1f} iter/s")
    print(f"   Relative: {speedup_sparse:.2f}x")
    if speedup_sparse < 1:
        print(f"   ⚠️  Slower due to masking overhead")
    print()

    # =========================================================================
    # 3. RESU MODE (OLD - recomputes M⊙W every pass)
    # =========================================================================
    print("3. RESU MODE (OLD - no caching)")
    layer_resu_old = RESULinear(in_features, out_features, device=device)
    layer_resu_old.prune_by_magnitude(sparsity)

    # Manually disable caching to simulate old behavior
    layer_resu_old.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)
    old_cached = layer_resu_old._W_active_cached
    layer_resu_old._W_active_cached = None  # Disable cache

    time_resu_old, tput_resu_old = benchmark_forward_backward(layer_resu_old, x, num_iters)
    speedup_resu_old = time_dense / time_resu_old

    layer_resu_old._W_active_cached = old_cached  # Restore

    print(f"   Time per iter: {time_resu_old:.3f} ms")
    print(f"   Throughput: {tput_resu_old:.1f} iter/s")
    print(f"   Relative: {speedup_resu_old:.2f}x")
    print(f"   ⚠️  Slow! Recomputes M⊙W every forward\n")

    # =========================================================================
    # 4. RESU MODE (NEW - cached active weights)
    # =========================================================================
    print("4. RESU MODE (NEW - with caching)")
    layer_resu_new = RESULinear(in_features, out_features, device=device)
    layer_resu_new.prune_by_magnitude(sparsity)
    layer_resu_new.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)

    time_resu_new, tput_resu_new = benchmark_forward_backward(layer_resu_new, x, num_iters)
    speedup_resu_new = time_dense / time_resu_new

    print(f"   Time per iter: {time_resu_new:.3f} ms")
    print(f"   Throughput: {tput_resu_new:.1f} iter/s")
    print(f"   Relative: {speedup_resu_new:.2f}x")

    improvement = (time_resu_old - time_resu_new) / time_resu_old * 100
    print(f"   ✓ {improvement:.1f}% faster than old RESU\n")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<25} {'Time (ms)':<12} {'Speedup':<10} {'Notes'}")
    print(f"{'-'*70}")
    print(f"{'Dense (baseline)':<25} {time_dense:>8.3f}     {1.00:>6.2f}x")
    print(f"{'Sparse (masked)':<25} {time_sparse:>8.3f}     {speedup_sparse:>6.2f}x")
    print(f"{'RESU (old)':<25} {time_resu_old:>8.3f}     {speedup_resu_old:>6.2f}x    {'← Slow!':<20}")
    print(f"{'RESU (new, cached)':<25} {time_resu_new:>8.3f}     {speedup_resu_new:>6.2f}x    {'← Our fix!':<20}")
    print(f"{'-'*70}\n")

    # Key insights
    overhead_eliminated = (1 - (time_resu_new / time_resu_old)) * 100
    print(f"💡 KEY INSIGHTS:")
    print(f"   • Cached optimization eliminates {overhead_eliminated:.1f}% of RESU overhead")
    print(f"   • RESU (new) is {time_resu_old/time_resu_new:.2f}x faster than RESU (old)")

    if speedup_resu_new >= 0.95:
        print(f"   • ✅ RESU is now competitive with sparse! ({speedup_resu_new:.2f}x vs baseline)")
    else:
        remaining_overhead = (1 - speedup_resu_new) * 100
        print(f"   • Remaining overhead: {remaining_overhead:.1f}% (Φ(θ) scatter + autograd)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Throughput benchmarks need GPU.")
        exit(1)

    device = torch.device("cuda")

    # Run benchmark
    benchmark_real_throughput(
        in_features=4096,
        out_features=4096,
        batch_size=32,
        sparsity=0.5,
        num_iters=100,
        device=device,
    )
