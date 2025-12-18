"""
Comprehensive RESU Benchmark Suite

Benchmarks ALL modes with TFLOPS, memory, and throughput:
1. Dense (baseline)
2. RESU (unstructured, all pruned positions)
3. RESU-Selective (unstructured, 20% of pruned)
4. RESU-Structured (2:4 throughout training)
5. QRESU (quantized W_A + FP32 θ)
6. QRESU-Selective (quantized + selective)

Metrics:
- Forward TFLOPS
- Backward TFLOPS
- Memory (peak allocation)
- Throughput (samples/sec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

# RESU imports
from resu.modules.linear import RESULinear, RESUMode
from resu.core.mask import SparseMask
from resu.core.structured import (
    score_to_partial_nm_structured,
    verify_nm_structure,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    out_features: int = 4096
    in_features: int = 4096
    batch_size: int = 32
    num_warmup: int = 10
    num_iterations: int = 100
    sparsity: float = 0.5
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    mode: str
    sparsity: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    forward_tflops: float
    backward_tflops: float
    memory_mb: float
    throughput_samples_sec: float
    extra_info: Dict = None


def compute_flops(out_features: int, in_features: int, batch_size: int) -> int:
    """Compute FLOPs for a linear layer forward pass."""
    # y = xW^T + b
    # FLOPs = 2 * batch * out * in (multiply-add)
    return 2 * batch_size * out_features * in_features


@contextmanager
def measure_memory():
    """Context manager to measure peak GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    """Get peak memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_mode(
    mode_name: str,
    layer: RESULinear,
    config: BenchmarkConfig,
    setup_fn=None,
) -> BenchmarkResult:
    """Benchmark a specific mode."""
    device = config.device

    # Setup layer for this mode
    if setup_fn:
        setup_fn(layer)

    # Create input
    x = torch.randn(config.batch_size, config.in_features, device=device, dtype=config.dtype)

    # Warmup
    for _ in range(config.num_warmup):
        y = layer(x)
        loss = y.sum()
        loss.backward()
        layer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark forward
    forward_times = []
    for _ in range(config.num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        y = layer(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

    # Benchmark backward
    backward_times = []
    for _ in range(config.num_iterations):
        y = layer(x)
        loss = y.sum()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)

        layer.zero_grad()

    # Get peak memory
    memory_mb = get_peak_memory_mb()

    # Calculate metrics
    forward_time_ms = sum(forward_times) / len(forward_times) * 1000
    backward_time_ms = sum(backward_times) / len(backward_times) * 1000
    total_time_ms = forward_time_ms + backward_time_ms

    flops = compute_flops(config.out_features, config.in_features, config.batch_size)
    forward_tflops = (flops / 1e12) / (forward_time_ms / 1000)
    backward_tflops = (flops / 1e12) / (backward_time_ms / 1000)  # Approximate

    throughput = config.batch_size / (total_time_ms / 1000)

    # Get actual sparsity
    actual_sparsity = 1.0 - (layer.weight.data != 0).float().mean().item()

    return BenchmarkResult(
        mode=mode_name,
        sparsity=actual_sparsity,
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        total_time_ms=total_time_ms,
        forward_tflops=forward_tflops,
        backward_tflops=backward_tflops,
        memory_mb=memory_mb,
        throughput_samples_sec=throughput,
    )


def run_all_benchmarks(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run benchmarks for all modes."""
    results = []
    device = config.device

    print(f"\n{'='*80}")
    print(f"RESU Comprehensive Benchmark")
    print(f"{'='*80}")
    print(f"Layer: {config.out_features} x {config.in_features}")
    print(f"Batch: {config.batch_size}")
    print(f"Target sparsity: {config.sparsity:.0%}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # 1. Dense baseline
    print("Benchmarking: Dense (baseline)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)
    results.append(benchmark_mode("Dense", layer, config))
    del layer
    gc.collect()

    # 2. RESU (unstructured, all pruned)
    print("Benchmarking: RESU (unstructured)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)

    def setup_resu(l):
        # Magnitude pruning
        with torch.no_grad():
            threshold = torch.quantile(l.weight.data.abs().flatten(), config.sparsity)
            mask = (l.weight.data.abs() > threshold).float()
            l.weight.data *= mask
            pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
            l._mask = SparseMask(pruned_indices, mask.shape, device=l.device)
        l.enter_resu_mode(epsilon=0.1, use_selective=False, freeze_active=True)

    results.append(benchmark_mode("RESU", layer, config, setup_resu))
    layer.exit_resu_mode()
    del layer
    gc.collect()

    # 3. RESU-Selective (unstructured, 20% of pruned)
    print("Benchmarking: RESU-Selective (20%)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)

    def setup_resu_selective(l):
        with torch.no_grad():
            threshold = torch.quantile(l.weight.data.abs().flatten(), config.sparsity)
            mask = (l.weight.data.abs() > threshold).float()
            l.weight.data *= mask
            pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
            l._mask = SparseMask(pruned_indices, mask.shape, device=l.device)
        l.enter_resu_mode(epsilon=0.1, use_selective=True, selection_ratio=0.2, freeze_active=True)

    results.append(benchmark_mode("RESU-Selective", layer, config, setup_resu_selective))
    layer.exit_resu_mode()
    del layer
    gc.collect()

    # 4. RESU-Structured (2:4)
    print("Benchmarking: RESU-Structured (2:4)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)

    def setup_resu_structured(l):
        with torch.no_grad():
            # Partial 2:4 projection
            scores = l.weight.data.abs()
            mask = score_to_partial_nm_structured(scores, sparsity=0.7, n=2, m=4, dim=1)
            l.weight.data *= mask
            pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
            l._mask = SparseMask(pruned_indices, mask.shape, device=l.device)
        l.enter_resu_mode_structured(n=2, m=4, dim=1, epsilon=0.1)

    results.append(benchmark_mode("RESU-Structured 2:4", layer, config, setup_resu_structured))
    layer.exit_resu_mode()
    del layer
    gc.collect()

    # 5. QRESU (quantized)
    print("Benchmarking: QRESU (4-bit quantized)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)

    def setup_qresu(l):
        with torch.no_grad():
            threshold = torch.quantile(l.weight.data.abs().flatten(), config.sparsity)
            mask = (l.weight.data.abs() > threshold).float()
            l.weight.data *= mask
            pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
            l._mask = SparseMask(pruned_indices, mask.shape, device=l.device)
        l.enter_qresu_mode(bits=4, epsilon=0.1)

    results.append(benchmark_mode("QRESU (4-bit)", layer, config, setup_qresu))
    layer.exit_qresu_mode()
    del layer
    gc.collect()

    # 6. QRESU-Selective (quantized + selective)
    print("Benchmarking: QRESU-Selective (4-bit + 20%)...")
    layer = RESULinear(config.in_features, config.out_features, device=device, dtype=config.dtype)

    def setup_qresu_selective(l):
        with torch.no_grad():
            threshold = torch.quantile(l.weight.data.abs().flatten(), config.sparsity)
            mask = (l.weight.data.abs() > threshold).float()
            l.weight.data *= mask
            pruned_indices = (~mask.bool()).flatten().nonzero(as_tuple=True)[0]
            l._mask = SparseMask(pruned_indices, mask.shape, device=l.device)
        l.enter_qresu_selective_mode(bits=4, epsilon=0.1, selection_ratio=0.2)

    results.append(benchmark_mode("QRESU-Selective", layer, config, setup_qresu_selective))
    layer.exit_qresu_mode()
    del layer
    gc.collect()

    return results


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results as a table."""
    print(f"\n{'='*100}")
    print(f"{'Mode':<25} {'Sparsity':>10} {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'TFLOPS':>10} {'Mem (MB)':>10} {'Samples/s':>12}")
    print(f"{'='*100}")

    dense_tflops = None
    for r in results:
        if r.mode == "Dense":
            dense_tflops = r.forward_tflops

        # Calculate relative performance
        rel_perf = f"({r.forward_tflops/dense_tflops:.2f}x)" if dense_tflops else ""

        print(f"{r.mode:<25} {r.sparsity:>9.1%} {r.forward_time_ms:>10.3f} {r.backward_time_ms:>10.3f} "
              f"{r.forward_tflops:>10.1f} {r.memory_mb:>10.1f} {r.throughput_samples_sec:>12.1f}")

    print(f"{'='*100}")

    # Summary
    print("\nSummary (relative to Dense):")
    print("-" * 60)
    for r in results:
        if r.mode != "Dense" and dense_tflops:
            speedup = r.forward_tflops / dense_tflops
            print(f"  {r.mode:<25}: {speedup:.2f}x TFLOPS")


def benchmark_sparsity_sweep(sparsities: List[float] = [0.5, 0.7, 0.9]):
    """Run benchmarks across different sparsity levels."""
    all_results = {}

    for sparsity in sparsities:
        print(f"\n\n{'#'*80}")
        print(f"# Sparsity: {sparsity:.0%}")
        print(f"{'#'*80}")

        config = BenchmarkConfig(sparsity=sparsity)

        if not torch.cuda.is_available():
            config.device = "cpu"
            config.num_iterations = 20
            print("WARNING: CUDA not available, using CPU (results will be slower)")

        results = run_all_benchmarks(config)
        print_results(results)
        all_results[sparsity] = results

    return all_results


def main():
    """Main benchmark entry point."""
    print("=" * 80)
    print("RESU COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print()
    print("Modes benchmarked:")
    print("  1. Dense (baseline)")
    print("  2. RESU (unstructured, all pruned positions)")
    print("  3. RESU-Selective (unstructured, 20% of pruned)")
    print("  4. RESU-Structured (2:4 throughout training)")
    print("  5. QRESU (4-bit quantized W_A + FP32 θ)")
    print("  6. QRESU-Selective (quantized + selective)")
    print()

    # Quick benchmark at 50% sparsity
    config = BenchmarkConfig(
        out_features=4096,
        in_features=4096,
        batch_size=32,
        sparsity=0.5,
    )

    if not torch.cuda.is_available():
        config.device = "cpu"
        config.num_iterations = 20
        print("WARNING: CUDA not available, using CPU")

    results = run_all_benchmarks(config)
    print_results(results)

    # Optionally run sparsity sweep
    print("\n" + "=" * 80)
    print("Run sparsity sweep? (50%, 70%, 90%)")
    print("=" * 80)
    # Uncomment to run sweep:
    # benchmark_sparsity_sweep([0.5, 0.7, 0.9])


if __name__ == "__main__":
    main()
