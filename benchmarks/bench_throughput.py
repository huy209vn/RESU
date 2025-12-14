"""
Throughput benchmarks for RESU.

Measures:
- Forward pass time
- Backward pass time
- RESU update time
- Comparison with dense and standard sparse training
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear
from resu.core.mask import SparseMask


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    mean_time_ms: float
    std_time_ms: float
    throughput_samples_per_sec: float
    memory_mb: float


def benchmark_forward_pass(
    layer: nn.Module,
    batch_size: int,
    in_features: int,
    num_iterations: int = 100,
    warmup: int = 10,
    device: torch.device = torch.device("cuda"),
) -> BenchmarkResult:
    """Benchmark forward pass throughput.

    Args:
        layer: Layer to benchmark
        batch_size: Batch size
        in_features: Input features
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations
        device: Device to run on

    Returns:
        BenchmarkResult with timing statistics
    """
    layer = layer.to(device)
    x = torch.randn(batch_size, in_features, device=device)

    # Warmup
    for _ in range(warmup):
        _ = layer(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        y = layer(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (batch_size * 1000) / mean_time  # samples/sec

    # Memory
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0

    return BenchmarkResult(
        name="forward",
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
        memory_mb=memory_mb,
    )


def benchmark_backward_pass(
    layer: nn.Module,
    batch_size: int,
    in_features: int,
    num_iterations: int = 100,
    warmup: int = 10,
    device: torch.device = torch.device("cuda"),
) -> BenchmarkResult:
    """Benchmark backward pass throughput."""
    layer = layer.to(device)
    x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

    # Warmup
    for _ in range(warmup):
        y = layer(x)
        loss = y.sum()
        loss.backward()
        layer.zero_grad()
        x.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        y = layer(x)
        loss = y.sum()

        start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

        layer.zero_grad()
        x.grad = None

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (batch_size * 1000) / mean_time

    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0

    return BenchmarkResult(
        name="backward",
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
        memory_mb=memory_mb,
    )


def benchmark_resu_update(
    layer: RESULinear,
    batch_size: int,
    in_features: int,
    num_iterations: int = 100,
    warmup: int = 10,
    device: torch.device = torch.device("cuda"),
) -> BenchmarkResult:
    """Benchmark RESU update step."""
    layer = layer.to(device)
    x = torch.randn(batch_size, in_features, device=device)

    # Warmup
    for _ in range(warmup):
        y = layer(x)
        loss = y.sum()
        loss.backward()
        grad = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)
        layer.resu_step(grad)
        layer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        y = layer(x)
        loss = y.sum()
        loss.backward()

        grad = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)

        start = time.perf_counter()
        layer.resu_step(grad)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)
        layer.zero_grad()

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (batch_size * 1000) / mean_time

    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0

    return BenchmarkResult(
        name="resu_update",
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
        memory_mb=memory_mb,
    )


def run_comprehensive_benchmark(
    in_features: int = 2048,
    out_features: int = 2048,
    batch_size: int = 32,
    sparsity: float = 0.5,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, BenchmarkResult]:
    """Run comprehensive throughput benchmark.

    Compares:
    - Dense nn.Linear
    - RESU sparse mode
    - RESU resurrection mode
    """
    print(f"\n{'='*70}")
    print(f"RESU Throughput Benchmark")
    print(f"{'='*70}")
    print(f"Shape: ({in_features}, {out_features})")
    print(f"Batch size: {batch_size}")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    results = {}

    # Dense baseline
    print("Benchmarking: Dense nn.Linear")
    dense = nn.Linear(in_features, out_features, device=device)

    results["dense_forward"] = benchmark_forward_pass(dense, batch_size, in_features, device=device)
    results["dense_backward"] = benchmark_backward_pass(dense, batch_size, in_features, device=device)

    print(f"  Forward:  {results['dense_forward'].mean_time_ms:.3f} ± {results['dense_forward'].std_time_ms:.3f} ms")
    print(f"  Backward: {results['dense_backward'].mean_time_ms:.3f} ± {results['dense_backward'].std_time_ms:.3f} ms")

    # RESU sparse mode
    print("\nBenchmarking: RESU Sparse Mode")
    resu_sparse = RESULinear(in_features, out_features, device=device)
    resu_sparse.prune_by_magnitude(sparsity)

    results["sparse_forward"] = benchmark_forward_pass(resu_sparse, batch_size, in_features, device=device)
    results["sparse_backward"] = benchmark_backward_pass(resu_sparse, batch_size, in_features, device=device)

    print(f"  Forward:  {results['sparse_forward'].mean_time_ms:.3f} ± {results['sparse_forward'].std_time_ms:.3f} ms")
    print(f"  Backward: {results['sparse_backward'].mean_time_ms:.3f} ± {results['sparse_backward'].std_time_ms:.3f} ms")

    # RESU resurrection mode
    print("\nBenchmarking: RESU Resurrection Mode")
    resu_res = RESULinear(in_features, out_features, device=device)
    resu_res.prune_by_magnitude(sparsity)
    resu_res.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)

    results["resu_forward"] = benchmark_forward_pass(resu_res, batch_size, in_features, device=device)
    results["resu_backward"] = benchmark_backward_pass(resu_res, batch_size, in_features, device=device)
    results["resu_update"] = benchmark_resu_update(resu_res, batch_size, in_features, device=device)

    print(f"  Forward:  {results['resu_forward'].mean_time_ms:.3f} ± {results['resu_forward'].std_time_ms:.3f} ms")
    print(f"  Backward: {results['resu_backward'].mean_time_ms:.3f} ± {results['resu_backward'].std_time_ms:.3f} ms")
    print(f"  Update:   {results['resu_update'].mean_time_ms:.3f} ± {results['resu_update'].std_time_ms:.3f} ms")

    # Summary
    print(f"\n{'='*70}")
    print("Summary (Speedup vs Dense)")
    print(f"{'='*70}")

    sparse_speedup_fwd = results["dense_forward"].mean_time_ms / results["sparse_forward"].mean_time_ms
    sparse_speedup_bwd = results["dense_backward"].mean_time_ms / results["sparse_backward"].mean_time_ms

    resu_speedup_fwd = results["dense_forward"].mean_time_ms / results["resu_forward"].mean_time_ms
    resu_speedup_bwd = results["dense_backward"].mean_time_ms / results["resu_backward"].mean_time_ms

    print(f"Sparse mode forward:  {sparse_speedup_fwd:.2f}x")
    print(f"Sparse mode backward: {sparse_speedup_bwd:.2f}x")
    print(f"RESU mode forward:    {resu_speedup_fwd:.2f}x")
    print(f"RESU mode backward:   {resu_speedup_bwd:.2f}x")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Running on CPU. Benchmarks will be slow.")

    # Run benchmarks for various sizes
    configs = [
        {"in_features": 512, "out_features": 512, "batch_size": 64},
        {"in_features": 2048, "out_features": 2048, "batch_size": 32},
        {"in_features": 4096, "out_features": 4096, "batch_size": 16},
    ]

    all_results = {}
    for config in configs:
        name = f"{config['in_features']}x{config['out_features']}_bs{config['batch_size']}"
        all_results[name] = run_comprehensive_benchmark(**config, device=device)

    print("\nAll benchmarks complete!")
