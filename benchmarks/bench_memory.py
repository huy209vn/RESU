"""
Memory usage benchmarks for RESU.

Verifies the paper's claim: RESU adds no memory overhead beyond dense storage.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict
import gc
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear
from resu.core.resurrection import StorageMode


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    parameter_memory_mb: float
    optimizer_memory_mb: float


def get_memory_stats(device: torch.device) -> MemoryStats:
    """Get current memory statistics."""
    if device.type != "cuda":
        return MemoryStats(0, 0, 0, 0, 0)

    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    return MemoryStats(
        allocated_mb=allocated,
        reserved_mb=reserved,
        peak_mb=peak,
        parameter_memory_mb=0,  # Will be filled separately
        optimizer_memory_mb=0,
    )


def reset_memory_stats(device: torch.device):
    """Reset memory statistics."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def measure_parameter_memory(layer: nn.Module) -> float:
    """Measure memory used by parameters in MB."""
    total_bytes = 0
    for p in layer.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / 1024 / 1024


def measure_optimizer_memory(optimizer: torch.optim.Optimizer) -> float:
    """Measure memory used by optimizer state in MB."""
    total_bytes = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total_bytes += v.numel() * v.element_size()
    return total_bytes / 1024 / 1024


def benchmark_dense_memory(
    in_features: int,
    out_features: int,
    device: torch.device,
) -> Dict[str, float]:
    """Benchmark memory for dense nn.Linear + Adam."""
    reset_memory_stats(device)

    layer = nn.Linear(in_features, out_features, device=device)
    param_memory = measure_parameter_memory(layer)

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

    # Trigger optimizer state creation
    x = torch.randn(1, in_features, device=device)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    optim_memory = measure_optimizer_memory(optimizer)
    total_memory = param_memory + optim_memory

    stats = get_memory_stats(device)

    return {
        "parameter_mb": param_memory,
        "optimizer_mb": optim_memory,
        "total_mb": total_memory,
        "peak_mb": stats.peak_mb,
    }


def benchmark_sparse_memory(
    in_features: int,
    out_features: int,
    sparsity: float,
    device: torch.device,
) -> Dict[str, float]:
    """Benchmark memory for sparse RESULinear + Adam."""
    reset_memory_stats(device)

    layer = RESULinear(in_features, out_features, device=device)
    layer.prune_by_magnitude(sparsity)

    param_memory = measure_parameter_memory(layer)

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

    # Trigger optimizer state
    x = torch.randn(1, in_features, device=device)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    optim_memory = measure_optimizer_memory(optimizer)
    total_memory = param_memory + optim_memory

    stats = get_memory_stats(device)

    return {
        "parameter_mb": param_memory,
        "optimizer_mb": optim_memory,
        "total_mb": total_memory,
        "peak_mb": stats.peak_mb,
    }


def benchmark_resu_memory(
    in_features: int,
    out_features: int,
    sparsity: float,
    storage_mode: StorageMode,
    device: torch.device,
) -> Dict[str, float]:
    """Benchmark memory for RESU mode + RESU updater."""
    reset_memory_stats(device)

    layer = RESULinear(in_features, out_features, device=device, storage_mode=storage_mode)
    layer.prune_by_magnitude(sparsity)
    layer.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)

    param_memory = measure_parameter_memory(layer)

    # RESU uses its own updater, but let's also test with main optimizer
    # (in practice, main optimizer doesn't touch params during RESU)

    # Measure RESU-specific memory
    resu_memory = 0
    if layer.resurrection is not None:
        # Theta
        resu_memory += layer.resurrection.theta.numel() * layer.resurrection.theta.element_size()

        # Selective state (m, v, consistency)
        if layer._selective is not None:
            resu_memory += layer._selective.m.numel() * layer._selective.m.element_size()
            resu_memory += layer._selective.v.numel() * layer._selective.v.element_size()
            resu_memory += layer._selective.consistency.numel() * layer._selective.consistency.element_size()

    resu_memory_mb = resu_memory / 1024 / 1024

    # Run one step to allocate everything
    x = torch.randn(1, in_features, device=device)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    grad = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)
    layer.resu_step(grad)

    stats = get_memory_stats(device)

    return {
        "parameter_mb": param_memory,
        "resu_state_mb": resu_memory_mb,
        "total_mb": param_memory + resu_memory_mb,
        "peak_mb": stats.peak_mb,
    }


def run_memory_benchmark(
    in_features: int = 4096,
    out_features: int = 4096,
    sparsity: float = 0.5,
    device: torch.device = torch.device("cuda"),
):
    """Run comprehensive memory benchmark."""
    print(f"\n{'='*70}")
    print(f"RESU Memory Benchmark")
    print(f"{'='*70}")
    print(f"Shape: ({in_features}, {out_features})")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Dense baseline
    print("Benchmarking: Dense nn.Linear")
    dense_stats = benchmark_dense_memory(in_features, out_features, device)
    print(f"  Parameters: {dense_stats['parameter_mb']:.2f} MB")
    print(f"  Optimizer:  {dense_stats['optimizer_mb']:.2f} MB")
    print(f"  Total:      {dense_stats['total_mb']:.2f} MB")
    print(f"  Peak:       {dense_stats['peak_mb']:.2f} MB")

    # Sparse (no RESU)
    print("\nBenchmarking: Sparse (no RESU)")
    sparse_stats = benchmark_sparse_memory(in_features, out_features, sparsity, device)
    print(f"  Parameters: {sparse_stats['parameter_mb']:.2f} MB")
    print(f"  Optimizer:  {sparse_stats['optimizer_mb']:.2f} MB")
    print(f"  Total:      {sparse_stats['total_mb']:.2f} MB")
    print(f"  Peak:       {sparse_stats['peak_mb']:.2f} MB")

    # RESU Compact mode
    print("\nBenchmarking: RESU Compact Mode")
    resu_compact_stats = benchmark_resu_memory(in_features, out_features, sparsity, StorageMode.COMPACT, device)
    print(f"  Parameters:  {resu_compact_stats['parameter_mb']:.2f} MB")
    print(f"  RESU state:  {resu_compact_stats['resu_state_mb']:.2f} MB")
    print(f"  Total:       {resu_compact_stats['total_mb']:.2f} MB")
    print(f"  Peak:        {resu_compact_stats['peak_mb']:.2f} MB")

    # RESU Dense mode
    print("\nBenchmarking: RESU Dense Mode")
    resu_dense_stats = benchmark_resu_memory(in_features, out_features, sparsity, StorageMode.DENSE, device)
    print(f"  Parameters:  {resu_dense_stats['parameter_mb']:.2f} MB")
    print(f"  RESU state:  {resu_dense_stats['resu_state_mb']:.2f} MB")
    print(f"  Total:       {resu_dense_stats['total_mb']:.2f} MB")
    print(f"  Peak:        {resu_dense_stats['peak_mb']:.2f} MB")

    # Analysis
    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}")

    # Dense vs RESU Dense mode
    overhead_dense = resu_dense_stats['total_mb'] - dense_stats['parameter_mb']
    overhead_pct_dense = (overhead_dense / dense_stats['parameter_mb']) * 100

    print(f"\nMemory overhead (RESU Dense mode vs Dense parameters only):")
    print(f"  Absolute: {overhead_dense:.2f} MB")
    print(f"  Relative: {overhead_pct_dense:.1f}%")

    # RESU state breakdown
    print(f"\nRESU state consists of:")
    print(f"  - θ (resurrection parameters): p-dimensional vector")
    print(f"  - m, v (EMA buffers): p-dimensional vectors")
    print(f"  - C (consistency): p-dimensional vector")
    print(f"  Total: 4 × p floats")

    n_pruned = int(sparsity * in_features * out_features)
    theoretical_resu_mb = (4 * n_pruned * 4) / 1024 / 1024  # 4 vectors, 4 bytes/float
    print(f"  Theoretical for p={n_pruned}: {theoretical_resu_mb:.2f} MB")
    print(f"  Actual measured: {resu_compact_stats['resu_state_mb']:.2f} MB")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Memory benchmarks are only meaningful on GPU.")
        exit(1)

    device = torch.device("cuda")

    # Run for different sizes
    configs = [
        {"in_features": 1024, "out_features": 1024, "sparsity": 0.5},
        {"in_features": 4096, "out_features": 4096, "sparsity": 0.5},
        {"in_features": 4096, "out_features": 4096, "sparsity": 0.7},
    ]

    for config in configs:
        run_memory_benchmark(**config, device=device)
