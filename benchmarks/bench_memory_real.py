"""
Real-world memory benchmark: Tests the actual training scenario.

This simulates what happens in actual training:
1. Dense training with optimizer (has W states)
2. Enter RESU (should clear W states with our fix)
3. RESU training (only theta states should exist)
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear


def get_memory_mb(device):
    """Get current GPU memory in MB."""
    if device.type != "cuda":
        return 0
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def measure_optimizer_states(optimizer):
    """Measure memory used by optimizer states."""
    total_bytes = 0
    num_params_with_state = 0
    for param, state in optimizer.state.items():
        num_params_with_state += 1
        for v in state.values():
            if torch.is_tensor(v):
                total_bytes += v.numel() * v.element_size()
    return total_bytes / 1024 / 1024, num_params_with_state


def benchmark_real_scenario(
    in_features=4096,
    out_features=4096,
    sparsity=0.5,
    device=torch.device("cuda"),
):
    """Benchmark the ACTUAL scenario: dense training → RESU training."""

    print(f"\n{'='*70}")
    print(f"Real-World Memory Benchmark")
    print(f"{'='*70}")
    print(f"Shape: ({in_features}, {out_features})")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Reset
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create layer
    layer = RESULinear(in_features, out_features, device=device)
    param_size_mb = (layer.weight.numel() * layer.weight.element_size()) / 1024 / 1024

    print(f"1. CREATED LAYER")
    print(f"   Weight size: {param_size_mb:.2f} MB")
    mem_after_layer = get_memory_mb(device)
    print(f"   GPU memory: {mem_after_layer:.2f} MB\n")

    # Create optimizer (like in real training)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

    # Do one step to allocate optimizer states
    x = torch.randn(32, in_features, device=device)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    optim_mem, num_params = measure_optimizer_states(optimizer)
    mem_after_optim = get_memory_mb(device)

    print(f"2. DENSE TRAINING (with optimizer)")
    print(f"   Optimizer tracking: {num_params} parameters")
    print(f"   Optimizer states: {optim_mem:.2f} MB")
    print(f"   GPU memory: {mem_after_optim:.2f} MB")
    print(f"   Expected: ~2x param size for Adam (momentum + variance)")
    print(f"   Actual overhead: {optim_mem / param_size_mb:.2f}x\n")

    # Prune
    layer.prune_by_magnitude(sparsity)

    # OLD WAY: Just enter RESU (optimizer keeps W states)
    print(f"3. ENTERING RESU (OLD WAY - no optimizer clearing)")

    # Save current state
    saved_state = {k: {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                   for k, v in optimizer.state.items()}

    layer.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)

    optim_mem_old, num_params_old = measure_optimizer_states(optimizer)

    # Measure RESU state
    resu_state_mb = 0
    if layer._selective is not None:
        resu_state_mb += layer._selective.m.numel() * 4 / 1024 / 1024
        resu_state_mb += layer._selective.v.numel() * 4 / 1024 / 1024
        resu_state_mb += layer._selective.consistency.numel() * 4 / 1024 / 1024

    mem_resu_old = get_memory_mb(device)

    print(f"   Optimizer still has: {num_params_old} parameters")
    print(f"   Optimizer states: {optim_mem_old:.2f} MB (WASTED!)")
    print(f"   RESU states (θ,m,v,C): {resu_state_mb:.2f} MB")
    print(f"   Total overhead: {(optim_mem_old + resu_state_mb) / param_size_mb:.2f}x")
    print(f"   GPU memory: {mem_resu_old:.2f} MB\n")

    # Exit RESU
    layer.exit_resu_mode(commit=False)
    optimizer.state = saved_state  # Restore

    # NEW WAY: Clear optimizer states before RESU
    print(f"4. ENTERING RESU (NEW WAY - with optimizer clearing)")

    # This is what our fix does in RESUCycle
    if layer.weight in optimizer.state:
        del optimizer.state[layer.weight]
        print(f"   ✓ Cleared optimizer state for weight")

    layer.enter_resu_mode(epsilon=0.1, use_selective=True, lr=0.001)

    optim_mem_new, num_params_new = measure_optimizer_states(optimizer)
    mem_resu_new = get_memory_mb(device)

    print(f"   Optimizer now has: {num_params_new} parameters")
    print(f"   Optimizer states: {optim_mem_new:.2f} MB")
    print(f"   RESU states (θ,m,v,C): {resu_state_mb:.2f} MB")
    print(f"   Total overhead: {(optim_mem_new + resu_state_mb) / param_size_mb:.2f}x")
    print(f"   GPU memory: {mem_resu_new:.2f} MB\n")

    # Compare
    print(f"{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"Old way (no clearing):")
    print(f"  Optimizer: {optim_mem_old:.2f} MB")
    print(f"  RESU: {resu_state_mb:.2f} MB")
    print(f"  Total: {optim_mem_old + resu_state_mb:.2f} MB")
    print(f"  Overhead: {(optim_mem_old + resu_state_mb) / param_size_mb:.2f}x\n")

    print(f"New way (with clearing):")
    print(f"  Optimizer: {optim_mem_new:.2f} MB")
    print(f"  RESU: {resu_state_mb:.2f} MB")
    print(f"  Total: {optim_mem_new + resu_state_mb:.2f} MB")
    print(f"  Overhead: {(optim_mem_new + resu_state_mb) / param_size_mb:.2f}x\n")

    saved_mb = (optim_mem_old + resu_state_mb) - (optim_mem_new + resu_state_mb)
    saved_pct = (saved_mb / (optim_mem_old + resu_state_mb)) * 100

    print(f"💰 MEMORY SAVED: {saved_mb:.2f} MB ({saved_pct:.1f}%)\n")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        exit(1)

    device = torch.device("cuda")
    benchmark_real_scenario(device=device)
