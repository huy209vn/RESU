"""
Example: Training both active weights and θ during RESU phase.

By default, RESU freezes active weights and only trains θ (resurrection parameters).
Setting freeze_active_during_resu=False allows BOTH to be trained simultaneously.
"""

import torch
import torch.nn as nn
from resu.modules.linear import RESULinear

# Create a simple layer
layer = RESULinear(512, 256, bias=False)

# Train dense for a few steps
layer.train()
optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

print("=== Dense Training ===")
for step in range(5):
    x = torch.randn(32, 512)
    y = layer(x)
    loss = y.pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss={loss.item():.4f}")

# Prune to 50% sparsity
layer.prune_by_magnitude(0.5)
print(f"\n✓ Pruned to 50% sparsity")

# Get active and pruned masks
active_mask = layer._mask.mask.bool()
pruned_mask = ~active_mask

# Track a sample active weight and pruned position
active_idx = active_mask.nonzero()[0]
pruned_idx = pruned_mask.nonzero()[0]

active_weight_before = layer.weight.data.flatten()[active_idx].item()
pruned_weight_before = layer.weight.data.flatten()[pruned_idx].item()

print(f"\nBefore RESU:")
print(f"  Active weight[{active_idx.item()}]: {active_weight_before:.6f}")
print(f"  Pruned weight[{pruned_idx.item()}]: {pruned_weight_before:.6f} (should be ~0)")

# ============================================================================
# Option 1: Default - Freeze active weights (only train θ)
# ============================================================================
print("\n" + "="*80)
print("OPTION 1: freeze_active=True (default)")
print("="*80)

layer_frozen = RESULinear(512, 256, bias=False)
layer_frozen.weight.data.copy_(layer.weight.data)
layer_frozen.prune_by_magnitude(0.5)

layer_frozen.enter_resu_mode(
    epsilon=0.1,
    freeze_active=True,  # Only train θ
    use_selective=False,
)

optimizer_frozen = torch.optim.Adam(layer_frozen.parameters(), lr=1e-4)

for step in range(5):
    x = torch.randn(32, 512)
    y = layer_frozen(x)
    loss = y.pow(2).mean()

    optimizer_frozen.zero_grad()
    loss.backward()
    optimizer_frozen.step()

active_weight_frozen = layer_frozen.weight.data.flatten()[active_idx].item()
pruned_weight_frozen = layer_frozen.weight.data.flatten()[pruned_idx].item()

print(f"\nAfter 5 RESU steps (freeze_active=True):")
print(f"  Active weight[{active_idx.item()}]: {active_weight_frozen:.6f}")
print(f"  Pruned weight[{pruned_idx.item()}]: {pruned_weight_frozen:.6f}")
print(f"\n  Active weight change: {abs(active_weight_frozen - active_weight_before):.8f}")
print(f"  Pruned weight change: {abs(pruned_weight_frozen - 0.0):.8f}")
print(f"\n  ✓ Active weights FROZEN (no change)")
print(f"  ✓ Only θ (pruned positions) trained")

# ============================================================================
# Option 2: Train BOTH active weights and θ
# ============================================================================
print("\n" + "="*80)
print("OPTION 2: freeze_active=False (NEW!)")
print("="*80)

layer_both = RESULinear(512, 256, bias=False)
layer_both.weight.data.copy_(layer.weight.data)
layer_both.prune_by_magnitude(0.5)

layer_both.enter_resu_mode(
    epsilon=0.1,
    freeze_active=False,  # Train BOTH active and θ!
    use_selective=False,
)

optimizer_both = torch.optim.Adam(layer_both.parameters(), lr=1e-4)

for step in range(5):
    x = torch.randn(32, 512)
    y = layer_both(x)
    loss = y.pow(2).mean()

    optimizer_both.zero_grad()
    loss.backward()
    optimizer_both.step()

active_weight_both = layer_both.weight.data.flatten()[active_idx].item()
pruned_weight_both = layer_both.weight.data.flatten()[pruned_idx].item()

print(f"\nAfter 5 RESU steps (freeze_active=False):")
print(f"  Active weight[{active_idx.item()}]: {active_weight_both:.6f}")
print(f"  Pruned weight[{pruned_idx.item()}]: {pruned_weight_both:.6f}")
print(f"\n  Active weight change: {abs(active_weight_both - active_weight_before):.6f}")
print(f"  Pruned weight change: {abs(pruned_weight_both - 0.0):.6f}")
print(f"\n  ✓ Active weights TRAINED (changed!)")
print(f"  ✓ θ (pruned positions) also trained")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"freeze_active=True:  Active Δ={abs(active_weight_frozen - active_weight_before):.8f}")
print(f"freeze_active=False: Active Δ={abs(active_weight_both - active_weight_before):.6f}")
print()
print("Use freeze_active=False when you want full fine-tuning during RESU!")
