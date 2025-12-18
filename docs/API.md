# RESU API Reference

Complete API documentation for the RESU library.

---

## Table of Contents

- [Core Modules](#core-modules)
  - [RESULinear](#resulinear)
  - [SparseMask](#sparsemask)
  - [SelectionConfig](#selectionconfig)
- [Utilities](#utilities)
  - [Quantization](#quantization)
  - [Memory Tracking](#memory-tracking)
- [Training Cycle](#training-cycle)

---

## Core Modules

### RESULinear

Drop-in replacement for `nn.Linear` with RESU/QRESU support.

**Location:** `resu.modules.linear.RESULinear`

#### Constructor

```python
RESULinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

**Parameters:**
- `in_features` - Size of input features
- `out_features` - Size of output features
- `bias` - If True, adds learnable bias (default: True)
- `device` - Device to place parameters on
- `dtype` - Data type for parameters

**Example:**
```python
from resu.modules.linear import RESULinear

# Drop-in replacement for nn.Linear
layer = RESULinear(512, 256, bias=True)
```

---

#### Operating Modes

RESULinear operates in different modes during training:

**RESUMode Enum:**
- `DENSE` - Standard forward pass (no pruning)
- `SPARSE` - Forward with mask applied (zeroed pruned weights)
- `RESU` - Resurrection phase (θ parameters active)
- `QRESU` - Quantized RESU (4/8-bit W_A, FP32 θ)
- `QRESU_SELECTIVE` - QRESU with gradient filtering

**Check current mode:**
```python
layer._mode  # Returns RESUMode enum
```

---

### Pruning Methods

#### `prune_by_magnitude()`

Prune weights by magnitude (smallest weights removed).

```python
layer.prune_by_magnitude(
    sparsity: float,
    global_pruning: bool = False,
) -> None
```

**Parameters:**
- `sparsity` - Fraction of weights to prune (0.0 to 1.0)
- `global_pruning` - If True, prune globally across entire weight matrix. If False, prune per-row (default: False)

**Example:**
```python
# Prune 50% of weights by magnitude
layer.prune_by_magnitude(0.5)

# Check sparsity
print(f"Sparsity: {layer._mask.sparsity:.1%}")  # "Sparsity: 50.0%"
```

**What it does:**
1. Computes magnitude `|W|` for all weights
2. Selects bottom `sparsity * total_params` weights
3. Creates mask marking these positions as pruned
4. Zeros out pruned positions in weight matrix

---

#### `prune_by_wanda()`

Prune using Wanda (magnitude × activation sensitivity).

```python
layer.prune_by_wanda(
    sparsity: float,
    activation_norms: torch.Tensor,
    global_pruning: bool = False,
) -> None
```

**Parameters:**
- `sparsity` - Fraction to prune (0.0 to 1.0)
- `activation_norms` - Activation magnitudes (shape: `[in_features]`)
- `global_pruning` - Global vs per-row pruning

**Example:**
```python
# Capture activations during forward pass
layer._capture_activations = True
outputs = layer(inputs)
activation_norms = layer._activation_norms

# Prune using Wanda
layer.prune_by_wanda(0.5, activation_norms)
```

**Formula:** Prunes by `score = |W| * |X|` (weights with low magnitude AND low activation importance)

---

### RESU Mode Methods

#### `enter_resu_mode()`

Enter RESU mode: θ parameters stored at pruned positions.

```python
layer.enter_resu_mode(
    epsilon: float = 0.1,
) -> None
```

**Parameters:**
- `epsilon` - Initialization scale for θ parameters (default: 0.1)

**What it does:**
1. Initializes θ ∼ N(0, ε²σ²) at pruned positions
2. Stores θ in `W[pruned_positions]` (in-place)
3. Adds gradient hooks to mask active weight gradients
4. Sets mode to `RESUMode.RESU`

**Memory:**
- Weight matrix: `(out_features, in_features)` FP32
- Mask indices: `min(n_pruned, n_active)` int32

**Example:**
```python
layer.prune_by_magnitude(0.5)
layer.enter_resu_mode(epsilon=0.1)

# Now layer.weight contains:
# - W_active at active positions
# - θ at pruned positions
```

**Forward pass:** Returns `F.linear(x, W)` where W contains both W_active and θ

---

#### `enter_qresu_mode()`

Enter QRESU mode: Quantize W_A, store θ as flat 1D tensor.

```python
layer.enter_qresu_mode(
    bits: int = 4,
    epsilon: float = 0.1,
    qscheme: Literal["per_channel", "per_tensor"] = "per_channel",
) -> None
```

**Parameters:**
- `bits` - Quantization bit-width (4 or 8)
- `epsilon` - θ initialization scale
- `qscheme` - Quantization scheme:
  - `"per_channel"`: Separate scale/zero per output channel (recommended)
  - `"per_tensor"`: Single scale/zero for entire tensor

**What it does:**
1. Quantizes active weights W_A to specified bit-width
2. Initializes θ as **flat 1D tensor** `(n_pruned,)` FP32
3. Stores quantization parameters (scale, zero_point)
4. **Frees the original weight matrix** (no longer needed!)

**Memory (50% sparsity, 512→256 layer):**
- θ (flat FP32): 0.25 MB
- W_A (4-bit quantized): 0.125 MB
- Mask (int32 indices): 0.25 MB
- QParams: ~0.002 MB
- **Total: 0.627 MB** (vs 0.75 MB RESU - **16% savings!**)

**Example:**
```python
layer.prune_by_magnitude(0.5)
layer.enter_qresu_mode(bits=4, epsilon=0.1, qscheme="per_channel")

# Storage breakdown:
# - layer._theta: (n_pruned,) FP32 - trainable
# - layer._W_A_quantized: (out, in) uint8 - frozen
# - layer._qscale: (out_features,) FP32
# - layer._qzero: (out_features,) FP32
# - layer.weight: NOT USED (gradients disabled)
```

**Forward pass:**
1. Dequantize W_A
2. Reconstruct W_eff on-the-fly: `W_eff[active] = W_A_dequant[active]`, `W_eff[pruned] = θ`
3. Return `F.linear(x, W_eff)`

---

#### `enter_qresu_selective_mode()`

QRESU with intelligent gradient filtering (updates only high-quality θ coordinates).

```python
layer.enter_qresu_selective_mode(
    bits: int = 4,
    epsilon: float = 0.1,
    qscheme: Literal["per_channel", "per_tensor"] = "per_channel",
    selective_config: Optional[SelectionConfig] = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> None
```

**Parameters:**
- `bits`, `epsilon`, `qscheme` - Same as `enter_qresu_mode()`
- `selective_config` - Configuration for selective updates (see [SelectionConfig](#selectionconfig))
- `lr` - Learning rate for θ parameters
- `weight_decay` - L2 regularization for θ

**What it does:**
1. Enters QRESU mode (quantize W_A, flat θ storage)
2. Initializes EMA tracking (momentum, magnitude, consistency)
3. Registers gradient hook for selective filtering

**Selective Filtering:**
- Tracks gradient consistency: `C = |m| / (v + δ)`
- Selects ~20% of coordinates per step (highest consistency + magnitude)
- Only updates selected coordinates

**Memory overhead:** +3× θ storage for EMA buffers (m, v, consistency)

**Example:**
```python
from resu.core.selective import SelectionConfig

config = SelectionConfig(
    beta=0.9,           # EMA coefficient
    tau_stable=0.5,     # Consistency threshold
    k_select_ratio=0.2, # Select top 20%
)

layer.prune_by_magnitude(0.5)
layer.enter_qresu_selective_mode(
    bits=4,
    selective_config=config,
    lr=1e-4,
)

# Training automatically applies selective updates via gradient hooks
optimizer.zero_grad()
loss.backward()
optimizer.step()  # Only ~20% of θ coordinates updated!
```

---

#### `exit_resu_mode()` / `exit_qresu_mode()`

Exit RESU/QRESU mode and return to standard dense layer.

```python
layer.exit_resu_mode(commit: bool = True) -> None
layer.exit_qresu_mode(commit: bool = True) -> None
```

**Parameters:**
- `commit` - If True, merge θ back into weights. If False, discard θ.

**What it does:**
- **RESU:** Clears gradient hooks, keeps W with merged θ
- **QRESU:** Dequantizes W_A, merges with θ, reconstructs full weight matrix

**Example:**
```python
# After RESU training
layer.exit_resu_mode(commit=True)
# Now layer.weight contains: W_active (trained) + θ (merged)
# Mask and hooks are freed

# If you don't want to keep θ:
layer.exit_resu_mode(commit=False)
# Pruned positions stay zero
```

---

### SparseMask

Optimized sparse mask with int32 indices and adaptive storage.

**Location:** `resu.core.mask.SparseMask`

#### Constructor

```python
SparseMask(
    pruned_indices: torch.Tensor,
    shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    adaptive: bool = True,
)
```

**Parameters:**
- `pruned_indices` - Flat indices of pruned positions (1D int64 tensor)
- `shape` - Shape of weight matrix (e.g., `(out_features, in_features)`)
- `device` - Device to place mask on
- `adaptive` - If True, store whichever is smaller: active or pruned indices

**Adaptive Storage:**
- At 10% sparsity: stores 10% (pruned indices)
- At 90% sparsity: stores 10% (active indices - complement)
- Minimizes memory: `O(min(n_pruned, n_active))`

---

#### Factory Methods

```python
# Create from dense boolean mask
mask = SparseMask.from_dense_mask(dense_mask)

# Create by magnitude pruning
mask = SparseMask.from_magnitude(weights, sparsity=0.5)

# Create random mask
mask = SparseMask.random(shape=(512, 256), sparsity=0.5)
```

---

#### Properties

```python
mask.shape          # (out_features, in_features)
mask.sparsity       # 0.5 (50% pruned)
mask.n_pruned       # Number of pruned parameters
mask.n_active       # Number of active parameters
mask.device         # torch.device

# Get indices (returns int64 for indexing compatibility)
mask.pruned_indices  # Flat indices of pruned positions
mask.active_indices  # Flat indices of active positions

# Dense mask (EXPENSIVE - computed on demand!)
mask.mask           # Boolean tensor (1=active, 0=pruned)

# Statistics
mask.stats          # MaskStats(total=..., n_active=..., n_pruned=..., sparsity=...)
```

---

#### Methods

```python
# Apply mask (zero out pruned positions)
masked_tensor = mask.apply(tensor)
mask.apply_inplace(tensor)  # In-place version

# Extract elements
active_values = mask.get_active(tensor)   # 1D tensor
pruned_values = mask.get_pruned(tensor)   # 1D tensor

# Update mask with new pruning pattern
mask.update(new_dense_mask, inplace=True)

# Move to device
mask_cuda = mask.to(torch.device('cuda'))

# Serialization
state = mask.state_dict()
mask = SparseMask.from_state_dict(state)
```

---

### SelectionConfig

Configuration for RESU-Selective gradient filtering.

**Location:** `resu.core.selective.SelectionConfig`

```python
from dataclasses import dataclass

@dataclass
class SelectionConfig:
    beta: float = 0.9           # EMA coefficient for momentum/magnitude
    delta: float = 1e-8         # Stability constant for division
    tau_stable: float = 0.5     # Consistency threshold
    k_screen_ratio: float = 0.5 # Fraction for magnitude screening
    k_select_ratio: float = 0.2 # Fraction for final selection
```

**Parameters:**
- `beta` - EMA decay (higher = slower adaptation)
- `delta` - Small constant to prevent division by zero
- `tau_stable` - Minimum consistency to be considered "stable"
- `k_screen_ratio` - Screen top K by gradient magnitude
- `k_select_ratio` - Final selection ratio (after filtering)

**Selection Algorithm:**
1. Compute `P_mag` = Top-K by gradient magnitude (k_screen_ratio)
2. Compute `P_con` = Coords with consistency > tau_stable
3. Intersection: `P_int = P_mag ∩ P_con`
4. Select Top-K from intersection by magnitude (k_select_ratio)

**Example:**
```python
config = SelectionConfig(
    beta=0.9,           # Smooth EMA
    tau_stable=0.5,     # Only update if C > 0.5
    k_select_ratio=0.2, # Update top 20%
)
```

---

## Utilities

### Quantization

**Location:** `resu.utils.quantization`

#### Per-Channel Quantization (Recommended)

```python
from resu.utils.quantization import quantize_per_channel, dequantize_per_channel

# Quantize
W_q, scale, zero_point = quantize_per_channel(W, bits=4)
# W_q: (out, in) uint8
# scale: (out,) FP32
# zero_point: (out,) FP32

# Dequantize
W = dequantize_per_channel(W_q, scale, zero_point)
```

**Why per-channel?**
- Better quality (separate scale per output channel)
- Minimal overhead (~2KB for 256 output channels)

---

#### Per-Tensor Quantization

```python
from resu.utils.quantization import quantize_per_tensor, dequantize_per_tensor

W_q, scale, zero_point = quantize_per_tensor(W, bits=4)
# scale: scalar
# zero_point: scalar

W = dequantize_per_tensor(W_q, scale, zero_point)
```

**When to use:**
- Simplest (single scale for entire tensor)
- Lowest overhead
- Acceptable quality loss

---

#### Generic Quantize/Dequantize

```python
from resu.utils.quantization import quantize, dequantize

# Quantize
W_q, qparams = quantize(W, bits=4, scheme="per_channel")

# Dequantize
W = dequantize(W_q, qparams, scheme="per_channel")
```

---

### Memory Tracking

Utility functions for measuring memory usage.

```python
def get_tensor_memory(tensor: torch.Tensor) -> float:
    """Get memory used by tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024

# Example usage
mem_mb = get_tensor_memory(layer.weight)
print(f"Weight memory: {mem_mb:.2f} MB")
```

**Measuring RESULinear memory:**

```python
def measure_layer_memory(layer: RESULinear) -> dict:
    breakdown = {}

    # Weight (FP32) - only in Dense/RESU modes
    if layer._mode not in [RESUMode.QRESU, RESUMode.QRESU_SELECTIVE]:
        breakdown['weight'] = get_tensor_memory(layer.weight)

    # Mask indices
    if layer._mask is not None:
        breakdown['mask'] = get_tensor_memory(layer._mask._indices)

    # QRESU: flat θ
    if layer._theta is not None:
        breakdown['theta'] = get_tensor_memory(layer._theta)

    # QRESU: quantized W_A
    if layer._W_A_quantized is not None:
        breakdown['W_A_quantized'] = get_tensor_memory(layer._W_A_quantized)
        breakdown['qparams'] = (
            get_tensor_memory(layer._qscale) +
            get_tensor_memory(layer._qzero)
        )

    breakdown['total'] = sum(breakdown.values())
    return breakdown
```

---

## Training Cycle

### Standard RESU Cycle

```python
from resu.modules.linear import RESULinear

# 1. Create layer and train dense
layer = RESULinear(512, 256)
# ... train for N epochs ...

# 2. Prune
layer.prune_by_magnitude(0.5)  # 50% sparsity

# 3. Enter RESU mode
layer.enter_resu_mode(epsilon=0.1)

# 4. Train with resurrection
optimizer = torch.optim.Adam([layer.weight], lr=1e-4)
for epoch in range(resu_epochs):
    # Forward/backward as usual
    # θ parameters are automatically trained via gradient hooks
    loss = train_step(layer, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Exit RESU mode
layer.exit_resu_mode(commit=True)
# Now layer.weight contains merged W_active + θ
```

---

### QRESU Cycle (Memory-Optimized)

```python
# 1-2. Dense training + pruning (same as above)
layer.prune_by_magnitude(0.5)

# 3. Enter QRESU mode
layer.enter_qresu_mode(bits=4, epsilon=0.1, qscheme="per_channel")

# 4. Train (only θ is trainable - W_A is quantized/frozen)
optimizer = torch.optim.Adam([layer._theta], lr=1e-4)
for epoch in range(qresu_epochs):
    loss = train_step(layer, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Exit QRESU mode
layer.exit_qresu_mode(commit=True)
# Dequantizes W_A and merges with θ into dense weight
```

**Memory Savings:** 16-59% vs RESU (depending on sparsity)

---

### QRESU-Selective Cycle (Update-Filtered)

```python
# 1-2. Dense training + pruning
layer.prune_by_magnitude(0.5)

# 3. Enter QRESU-Selective
from resu.core.selective import SelectionConfig

config = SelectionConfig(k_select_ratio=0.2)
layer.enter_qresu_selective_mode(
    bits=4,
    selective_config=config,
    lr=1e-4,  # Learning rate for θ
)

# 4. Train (selective updates applied automatically via hooks)
# No optimizer needed for θ - gradient hook handles updates!
optimizer = torch.optim.Adam(other_params, lr=1e-4)
for epoch in range(selective_epochs):
    loss = train_step(layer, data)
    optimizer.zero_grad()
    loss.backward()  # Hook applies selective update to θ
    optimizer.step()

# 5. Exit
layer.exit_qresu_mode(commit=True)
```

**Update Efficiency:** Only ~20% of θ coordinates updated per step

---

## Performance Characteristics

### Memory Comparison (512→256 layer, 50% sparsity)

| Mode | Memory | vs Dense | vs RESU |
|------|--------|----------|---------|
| Dense | 0.50 MB | 1.0× | - |
| RESU | 0.75 MB | 1.5× | 1.0× |
| QRESU (4-bit) | 0.63 MB | 1.26× | **0.84×** (16% savings) |
| QRESU-Selective | 1.38 MB | 2.76× | 1.84× (+EMA buffers) |

### Speed

- **RESU:** ~1.1× slower than dense (optimized!)
- **QRESU:** ~1.2× slower (dequantization overhead)
- **Dense:** 1.0× (baseline)

### Best Sparsity for QRESU

| Sparsity | QRESU Savings vs RESU | Recommendation |
|----------|----------------------|----------------|
| 10% | **59%** | ⭐⭐⭐ Excellent |
| 30% | **34%** | ⭐⭐ Very Good |
| 50% | **16%** | ⭐ Good |
| 70% | **4%** | Marginal |
| 90% | -14% (overhead) | ❌ Don't use |

**Optimal:** 10-50% sparsity

---

## High-Level Training API

For complete RESU training cycles with densification, use the built-in training infrastructure.

### RESUConfig

Complete configuration for RESU training:

```python
from resu.training.config import RESUConfig, SparsitySchedule

config = RESUConfig(
    # Sparsity
    initial_sparsity=0.0,           # Start dense
    target_sparsity=0.7,            # Target 70% sparsity
    sparsity_schedule=SparsitySchedule.LINEAR,

    # Densification (optional)
    densify=False,                   # Set True to decrease sparsity
    densification_steps=[0.7, 0.5, 0.3, 0.1, 0.0],

    # Cycle structure
    num_cycles=5,
    steps_per_cycle=1000,
    train_fraction=0.6,             # 60% training
    stabilize_fraction=0.1,         # 10% DSNoT stabilization
    resu_fraction=0.3,              # 30% RESU resurrection

    # RESU parameters
    resu_lr=1e-4,
    resu_epsilon=0.1,
    freeze_active_during_resu=True,

    # RESU-Selective
    use_selective=True,
    selective_beta=0.9,
    selective_tau=0.5,
    selective_k_select_ratio=0.2,

    # Amnesty
    use_amnesty=True,
    commit_strategy="amnesty",       # "amnesty" | "wanda_reprune" | "simple"
    amnesty_r_start=0.10,
    amnesty_r_end=0.02,
    amnesty_score_type="magnitude",

    # Pruning
    pruning_method="wanda",          # "wanda" | "magnitude" | "random"

    # Optimizer
    base_lr=1e-3,
    weight_decay=0.01,
)
```

**Preset configurations:**

```python
from resu.training.config import (
    default_config,
    aggressive_pruning_config,    # 90% sparsity, 7 cycles
    conservative_pruning_config,  # 50% sparsity, 3 cycles
    densification_config,         # Recover from 70% → 0% sparsity
    quick_test_config,            # Minimal config for testing
)

config = aggressive_pruning_config()
```

---

### RESUCycle

Executes one complete RESU training cycle:

**Phases:**
1. **TRAIN**: Standard training with current sparse mask
2. **PRUNE**: Prune to target sparsity (Wanda or magnitude)
3. **STABILIZE**: DSNoT refinement (optional)
4. **RESU**: Train resurrection parameters θ
5. **COMMIT**: Merge θ with amnesty tournament

**Example:**

```python
from resu.training.cycle import RESUCycle
from resu.training.config import RESUConfig

config = RESUConfig(target_sparsity=0.7, num_cycles=5)

# Define training function
def train_step(model, batch):
    x, y = batch
    logits = model(x)
    return nn.CrossEntropyLoss()(logits, y)

# Create cycle
cycle = RESUCycle(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    cycle_num=0,  # First cycle
)

# Run complete cycle
stats = cycle.run(train_loader)

print(f"Cycle complete: {stats.actual_sparsity:.1%} sparsity")
print(f"Resurrected: {stats.n_resurrected} weights")
```

**CycleStats output:**

```python
@dataclass
class CycleStats:
    cycle: int
    target_sparsity: float
    actual_sparsity: float
    train_loss: float
    train_steps: int
    resu_steps: int
    resu_updates: int
    mean_consistency: float         # RESU-Selective metric
    mean_selection_ratio: float     # % of θ updated per step
    resurrection_budget: float
    n_resurrected: int
    n_active_kept: int
    resurrection_rate: float
    duration_seconds: float
```

---

### RESUTrainer

Manages full multi-cycle RESU training:

**Example:**

```python
from resu.training.cycle import RESUTrainer
from resu.training.config import densification_config

config = densification_config()  # Recover from 70% → 0% sparsity

trainer = RESUTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    eval_fn=evaluate_fn,  # Optional
)

# Run 5 cycles
all_stats = trainer.train(train_loader, num_cycles=5)

# Get summary
summary = trainer.get_training_summary()
print(f"Final sparsity: {summary['final_sparsity']:.1%}")
print(f"Total resurrected: {summary['total_resurrected']}")
```

**With evaluation:**

```python
def evaluate_fn(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total}

trainer = RESUTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    eval_fn=evaluate_fn,
)

trainer.train(train_loader)
```

---

### Commit Strategies

The `commit_strategy` parameter controls how resurrections are merged:

#### 1. Amnesty (Default)

Tournament between active and resurrected weights:

```python
config = RESUConfig(
    commit_strategy="amnesty",
    amnesty_r_start=0.10,  # 10% resurrection budget at cycle 0
    amnesty_r_end=0.02,    # 2% at final cycle (decays linearly)
    amnesty_score_type="magnitude",
)
```

**How it works:**
- Score all weights (magnitude, gradient, or Wanda)
- Tournament: top r% of resurrected weights compete with bottom r% of active weights
- Winners become the new active set

#### 2. Wanda Re-Pruning

Merge all resurrections, then re-prune with structure-aware Wanda:

```python
config = RESUConfig(
    commit_strategy="wanda_reprune",
    pruning_method="wanda",
)
```

**How it works:**
- Merge ALL θ → W (no tournament)
- Next cycle: re-prune with Wanda++ (considers structure)
- Better for structured pruning

#### 3. Simple Merge

Just merge θ → W without re-pruning:

```python
config = RESUConfig(
    commit_strategy="simple",
)
```

**Use when:** You want to densify without pruning (recover all weights)

---

### Densification Algorithm

Progressive recovery from high sparsity to dense:

**Example:**

```python
from resu.training.config import RESUConfig, DensificationSchedule

config = RESUConfig(
    densify=True,
    initial_sparsity=0.7,          # Start at 70% sparse
    densification_schedule=DensificationSchedule.STEPPED,
    densification_steps=[0.7, 0.5, 0.3, 0.1, 0.0],
    num_cycles=5,

    # More generous resurrection budget for densification
    amnesty_r_start=0.30,
    amnesty_r_end=0.10,
)

trainer = RESUTrainer(model, config, optimizer, train_step)
stats = trainer.train(train_loader)

# Watch sparsity decrease across cycles
for s in stats:
    print(f"Cycle {s.cycle}: {s.actual_sparsity:.1%} sparsity, "
          f"{s.n_resurrected} resurrected")
```

**Output:**
```
Cycle 0: 70.0% sparsity, 15000 resurrected
Cycle 1: 50.0% sparsity, 12000 resurrected
Cycle 2: 30.0% sparsity, 8000 resurrected
Cycle 3: 10.0% sparsity, 4000 resurrected
Cycle 4: 0.0% sparsity, 2000 resurrected
```

**Use cases:**
- Recover from over-pruning
- Adapt to new task requiring more capacity
- Gradually restore model performance

---

### Integration with Custom Pruners

Use your own Wanda++ or DSNoT implementation:

```python
from resu.training.cycle import RESUTrainer

# Your custom pruner
class MyWandaPruner:
    def prune(self, sparsity):
        # Your Wanda++ logic
        pass

pruner = MyWandaPruner(model)
stabilizer = MyDSNoTStabilizer(model)

trainer = RESUTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    pruner=pruner,           # Optional: custom pruner
    stabilizer=stabilizer,   # Optional: custom stabilizer
)

trainer.train(train_loader)
```

If not provided, falls back to magnitude pruning on `RESULinear` layers.

---

### Advanced: Densification with RL Pauses

For reinforcement learning workflows, use `DensificationTrainer` which supports pause points:

```python
from resu.training.densification import DensificationTrainer, PauseConfig, PauseReason

# Define pause points for RL training
pauses = [
    PauseConfig(after_cycle=0, reason=PauseReason.RL_TRAINING, duration_steps=5000),
    PauseConfig(after_cycle=1, reason=PauseReason.RL_TRAINING, duration_steps=5000),
    PauseConfig(after_cycle=2, reason=PauseReason.RL_TRAINING, duration_steps=5000),
]

trainer = DensificationTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    pause_configs=pauses,
)

# Run with pauses
for cycle_stats in trainer.train_with_pauses(train_loader):
    if trainer.is_paused:
        # Do RL training here
        print(f"Paused after cycle {cycle_stats.cycle}")
        run_rl_training(model, num_steps=5000)

        # Resume
        trainer.resume()
```

**Use case:** Interleave supervised RESU densification with RL fine-tuning phases

---

## See Also

- [USAGE.md](USAGE.md) - Quick start guide
- [OPTIMIZATION_RESULTS.md](../OPTIMIZATION_RESULTS.md) - RESU optimization details
- [QRESU_OPTIMIZATION_COMPLETE.md](../QRESU_OPTIMIZATION_COMPLETE.md) - QRESU implementation
