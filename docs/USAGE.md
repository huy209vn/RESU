# RESU Quick Start Guide

Get started with RESU in 5 minutes!

---

## Installation

```bash
git clone https://github.com/yourusername/resu.git
cd resu
pip install -e .
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- (Optional) Triton for CUDA acceleration

---

## What is RESU?

**RESU (REsurrection of Sparse Updates)** is a memory-efficient fine-tuning technique that:

1. **Prunes** a neural network (remove 50% of weights)
2. **Resurrects** pruned weights during training (they learn to compensate)
3. **Saves memory** compared to dense fine-tuning

**QRESU** adds quantization for even more memory savings!

---

## Basic Usage

### Step 1: Replace nn.Linear with RESULinear

```python
import torch
import torch.nn as nn
from resu.modules.linear import RESULinear

# Before: Standard PyTorch
# layer = nn.Linear(512, 256)

# After: RESU-enabled
layer = RESULinear(512, 256)
```

**That's it!** RESULinear is a drop-in replacement for `nn.Linear`.

---

### Step 2: Train Dense (Optional but Recommended)

```python
# Standard training loop
model = nn.Sequential(
    RESULinear(784, 512),
    nn.ReLU(),
    RESULinear(512, 256),
    nn.ReLU(),
    RESULinear(256, 10),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):  # Dense pre-training
    for batch in dataloader:
        loss = train_step(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### Step 3: Prune the Network

```python
# Prune all RESU layers to 50% sparsity
for module in model.modules():
    if isinstance(module, RESULinear):
        module.prune_by_magnitude(0.5)

print("✓ Pruned 50% of weights")
```

**What happened:** 50% of smallest-magnitude weights are marked as "pruned" and zeroed out.

---

### Step 4: Enter RESU Mode

```python
# Enter RESU mode (initialize resurrection parameters θ)
for module in model.modules():
    if isinstance(module, RESULinear):
        module.enter_resu_mode(epsilon=0.1)

print("✓ Entered RESU mode - resurrection parameters initialized")
```

**What changed:**
- Pruned positions now contain small random values (θ parameters)
- These θ parameters will be trained to compensate for pruning
- Gradient hooks ensure only θ gets trained (not active weights)

---

### Step 5: Train with Resurrection

```python
# Create optimizer for RESU phase
# Only optimize the weight matrices (θ is part of them)
resu_params = [m.weight for m in model.modules() if isinstance(m, RESULinear)]
optimizer = torch.optim.Adam(resu_params, lr=1e-4)

for epoch in range(20):  # RESU fine-tuning
    for batch in dataloader:
        loss = train_step(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("✓ RESU training complete")
```

**What's happening:**
- Forward pass uses both active weights AND θ
- Backward pass: gradient hooks ensure only θ gets updated
- θ learns to compensate for the pruned weights!

---

### Step 6: Exit RESU Mode (Optional)

```python
# Merge θ back into weights and clean up
for module in model.modules():
    if isinstance(module, RESULinear):
        module.exit_resu_mode(commit=True)

print("✓ Exited RESU mode - θ merged into weights")
```

**Result:** You now have a 50%-pruned model with recovered performance!

---

## Memory-Optimized: QRESU

For even lower memory, use **QRESU** (quantized RESU):

### Quick Start with QRESU

```python
# Steps 1-3: Same as above (create model, train dense, prune)

# Step 4: Enter QRESU mode instead of RESU
for module in model.modules():
    if isinstance(module, RESULinear):
        module.enter_qresu_mode(
            bits=4,              # 4-bit quantization
            epsilon=0.1,
            qscheme="per_channel",
        )

# Step 5: Train (only θ parameters, optimizer setup different!)
theta_params = [m._theta for m in model.modules()
                if isinstance(m, RESULinear) and m._theta is not None]
optimizer = torch.optim.Adam(theta_params, lr=1e-4)

for epoch in range(20):
    for batch in dataloader:
        loss = train_step(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Step 6: Exit QRESU
for module in model.modules():
    if isinstance(module, RESULinear):
        module.exit_qresu_mode(commit=True)
```

**Memory Savings:** 16-59% less than RESU (depends on sparsity)!

---

## Advanced: QRESU-Selective

For noisy gradients, use **selective filtering**:

```python
from resu.core.selective import SelectionConfig

# Configure selective updates
config = SelectionConfig(
    beta=0.9,           # EMA smoothing
    tau_stable=0.5,     # Consistency threshold
    k_select_ratio=0.2, # Update top 20% per step
)

# Enter QRESU-Selective mode
for module in model.modules():
    if isinstance(module, RESULinear):
        module.enter_qresu_selective_mode(
            bits=4,
            selective_config=config,
            lr=1e-4,  # Learning rate built into mode
        )

# Train (no optimizer needed for θ - hooks handle it!)
optimizer = torch.optim.Adam(other_params, lr=1e-4)

for epoch in range(20):
    for batch in dataloader:
        loss = train_step(model, batch)
        optimizer.zero_grad()
        loss.backward()  # Selective update applied automatically!
        optimizer.step()
```

**Update Efficiency:** Only ~20% of θ updated per step (filters out noisy gradients)

---

## High-Level Training API

For production use, leverage the built-in training infrastructure:

### Using RESUTrainer

```python
import torch
import torch.nn as nn
from resu.modules.linear import RESULinear
from resu.training.cycle import RESUTrainer
from resu.training.config import RESUConfig

# 1. Define model with RESU layers
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = RESULinear(784, 512)
        self.fc2 = RESULinear(512, 256)
        self.fc3 = RESULinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()

# 2. Configure RESU training
config = RESUConfig(
    target_sparsity=0.7,          # 70% sparse
    num_cycles=5,                 # 5 RESU cycles
    steps_per_cycle=1000,
    use_selective=True,           # Use QRESU-Selective filtering
    commit_strategy="amnesty",    # Amnesty tournament
)

# 3. Define training and evaluation functions
def train_step(model, batch):
    x, y = batch
    logits = model(x)
    return nn.CrossEntropyLoss()(logits, y)

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total}

# 4. Create trainer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = RESUTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    eval_fn=evaluate,  # Optional evaluation
)

# 5. Train across multiple cycles
all_stats = trainer.train(train_loader)

# 6. View results
for stats in all_stats:
    print(f"Cycle {stats.cycle}:")
    print(f"  Sparsity: {stats.actual_sparsity:.1%}")
    print(f"  Resurrected: {stats.n_resurrected} weights")
    print(f"  Training loss: {stats.train_loss:.4f}")

summary = trainer.get_training_summary()
print(f"\nFinal model: {summary['final_sparsity']:.1%} sparse")
print(f"Total resurrected: {summary['total_resurrected']}")
```

**What RESUTrainer handles automatically:**
- Phase orchestration (TRAIN → PRUNE → STABILIZE → RESU → COMMIT)
- Sparsity scheduling across cycles
- Amnesty tournament for weight selection
- Optimizer state cleanup during RESU phase
- Progress tracking and evaluation

---

### Densification Example

Recover from over-pruning using densification mode:

```python
from resu.training.config import densification_config

# Start with a 70% sparse model
# ... train/prune to 70% sparsity ...

# Configure densification
config = densification_config()  # Preset: 70% → 50% → 30% → 10% → 0%

# Override parameters
config.num_cycles = 5
config.steps_per_cycle = 1500
config.amnesty_r_start = 0.30  # More generous resurrection budget

trainer = RESUTrainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=train_step,
    eval_fn=evaluate,
)

# Run densification
stats = trainer.train(train_loader)

# Monitor sparsity decrease
for s in stats:
    print(f"Cycle {s.cycle}: {s.actual_sparsity:.1%} sparsity "
          f"({s.n_resurrected} weights resurrected)")
```

**Output:**
```
Cycle 0: 70.0% sparsity (15342 weights resurrected)
Cycle 1: 50.0% sparsity (12856 weights resurrected)
Cycle 2: 30.0% sparsity (8493 weights resurrected)
Cycle 3: 10.0% sparsity (4127 weights resurrected)
Cycle 4: 0.0% sparsity (2044 weights resurrected)

Model fully recovered to dense!
```

---

### Preset Configurations

Use built-in presets for common scenarios:

```python
from resu.training.config import (
    default_config,
    aggressive_pruning_config,
    conservative_pruning_config,
    densification_config,
    quick_test_config,
)

# Aggressive pruning (90% sparsity)
config = aggressive_pruning_config()
trainer = RESUTrainer(model, config, optimizer, train_step)
trainer.train(train_loader)

# Quick test (2 cycles, 100 steps each)
config = quick_test_config()
trainer = RESUTrainer(model, config, optimizer, train_step)
trainer.train(train_loader)
```

---

## Complete Example (Low-Level)

Here's a full end-to-end example using low-level API:

```python
import torch
import torch.nn as nn
from resu.modules.linear import RESULinear

# 1. Define model with RESU layers
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = RESULinear(784, 512)
        self.fc2 = RESULinear(512, 256)
        self.fc3 = RESULinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()

# 2. Dense pre-training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    for x, y in train_loader:
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 3. Prune to 50% sparsity
for module in model.modules():
    if isinstance(module, RESULinear):
        module.prune_by_magnitude(0.5)

# 4. Enter QRESU mode (memory-optimized)
for module in model.modules():
    if isinstance(module, RESULinear):
        module.enter_qresu_mode(bits=4, epsilon=0.1)

# 5. QRESU fine-tuning
theta_params = [m._theta for m in model.modules()
                if isinstance(m, RESULinear) and m._theta is not None]
optimizer = torch.optim.Adam(theta_params, lr=1e-4)

for epoch in range(20):
    for x, y in train_loader:
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 6. Exit and evaluate
for module in model.modules():
    if isinstance(module, RESULinear):
        module.exit_qresu_mode(commit=True)

# Model is now 50% sparse with recovered accuracy!
```

---

## Memory Tracking

Track memory usage throughout training:

```python
def get_model_memory(model):
    """Get total memory used by model in MB."""
    total_mb = 0
    for name, param in model.named_parameters():
        param_mb = param.numel() * param.element_size() / 1024 / 1024
        total_mb += param_mb
        print(f"{name}: {param_mb:.2f} MB")

    # Add mask memory for RESU layers
    for module in model.modules():
        if isinstance(module, RESULinear) and module._mask is not None:
            mask_mb = (module._mask._indices.numel() *
                      module._mask._indices.element_size() / 1024 / 1024)
            total_mb += mask_mb
            print(f"  └─ mask: {mask_mb:.2f} MB")

    return total_mb

# Track memory at each phase
print("Dense:", get_model_memory(model), "MB")
# ... prune ...
print("Sparse:", get_model_memory(model), "MB")
# ... enter QRESU ...
print("QRESU:", get_model_memory(model), "MB")
```

---

## Pruning Methods

### Magnitude Pruning (Simplest)

```python
layer.prune_by_magnitude(0.5)  # Remove 50% smallest weights
```

### Wanda Pruning (Better - considers activations)

```python
# Capture activations
layer._capture_activations = True
with torch.no_grad():
    outputs = layer(sample_inputs)
activation_norms = layer._activation_norms

# Prune using Wanda
layer.prune_by_wanda(0.5, activation_norms)
```

**Wanda is better** because it considers both weight magnitude AND activation importance.

---

## Common Patterns

### Converting Existing Model

```python
def convert_to_resu(model):
    """Replace all nn.Linear with RESULinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create RESULinear with same parameters
            resu_layer = RESULinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            # Copy weights
            resu_layer.weight.data = module.weight.data
            if module.bias is not None:
                resu_layer.bias.data = module.bias.data

            # Replace
            setattr(model, name, resu_layer)
        else:
            # Recurse into submodules
            convert_to_resu(module)

    return model

# Convert pretrained model
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
model = convert_to_resu(model)
```

---

### Saving/Loading RESU Models

```python
# Save model in RESU mode
torch.save({
    'model_state_dict': model.state_dict(),
    'mode': 'qresu',  # Track what mode we're in
}, 'model_qresu.pt')

# Load and resume
checkpoint = torch.load('model_qresu.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Re-enter QRESU mode if needed
if checkpoint['mode'] == 'qresu':
    for module in model.modules():
        if isinstance(module, RESULinear):
            module.enter_qresu_mode(bits=4)
```

---

## Tips and Best Practices

### 1. **Start with RESU, then try QRESU**
   - RESU is simpler (fewer hyperparameters)
   - QRESU adds memory savings but slightly more complex

### 2. **Use 40-60% sparsity**
   - Sweet spot for accuracy vs memory trade-off
   - Lower sparsity: better accuracy, less memory savings
   - Higher sparsity: more memory savings, harder to recover

### 3. **Initialize θ carefully**
   - `epsilon=0.1` is a good default
   - Too large: training instability
   - Too small: slow convergence

### 4. **Learning rates**
   - Dense phase: 1e-3 (standard)
   - RESU phase: 1e-4 (lower, fine-tuning θ)

### 5. **Selective mode when to use**
   - Use QRESU-Selective when gradients are noisy
   - Regular QRESU is fine for clean tasks

### 6. **Per-channel quantization**
   - Always use `qscheme="per_channel"` for better quality
   - Minimal overhead (~2KB per layer)

---

## Troubleshooting

### Loss explodes during RESU training
**Solution:** Lower `epsilon` or learning rate
```python
layer.enter_resu_mode(epsilon=0.05)  # Try smaller epsilon
optimizer = torch.optim.Adam(params, lr=5e-5)  # Lower LR
```

### QRESU uses more memory than expected
**Check:** Are you at high sparsity (>70%)?
```python
# QRESU works best at 10-50% sparsity
# At 90% sparsity, θ storage dominates!
print(f"Sparsity: {layer._mask.sparsity:.0%}")
```

### Model accuracy doesn't recover
**Try:**
1. More RESU epochs (θ needs time to learn)
2. Lower sparsity (easier recovery)
3. Use Wanda pruning instead of magnitude

---

## Next Steps

- 📖 **[API Reference](API.md)** - Complete API documentation
- 🔬 **[Benchmarks](../benchmarks/)** - Performance comparisons
- 📊 **[Optimization Results](../OPTIMIZATION_RESULTS.md)** - Technical deep-dive
- 💻 **[Examples](../examples/)** - Real-world usage examples

---

## Quick Reference

```python
from resu.modules.linear import RESULinear

# Create layer
layer = RESULinear(512, 256)

# Prune
layer.prune_by_magnitude(0.5)

# Enter mode
layer.enter_resu_mode(epsilon=0.1)           # RESU
layer.enter_qresu_mode(bits=4)               # QRESU (memory-optimized)
layer.enter_qresu_selective_mode(bits=4)     # QRESU-Selective (filtered updates)

# Train (gradients handled automatically via hooks)

# Exit mode
layer.exit_resu_mode(commit=True)
layer.exit_qresu_mode(commit=True)
```

---

**Happy training with RESU! 🚀**
