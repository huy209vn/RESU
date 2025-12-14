# CRITICAL Performance Issues in RESU

## Problem 1: Excessive Memory Usage (200% overhead)

### Root Cause
During RESU phase:
- Active weights are **frozen** (`freeze_active=True`)
- But `W.requires_grad = True` still
- Optimizer still tracks momentum/variance for **entire W matrix**
- PLUS we store θ, m, v, C for resurrection

### Current Memory:
```
Dense params: 64 MB
Optimizer (W): 128 MB  ← WASTED during RESU!
RESU state (θ,m,v,C): 128 MB
Total: 320 MB (200% overhead)
```

### What It Should Be:
```
Dense params: 64 MB
Optimizer (θ only): 64 MB  ← Only for pruned positions!
RESU state: 0 MB  ← Reuse optimizer states!
Total: 128 MB (0% overhead during RESU!)
```

### Fix Strategy:
1. **Disable W gradients** during RESU:
   ```python
   # Enter RESU mode
   self.weight.requires_grad_(False)  # Freeze W
   ```

2. **Remove W from optimizer**, add θ:
   ```python
   # Create new optimizer param group with only θ
   optimizer.add_param_group({'params': [theta], 'lr': resu_lr})
   ```

3. **Reuse optimizer states for θ**:
   - Don't create separate m, v, C
   - Let AdamW optimizer handle θ's momentum/variance
   - Only need to track directional consistency C separately (if using selective)

---

## Problem 2: Slow RESU Forward/Backward (0.14-0.28x)

### Root Cause
Current forward pass:
```python
# During RESU
W_eff = M⊙W + (1-M)⊙Φ(θ)  # Computes full dense matrix
out = F.linear(x, W_eff, bias)
```

Problems:
1. **Computing M⊙W every forward** (even though W is frozen!)
2. **Full dense matmul** with W_eff
3. **Scatter/gather overhead** for Φ(θ)

### Fix Strategy:

#### Option A: Cache Active Part
```python
class RESULinear:
    def enter_resu_mode(self):
        # Cache frozen active weights
        self._W_active_cached = self._mask.apply(self.weight.data.clone())
        self._W_active_cached.requires_grad = False

    def forward(self, x):
        if self._mode == RESUMode.RESU:
            # Don't recompute M⊙W, use cache!
            phi_theta = self._resurrection.phi(theta)
            W_eff = self._W_active_cached + phi_theta
            return F.linear(x, W_eff, bias)
```

**Speedup**: Eliminates one masked multiplication per forward

#### Option B: Sparse Ops for θ Only
```python
def forward(self, x):
    if self._mode == RESUMode.RESU:
        # Active part (cached)
        out_active = F.linear(x, self._W_active_cached, None)

        # Resurrection part (sparse if beneficial)
        if self.sparsity > 0.7:
            out_resurrected = self._forward_sparse_theta(x, theta)
        else:
            phi_theta = self._resurrection.phi(theta)
            out_resurrected = F.linear(x, phi_theta, None)

        return out_active + out_resurrected + bias
```

**Speedup**: Decomposes into active (cached) + sparse resurrection

---

## Proposed Implementation Plan

### Step 1: Fix Memory (Critical!)

**File**: `resu/modules/linear.py`

```python
def enter_resu_mode(self, ...):
    # Freeze W completely
    self.weight.requires_grad_(False)

    # Cache active weights (won't change during RESU)
    self._W_active_cached = self._mask.apply(self.weight.data).clone()

    # Create resurrection embedding
    self._resurrection = ResurrectionEmbedding(...)
    self._resurrection.theta.requires_grad_(True)  # Only θ needs grads!

    # NOTE: User must update optimizer manually!
    # See training/cycle.py for how to do this

def exit_resu_mode(self, commit=True):
    # Re-enable W gradients
    self.weight.requires_grad_(True)

    # Clear cache
    self._W_active_cached = None
```

**File**: `resu/training/cycle.py`

```python
def resu_phase(self, dataloader):
    # Collect θ parameters from all RESU modules
    theta_params = []
    for module in self.resu_modules.values():
        if module.is_resu_active:
            theta_params.append(module.resurrection.theta)

    # Create temporary optimizer for θ ONLY
    resu_optimizer = torch.optim.AdamW(
        theta_params,
        lr=self.config.resu_lr,
        weight_decay=self.config.weight_decay,
    )

    for step in range(self.config.resu_steps):
        # Forward/backward
        loss = self.train_fn(self.model, batch)
        loss.backward()

        # Update ONLY θ
        resu_optimizer.step()
        resu_optimizer.zero_grad()
```

### Step 2: Fix Speed

**File**: `resu/modules/linear.py`

```python
def forward(self, x, freeze_active=True, use_sparse=True):
    if self._mode == RESUMode.RESU:
        # Use cached active weights
        phi_theta = self._resurrection.phi(self._resurrection.theta)
        W_eff = self._W_active_cached + phi_theta
        return F.linear(x, W_eff, self.bias)
```

---

## Expected Improvements

### Memory:
**Before**: 200% overhead
**After**: ~0-50% overhead (only from selective state C if used)

### Speed:
**Before**: 0.14-0.28x vs dense
**After**: ~0.8-1.0x vs dense (cached active part eliminates overhead)

---

## Implementation Checklist

- [ ] Modify `enter_resu_mode()` to disable W gradients
- [ ] Add W caching in `enter_resu_mode()`
- [ ] Update `exit_resu_mode()` to re-enable gradients
- [ ] Modify `resu_phase()` to create separate optimizer for θ
- [ ] Update forward pass to use cached W_active
- [ ] Test memory usage decreases
- [ ] Test speed improves
- [ ] Ensure all 56 tests still pass

---

This is **CRITICAL** - without these fixes, RESU is not practical for real use!
