# RESU Refactoring Plan

**Goal:** Match paper's theoretical guarantees:
- Zero additional memory
- ~1.2× speed overhead (not 9×!)

---

## Current State (VIOLATIONS)

### Memory: 5.5× worse than paper expects
```
Expected: 0.50 MB (same as dense)
Actual:   2.75 MB (5.5×)
Overhead: +2.25 MB
```

**Breakdown:**
- ❌ Mask stored as dense (0.50 MB)
- ❌ Indices duplicated (1.00 MB)
- ❌ θ separate allocation (0.25 MB)
- ❌ W_active_cached useless (0.50 MB)

### Speed: 9.34× slower than dense
```
Expected: ~0.01 ms (1.2× dense)
Actual:   0.08 ms (9.34× dense)
```

**Bottleneck:** Φ(θ) scatter = 0.0496 ms (59% of forward pass)

---

## Root Causes

### 1. Storage Architecture
**Current:** Everything uses dense storage
```python
W:     dense tensor (out × in)
mask:  dense tensor (out × in) ← WRONG
θ:     separate dense tensor   ← WRONG
```

**Paper expects:** In-place storage
```python
W:     dense tensor where W[pruned] = θ  ← θ stored IN pruned positions
mask:  sparse indices only              ← Not full dense tensor
```

### 2. Φ(θ) Implementation
**Current:** Triton kernel `phi_scatter`
- Launches separate kernel
- Random memory access pattern
- No fusion with downstream ops
- **Time: 0.0496 ms** (5.5× full dense layer!)

**Better approach:**
```python
# Option A: In-place (no scatter needed)
W_eff = W.clone()  # θ already in pruned positions

# Option B: Fused computation
out = x @ W  # θ values flow through naturally
```

### 3. Optimizer Integration
**Current:** θ has separate optimizer states
```python
self._selective._m  # New allocation
self._selective._v  # New allocation
self._selective._C  # New allocation
```

**Paper expects:** Reuse pruned slots
```python
optimizer.state[W]['momentum'][pruned_positions] = m_θ
optimizer.state[W]['variance'][pruned_positions] = v_θ
```

---

## Refactoring Strategy

### Phase 1: In-Place θ Storage (Memory Fix)

**Goal:** θ lives in W's pruned positions. Zero new memory.

**Changes:**

1. **Remove separate θ tensor**
   ```python
   # OLD:
   self._resurrection = ResurrectionEmbedding(...)  # Separate storage

   # NEW:
   # θ stored directly in self.weight[pruned_positions]
   ```

2. **Modify enter_resu_mode()**
   ```python
   def enter_resu_mode(self, epsilon=0.1):
       # Initialize θ in pruned positions
       pruned_mask = ~self._mask.mask.bool()
       self.weight.data[pruned_mask] = torch.randn_like(
           self.weight.data[pruned_mask]
       ) * epsilon * active_std

       # Mark only pruned positions as trainable
       self.weight.register_hook(self._prune_grad_hook)
   ```

3. **Gradient masking hook**
   ```python
   def _prune_grad_hook(self, grad):
       # Only let gradients flow to pruned positions
       return grad * (~self._mask.mask)
   ```

**Expected savings:** 0.25 MB (θ) + 0.50 MB (cached W) = **0.75 MB**

### Phase 2: Eliminate Φ(θ) Scatter (Speed Fix)

**Goal:** No scatter operation. θ flows through naturally.

**Current forward:**
```python
W_active = M ⊙ W              # Extract active
phi_theta = Φ(θ)              # Scatter θ (SLOW!)
W_eff = W_active + phi_theta  # Add
out = x @ W_eff               # Matmul
```

**New forward (in-place θ):**
```python
# W already contains:
#   - Active weights at active positions
#   - θ values at pruned positions
W_eff = W  # No scatter needed!
out = x @ W_eff
```

**Expected speedup:** Eliminate 0.0496 ms overhead → **~9× faster**

### Phase 3: Sparse Mask Storage (Memory Fix)

**Goal:** Mask uses sparse indices, not dense tensor.

**Current:**
```python
mask.mask:            dense (out × in)  → 0.50 MB
mask.active_indices:  (n_active, 2)     → 0.50 MB
mask.pruned_indices:  (n_pruned, 2)     → 0.50 MB
Total:                                     1.50 MB
```

**Optimized:**
```python
# Store only pruned indices (smaller at high sparsity)
mask.pruned_indices: (n_pruned, 2)  → 0.50 MB at 50% sparsity
# Derive active from NOT pruned when needed
```

For 50% sparsity: **Save 1.00 MB**

### Phase 4: Optimizer State Reuse (Selective)

**Goal:** m, v, C reuse pruned slots in main optimizer.

**Current:**
```python
# RESUSelective allocates new tensors
self._m = torch.zeros_like(theta)  # New
self._v = torch.zeros_like(theta)  # New
self._C = torch.zeros_like(theta)  # New
```

**New approach:**
```python
# Reuse optimizer's state dict for W
optimizer.state[W]['momentum'][pruned_positions] = m_θ
optimizer.state[W]['variance'][pruned_positions] = v_θ
# Store C in a separate small tensor (only p elements)
```

**Expected savings:** 0.50 MB (m, v, C)

---

## Implementation Plan

### Step 1: Proof of Concept (In-Place θ)
- Create `RESULinearV2` with in-place storage
- Test forward/backward correctness
- Measure memory/speed improvement
- **Expected:** 0.75 MB memory, 5× speed improvement

### Step 2: Optimize Mask Storage
- Refactor `SparseMask` to use sparse indices only
- Remove redundant dense storage
- **Expected:** 1.00 MB memory savings

### Step 3: Optimizer Integration
- Hook into PyTorch optimizer state dict
- Reuse pruned slots for θ states
- **Expected:** 0.50 MB memory savings

### Step 4: Validate Against Paper
- Run full training cycle
- Verify memory = dense baseline
- Verify speed ≈ 1.2× dense
- Compare to paper's Algorithm 1

---

## Expected Final State

### Memory
```
Current:  2.75 MB (5.5× dense)
Phase 1:  2.00 MB (eliminate θ, cache)
Phase 2:  2.00 MB (no change, speed only)
Phase 3:  1.00 MB (sparse mask)
Phase 4:  0.50 MB (optimizer reuse)
Target:   0.50 MB (same as dense) ✓
```

### Speed
```
Current:   0.0840 ms (9.34× dense)
Phase 1:   0.0840 ms (no change)
Phase 2:   0.0100 ms (eliminate scatter) ← KEY FIX
Phase 3:   0.0100 ms (no change)
Phase 4:   0.0100 ms (no change)
Target:    ~0.011 ms (1.2× dense) ✓
```

---

## Risks & Challenges

### 1. PyTorch Optimizer API
**Problem:** Optimizers expect uniform treatment of all parameters.

**Challenge:** We need only pruned positions of W updated.

**Solutions:**
- Custom optimizer wrapper
- Gradient hook to mask active positions
- Per-parameter learning rate groups

### 2. Autograd Graph Complexity
**Problem:** In-place ops can break autograd.

**Challenge:** θ and W share storage but have different grad requirements.

**Solutions:**
- Careful use of `torch.no_grad()` for active positions
- Custom backward hooks
- Test gradient correctness extensively

### 3. Framework Assumptions
**Problem:** PyTorch assumes dense, contiguous, trainable.

**Challenge:** Our setup violates these assumptions.

**Solutions:**
- Mark W as trainable but mask gradients
- Ensure contiguous memory layout
- Avoid in-place ops that break graph

---

## Success Criteria

✅ **Memory:** RESU uses ≤ 0.50 MB (same as dense)
✅ **Speed:** RESU forward ≤ 0.011 ms (1.2× dense)
✅ **Correctness:** All 56 tests pass
✅ **Training:** Converges on CIFAR-10 with amnesty

---

## Next Steps

1. **Implement Phase 1** (in-place θ storage)
2. **Benchmark** against current implementation
3. **Validate** gradient correctness
4. **Iterate** based on results

Once Phases 1-2 are done, RESU will be:
- **2.75× less memory**
- **8× faster**

Then QRESU becomes viable.
