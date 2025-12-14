# RESU Performance Fixes Applied

## ✅ Critical Memory Fix Implemented

### What We Fixed
**Problem**: During RESU phases, optimizer was tracking states (momentum, variance) for **both** W and θ, causing 200% memory overhead.

**Root Cause**:
- `W.requires_grad = True` even though W was frozen
- Optimizer allocated states for all parameters with `requires_grad=True`
- Result: 2× memory (W states + θ states)

**Solution Implemented**:
```python
# In enter_resu_mode():
self.weight.requires_grad_(False)  # CRITICAL: Disable W gradients
self._resurrection.requires_grad = True  # Only θ needs gradients

# In exit_resu_mode():
self.weight.requires_grad_(True)  # Re-enable for next training phase
```

### Expected Memory Improvement

**Before Fix**:
```
Active weights W: frozen but requires_grad=True
Optimizer states: Allocated for W (wasted!)
Resurrection θ: Has own optimizer states
Total overhead: ~200%
```

**After Fix**:
```
Active weights W: frozen and requires_grad=False
Optimizer states: NOT allocated for W ✓
Resurrection θ: Has optimizer states (necessary)
Total overhead: ~50% (only from θ states, which we need!)
```

**For 50% sparsity**:
- θ size = 0.5 × W size
- θ optimizer states (Adam) = 2 × θ size = 1.0 × W size
- **Expected memory: ~100% of base (down from 200%)**

---

## Status: NEEDS BENCHMARKING

### ✅ What's Done
1. **Code Changes Applied**
   - `W.requires_grad_(False)` in `enter_resu_mode()`
   - `W.requires_grad_(True)` in `exit_resu_mode()`
   - Cached `_W_active` for future optimization
   - All 56 tests passing

2. **Gradient Flow Verified**
   - θ still receives gradients correctly
   - W correctly has no gradients during RESU
   - Autograd still works properly

### ⏳ What You Need To Do

**Run benchmarks on YOUR GPU**:
```bash
# Memory benchmark
python -m benchmarks.bench_memory

# Throughput benchmark
python -m benchmarks.bench_throughput
```

**What to look for**:
1. **Memory**: RESU state should now be ~50-100% overhead (down from 200%)
2. **Speed**: Should be similar to before (we didn't change computation)

---

## Additional Optimization Opportunity

### Future: Use Cached Active Weights

We're currently caching `_W_active` but not using it. Here's the optimization:

**Current** (after our fix):
```python
# Every forward pass
W_eff = effective_weight(self.weight, theta, mask)  # Recomputes M⊙W
out = F.linear(x, W_eff, bias)
```

**Potential optimization**:
```python
# In enter_resu_mode: Already cached
self._W_active_cached = M⊙W  # Done once

# Every forward pass
W_eff = self._W_active_cached + Φ(θ)  # No recomputation!
out = F.linear(x, W_eff, bias)
```

**Issue**: Need to ensure gradients flow correctly to θ. The `effective_weight` autograd function handles this automatically, but manual computation needs careful gradient routing.

**Recommendation**: Benchmark first. If RESU forward is still slow after memory fix, implement this optimization with custom autograd function.

---

## Files Modified

1. `resu/modules/linear.py`:
   - Added `W.requires_grad_(False)` in `enter_resu_mode()` (line 265)
   - Added `W.requires_grad_(True)` in `exit_resu_mode()` (line 314)
   - Added `_W_active_cached` initialization (line 103)
   - Cached active weights in `enter_resu_mode()` (line 268)

---

## Next Steps

### 1. Benchmark (YOU DO THIS)
```bash
python -m benchmarks.bench_memory
python -m benchmarks.bench_throughput
```

### 2. If Memory Is Still High
Check that:
- Optimizer is **NOT** configured with W during RESU phases
- Only θ parameters are in optimizer during RESU
- W has `requires_grad=False` during RESU (verify in debugger)

### 3. If Speed Is Still Slow
Potential causes:
- `effective_weight()` autograd overhead
- Dense matmul on GPU not utilizing sparsity
- Batch size too small

Solutions:
- Implement cached W_active optimization (see above)
- Use larger batch sizes
- Profile with `torch.profiler` to find bottleneck

### 4. Verify Training Still Works
Run a full training cycle:
```python
from resu.training.cycle import RESUTrainer

# Train for a few cycles
stats = trainer.train(dataloader)

# Check:
# - Resurrection happens (stats.n_resurrected > 0)
# - Loss decreases
# - Sparsity maintained
```

---

## Expected Results

### Memory (Target)
| Configuration | Before | After | Improvement |
|---------------|--------|-------|-------------|
| Dense params | 64 MB | 64 MB | - |
| RESU state (50% sparse) | 128 MB | 32-64 MB | **50-75%** |
| Total | 192 MB | 96-128 MB | **33-50%** |

### Speed (Target)
| Mode | Before | After | Change |
|------|--------|-------|--------|
| Dense | 1.0x | 1.0x | No change |
| Sparse | 0.3-0.8x | 0.3-0.8x | No change |
| **RESU** | **0.14-0.28x** | **0.5-0.8x** | **Goal: 2-3× faster** |

*Note: Speed improvement depends on implementing cached active weights optimization*

---

## Summary

✅ **Critical memory fix applied and tested**
- `W.requires_grad=False` during RESU prevents wasted optimizer states
- All tests pass
- Ready for benchmarking

⏳ **You need to benchmark** to verify improvements

🚀 **Future optimization available** (cached active weights) if needed

---

**Status**: All code changes done. **Your turn to benchmark!**
