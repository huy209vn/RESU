# RESU Memory Optimization Log

**Date:** 2024-12-18

## Problem
RESU peak memory was 242.8 MB vs Dense 146.8 MB (1.65× overhead).
Paper claims "zero additional memory" but we had ~96 MB overhead.

## Root Causes
1. `_grad_mask` stored as float32 (67 MB)
2. `SparseMask._indices` kept after RESU mode entry (32 MB)

## Fixes Applied

### 1. Bool gradient mask (saves 50 MB)
```python
# Before: float32
self._grad_mask = 1.0 - dense_mask  # 67 MB

# After: bool
self._grad_mask = ~dense_mask.bool()  # 17 MB
```

Hook changed from `grad.mul_()` to `grad.masked_fill_()`:
```python
def grad_mask_hook(grad):
    return grad.masked_fill_(~self._grad_mask, 0)
```

### 2. Clear SparseMask indices after setup (saves 32 MB)
```python
# Store stats before clearing
self._n_pruned = self._mask.n_pruned
self._n_active = self._mask.n_active

# Clear indices (32 MB at 50% sparsity)
self._mask._indices = torch.tensor([], dtype=torch.int32, device=self.device)
```

Added rebuild logic for re-entering RESU mode:
```python
if len(self._mask._indices) == 0:
    pruned_indices = (self.weight.data == 0).flatten().nonzero(as_tuple=True)[0]
    self._mask = SparseMask(pruned_indices, self.weight.shape, device=self.device)
```

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Peak memory | 242.8 MB | 178.3 MB | -64.5 MB |
| vs Dense | 1.65× | 1.21× | -0.44× |
| Stored overhead | 96 MB | 16 MB | -80 MB |

### Final Storage Breakdown
- Weight: 64 MB
- _grad_mask (bool): 16 MB
- **Total: 80 MB** (just 16 MB overhead)

Peak includes unavoidable runtime: weight.grad (64 MB) + autograd temps (~34 MB).

## Files Modified
- `resu/modules/linear.py`: enter_resu_mode, enter_resu_mode_structured, exit_resu_mode
