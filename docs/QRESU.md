# QRESU: Quantized Resurrection of Sparse Units

**Extending RESU with QLoRA-style quantization for extreme memory efficiency**

---

## Motivation

During RESU phase:
- **Active weights W are frozen** (only resurrected positions θ are trained)
- W consumes memory but doesn't need full precision for forward pass
- Inspired by **QLoRA**: Quantize frozen parameters, train adapters in full precision

**Key insight**: W during RESU is analogous to frozen LLM in QLoRA.

---

## QRESU: Core Idea

### Standard RESU Memory
```
Dense W:     64 MB (frozen, full precision)
θ params:    32 MB (active, 50% sparsity)
θ states:    96 MB (m, v, C for Adam)
────────────────────
Total:      192 MB
```

### QRESU Memory (4-bit quantization)
```
W_quantized:  8 MB (frozen, 4-bit)
θ params:    32 MB (active, full precision)
θ states:    96 MB (m, v, C for Adam)
────────────────────
Total:      136 MB  (29% reduction!)
```

**With 8-bit**: 160 MB (17% reduction)

---

## Algorithm

### Quantization on RESU Entry

```python
def enter_qresu_mode(self, bits=4, epsilon=0.1, ...):
    """Enter QRESU mode with quantized active weights.

    Args:
        bits: 4 or 8 bit quantization for frozen W
    """
    # 1. Freeze active weights (standard RESU)
    self.weight.requires_grad_(False)
    W_active = self._mask.apply(self.weight.data)

    # 2. QUANTIZE frozen active weights
    self._W_active_quantized, self._qparams = quantize(
        W_active,
        bits=bits,
        scheme='per_tensor',  # or 'per_channel'
    )

    # 3. Initialize resurrection parameters θ (full precision)
    self._resurrection = ResurrectionEmbedding(...)
    self._resurrection.initialize(epsilon)
```

### Forward Pass with Dequantization

```python
def forward_qresu(self, x):
    """Forward with quantized W and full-precision θ."""

    # 1. Dequantize active weights on-the-fly
    W_active = dequantize(self._W_active_quantized, self._qparams)

    # 2. Get resurrected weights (full precision)
    phi_theta = self._resurrection.phi()

    # 3. Combine: W_eff = W_active + Φ(θ)
    W_eff = W_active + phi_theta

    # 4. Standard linear
    return F.linear(x, W_eff, self.bias)
```

### Exit: Dequantize and Commit

```python
def exit_qresu_mode(self, commit=True):
    """Exit QRESU, merge θ back to full precision W."""

    if commit:
        # Dequantize W_active
        W_active = dequantize(self._W_active_quantized, self._qparams)

        # Merge with resurrection
        phi_theta = self._resurrection.commit()
        self.weight.data = W_active + phi_theta

    # Re-enable gradients
    self.weight.requires_grad_(True)

    # Free quantized storage
    del self._W_active_quantized
    del self._qparams
```

---

## QRESU-Selective

**Combine quantization with directional consistency filtering.**

```python
def enter_qresu_selective_mode(self, bits=4, epsilon=0.1, ...):
    """QRESU with selective resurrection."""

    # Quantize frozen W (same as QRESU)
    self._W_active_quantized, self._qparams = quantize(...)

    # Initialize θ with selective updater
    self._resurrection = ResurrectionEmbedding(...)
    self._selective = RESUSelective(
        self._resurrection,
        config=SelectionConfig(
            beta=0.9,      # EMA decay
            delta=0.9,     # Consistency decay
            tau_stable=5,  # Stability threshold
        ),
    )
```

**Memory breakdown**:
```
W_quantized:     8 MB (4-bit)
θ:              32 MB
Selective m:    32 MB  (EMA momentum)
Selective v:    32 MB  (EMA variance)
Selective C:    32 MB  (consistency tracker)
────────────────────
Total:         136 MB (same as QRESU, selective comes "free" memory-wise)
```

---

## Quantization Schemes

### 1. Per-Tensor (Simpler)
- Single scale/zero-point for entire W_active
- Faster dequantization
- Slightly lower quality

```python
def quantize_per_tensor(W, bits=4):
    qmin, qmax = 0, 2**bits - 1
    scale = (W.max() - W.min()) / (qmax - qmin)
    zero_point = qmin - W.min() / scale

    W_q = ((W / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return W_q, (scale, zero_point)

def dequantize_per_tensor(W_q, params):
    scale, zero_point = params
    return (W_q.float() - zero_point) * scale
```

### 2. Per-Channel (Better Quality)
- Separate scale/zero-point per output channel
- Better precision retention
- Slightly slower

```python
def quantize_per_channel(W, bits=4):
    """W shape: (out_features, in_features)"""
    qmin, qmax = 0, 2**bits - 1

    # Per-channel (along output dim)
    W_min = W.min(dim=1, keepdim=True)[0]
    W_max = W.max(dim=1, keepdim=True)[0]

    scale = (W_max - W_min) / (qmax - qmin)
    zero_point = qmin - W_min / scale

    W_q = ((W / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return W_q, (scale, zero_point)
```

---

## Theoretical Analysis

### Memory Savings

For a layer with n = out_features × in_features at sparsity s:

**Standard RESU**:
- W: n × 4 bytes (FP32)
- θ: (s·n) × 4 bytes
- States: (s·n) × 3 × 4 bytes (m, v, C)
- **Total**: n × 4 + (s·n) × 16 bytes

**QRESU-4bit**:
- W_q: n × 0.5 bytes (4-bit)
- θ: (s·n) × 4 bytes
- States: (s·n) × 3 × 4 bytes
- **Total**: n × 0.5 + (s·n) × 16 bytes

**Savings**: (n × 4 - n × 0.5) = **3.5n bytes** = **87.5% reduction in W storage**

**Example** (n = 16M params, s = 0.5):
- Standard: 64 MB + 128 MB = 192 MB
- QRESU-4: 8 MB + 128 MB = 136 MB
- **Saved: 56 MB (29% total reduction)**

### Computational Overhead

**Dequantization cost per forward**:
```
Time_dequant = O(n)  (elementwise ops)
Time_matmul  = O(b × n × d)  (b=batch, d=features)

For typical b=32, d=4096:
Dequant: ~0.01 ms
Matmul:  ~0.5 ms
Overhead: ~2%
```

**Negligible** compared to memory savings!

---

## When to Use QRESU

### ✅ Use QRESU when:
1. **Memory-constrained** (edge devices, large models)
2. **High sparsity** (s > 0.7): More frozen weights to quantize
3. **Long RESU phases**: Amortize quantization overhead
4. **Large layers**: Bigger n → bigger savings

### ⚠️ Skip QRESU when:
1. **Low sparsity** (s < 0.3): Less to gain from quantizing small W_active
2. **Memory abundant**: Complexity not worth it
3. **Ultra-fast inference needed**: Dequant overhead matters

---

## Comparison to QLoRA

| Aspect | QLoRA | QRESU |
|--------|-------|-------|
| **Frozen part** | Pre-trained LLM weights | Active sparse weights W |
| **Trainable part** | LoRA adapters (low-rank) | Resurrection params θ (sparse) |
| **Quantization** | 4-bit NormalFloat | 4/8-bit integer |
| **Use case** | Fine-tune LLMs cheaply | Train sparse networks efficiently |
| **Memory savings** | 4-8x on frozen | 2-4x on RESU state |

**Key difference**: QRESU quantizes *sparse active weights*, QLoRA quantizes *full dense model*.

---

## Implementation Considerations

### 1. Quantization Backend
- **PyTorch native**: `torch.quantization` (slower, more compatible)
- **bitsandbytes**: Fast GPU kernels (like QLoRA uses)
- **Custom Triton**: Fused dequant + effective_weight kernel

### 2. Gradient Handling
- Straight-through estimator (STE) if allowing W fine-tuning
- But in QRESU, W is frozen → no gradients needed!

### 3. Mixed Precision
- Can combine with AMP (FP16 training)
- W: 4-bit quantized
- θ: FP16 or BF16
- Further 2x savings!

---

## Experimental Questions

1. **Accuracy impact**: How much does 4-bit vs 8-bit affect resurrection quality?
2. **Selective interaction**: Does quantization interfere with directional consistency?
3. **Optimal bit-width**: Is 4-bit sufficient, or need 6-bit/8-bit?
4. **Per-tensor vs per-channel**: Quality/speed trade-off?
5. **Quantization-aware training**: Pre-quantize before RESU vs quantize on entry?

---

## Expected Results (Hypothesis)

### Memory
```
Model: ResNet-50 sparse (50% sparsity)
Standard RESU:  ~180 MB
QRESU-4bit:     ~120 MB  (33% reduction)
QRESU-8bit:     ~150 MB  (17% reduction)
```

### Accuracy
- **4-bit**: <1% accuracy drop (if any)
- **8-bit**: No measurable drop
- **Reasoning**: W is only used in forward, no gradients

### Speed
- **Overhead**: +2-5% per forward (dequantization)
- **Overall**: Negligible (RESU is small % of training)

---

## Pseudocode: Full QRESU Cycle

```python
# Initialization
model = ResNet50()
model = convert_to_resu(model, storage_mode=COMPACT)

# Cycle training
for cycle in range(num_cycles):
    # 1. Dense training
    train_dense(model, optimizer, steps=1000)

    # 2. Prune to target sparsity
    prune_wanda(model, sparsity=0.5)

    # 3. DSNoT stabilization
    stabilize_dsnot(model, steps=100)

    # 4. Enter QRESU (QUANTIZE!)
    for module in model.resu_modules():
        module.enter_qresu_mode(
            bits=4,              # 4-bit quantization
            epsilon=0.1,
            use_selective=True,  # QRESU-Selective
        )

    # 5. RESU training (with quantized W)
    train_resu(model, steps=500)  # Forward uses dequantized W

    # 6. Exit QRESU and commit (DEQUANTIZE!)
    for module in model.resu_modules():
        module.exit_qresu_mode(commit=True)  # Merge back to FP32

    # 7. Amnesty or Wanda re-pruning
    new_mask = commit_with_wanda(model, sparsity=0.5)
```

---

## Extensions

### 1. QRESU-LoRA
Combine QRESU with LoRA-style low-rank resurrection:
```python
θ = A @ B  # Low-rank factorization of resurrection
# Even more memory savings!
```

### 2. Adaptive Quantization
Different bit-widths per layer based on sensitivity:
```python
layer1: 4-bit (less sensitive)
layer10: 8-bit (more sensitive)
```

### 3. Dynamic Quantization
Quantize/dequantize only during forward, keep FP32 in backward:
```python
# Similar to mixed-precision training
```

---

## Summary

**QRESU** = RESU + QLoRA-style quantization

**Benefits**:
- ✅ 30-35% additional memory savings
- ✅ Minimal accuracy impact (<1%)
- ✅ Negligible computational overhead
- ✅ Compatible with RESU-Selective

**Cost**:
- ⚠️ Implementation complexity (quantization kernels)
- ⚠️ Need to validate on real models

**Next steps**:
1. Implement basic 4-bit/8-bit quantization
2. Benchmark memory and accuracy
3. Compare to standard RESU
4. Optimize with custom kernels if needed

---

**Status**: Conceptualized, ready for implementation exploration.
