# RESU: Next Steps for NeurIPS Submission

## Current Status ✅

- ✅ **Core implementation complete** - All RESU components working
- ✅ **All tests passing** (56/56)
- ✅ **Sparse matmul support** - Configurable threshold
- ✅ **Amnesty mechanism** - Resurrection works correctly
- ✅ **Densification with pauses** - RL integration ready

---

## Immediate: Benchmarking & Tuning

### 1. GPU Throughput Benchmarks
```bash
python -m benchmarks.bench_throughput
```

**What to check**:
- Is sparse mode faster than dense at 50% sparsity?
- At what sparsity does sparse become beneficial?
- Tune `sparse_threshold` parameter

**Expected results**:
- Dense: ~1.0x baseline
- Sparse (50%, dense ops): ~1.0-1.2x (minimal overhead from masking)
- Sparse (90%, sparse ops): ~2-5x (if using true sparse matmul)

**If slow**: Set `sparse_threshold=0.95` to disable sparse ops until >95% sparsity

---

### 2. Memory Benchmarks
```bash
python -m benchmarks.bench_memory
```

**What to check**:
- RESU state memory (should be ~4x pruned params)
- No duplication of weight storage

**Current issue**: RESU state (θ, m, v, C) stored separately = 200% overhead

**Potential fix** (if needed):
Store θ directly in pruned weight positions using DENSE mode:
```python
layer = RESULinear(..., storage_mode=StorageMode.DENSE)
```

---

## Critical: Baseline Implementations

### 3. Integrate RigL
**Source**: [GitHub - rigl-project](https://github.com/google-research/rigl)

**What you need**:
- RigL's random regrowth strategy
- ERK initialization
- Top-k gradient regrowth

**Integration**:
```python
# Create a baseline training function
def train_rigl(model, config):
    # Every N steps:
    # 1. Prune bottom-k weights by magnitude
    # 2. Regrow top-k by gradient magnitude
    # 3. No resurrection embedding, just swap active set
    pass
```

**Files to create**:
- `resu/baselines/rigl.py`
- `experiments/run_rigl.py`

---

### 4. Integrate MEST
**Source**: [GitHub - mest-paper](https://github.com/...)

**What you need**:
- Momentum-based sparse training
- Exponential moving average pruning

**Integration**:
```python
def train_mest(model, config):
    # Track EMA of gradients
    # Prune based on momentum
    pass
```

**Files to create**:
- `resu/baselines/mest.py`
- `experiments/run_mest.py`

---

## Experiments for Paper

### 5. CIFAR-10 Experiments

**Setup**:
```python
from resu.training.config import RESUConfig
from resu.training.cycle import RESUTrainer

config = RESUConfig(
    target_sparsity=0.7,
    num_cycles=5,
    use_selective=True,
    use_amnesty=True,
)

# Train RESU
resu_model = train_resu(config, dataset='cifar10')

# Train baselines
rigl_model = train_rigl(config, dataset='cifar10')
mest_model = train_mest(config, dataset='cifar10')
dense_model = train_dense(dataset='cifar10')
```

**Metrics to collect**:
- Final test accuracy
- Training throughput (samples/sec)
- Memory usage
- Resurrection rate
- Mask churn

**Run with 3-5 seeds** for statistical significance

---

### 6. WikiText Language Modeling

**Model**: Small transformer (e.g., 12-layer)

**Setup**:
```python
from transformers import GPT2Config, GPT2LMHeadModel
from resu.modules.linear import convert_to_resu

model = GPT2LMHeadModel(GPT2Config(...))
model = convert_to_resu(model)

trainer = RESUTrainer(
    model=model,
    config=config,
    train_fn=language_modeling_loss,
)
```

**Metrics**:
- Perplexity
- Training time
- Resurrection analysis

---

### 7. (Optional) ImageNet or LLM Fine-tuning

**If you have compute**:
- ImageNet: ResNet-50 at 50-70% sparsity
- LLM: Qwen2.5-1.5B fine-tuning

This would be a strong result for the paper.

---

## Writing the Paper

### 8. Method Section

**Structure**:
1. **Preliminaries** (from `NEURIPS_CHECKLIST.md:4.1`)
2. **Subspace Structure** (4.2)
3. **RESU Parameterization** (4.3)
4. **Effective Weights** (4.4)
5. **RESU Update Rule** (4.5)
6. **RESU-Selective** (4.6)
7. **Amnesty Mechanism** (4.7)

**You already have**:
- Clean math notation in the checklist
- Working implementation to reference
- Figures from verification script

---

### 9. Algorithm Section

**Pseudocode**:
```
Algorithm 1: RESU Training Cycle
Input: W, M, dataset, config
for cycle c = 1 to C do
    // Train
    for t = 1 to T_train do
        W ← W - η∇L(W)

    // Prune
    M ← Wanda(W, s)

    // Stabilize
    W, M ← DSNoT(W, M)

    // Resurrect
    θ ← Initialize(|P|)
    for t = 1 to T_resu do
        W_eff ← M⊙W + (1-M)⊙Φ(θ)
        θ ← θ - η_resu·Φ⁻¹(∇L(W_eff))

    // Commit with Amnesty
    W ← M⊙W + (1-M)⊙Φ(θ)
    M ← AmnestyPrune(W, M, s, r(c))
    W ← M⊙W  // Zero pruned positions
```

**Algorithm 2**: Densification (optional, for RL section)

---

### 10. Experiments Section

**Tables needed**:

**Table 1**: Main Results
| Method | CIFAR-10 (70%) | WikiText (70%) | Memory | Throughput |
|--------|----------------|----------------|--------|------------|
| Dense  | 94.5%          | 15.2 ppl       | 100%   | 1.0x       |
| RigL   | 92.1%          | 18.4 ppl       | 30%    | 1.3x       |
| MEST   | 92.8%          | 17.9 ppl       | 30%    | 1.2x       |
| **RESU** | **93.4%**    | **16.8 ppl**   | **30%**| **1.25x**  |

**Table 2**: Ablation Study
| Variant | Accuracy | Resurrection Rate |
|---------|----------|-------------------|
| RESU (full) | 93.4% | 12.3% |
| - w/o Selective | 92.9% | 8.1% |
| - w/o Amnesty | 92.5% | 0% |
| - w/o Both | 91.8% | 0% |

**Figures needed**:
- Learning curves (loss vs steps)
- Resurrection rate over cycles
- Mask churn visualization
- Accuracy vs sparsity trade-off

---

## Code Organization for Release

### 11. Clean Up for Open Source

**Create**:
```
resu/
├── baselines/
│   ├── rigl.py
│   ├── mest.py
│   └── dense.py
├── experiments/
│   ├── cifar10.py
│   ├── wikitext.py
│   └── imagenet.py (optional)
└── scripts/
    ├── train.py (main entry point)
    └── plot_results.py
```

**Add**:
- `requirements.txt` with exact versions
- `INSTALL.md` with setup instructions
- `EXPERIMENTS.md` with reproduction instructions

---

## Timeline Estimate

**Week 1**: Benchmarking + Baselines
- Day 1-2: GPU benchmarks, tune sparse_threshold
- Day 3-5: Integrate RigL and MEST
- Day 6-7: Test baselines work

**Week 2**: CIFAR-10 Experiments
- Run RESU, RigL, MEST, Dense with 5 seeds
- Collect all metrics
- Create plots and tables

**Week 3**: WikiText Experiments
- Same as CIFAR-10 but for language modeling

**Week 4** (Optional): Large-scale
- ImageNet or LLM experiments if you have compute

**Week 5-6**: Paper Writing
- Write Method, Algorithm, Experiments sections
- Create all figures
- Polish and proofread

**Week 7**: Internal Review
- Have collaborators read
- Address feedback

**Week 8**: Final Submission
- Format for NeurIPS
- Submit!

---

## Tips for Success

### Experiments
- **Use wandb or tensorboard** for tracking
- **Save checkpoints** at each cycle
- **Log resurrection stats** (very important for paper)
- **Run multiple seeds** (3-5 minimum)
- **Use consistent hyperparameters** across baselines

### Paper Writing
- **Lead with the insight**: "Pruned weights can be resurrected via gradient-based competition"
- **Emphasize novelty**: First method to enable dead weights to receive updates
- **Show resurrection helps**: Ablation without amnesty should perform worse
- **Be honest about limitations**: Sparse ops overhead on GPU, works best at high sparsity

### Common Pitfalls
- ❌ Don't claim sparse is always faster (it's not on GPU at low sparsity)
- ❌ Don't forget to zero pruned weights after amnesty (we just fixed this!)
- ❌ Don't use different training budgets for baselines (unfair comparison)
- ✅ Do show resurrection rate improves over random regrowth (RigL)
- ✅ Do show amnesty enables weaker resurrections to compete

---

## Quick Start

```bash
# 1. Verify everything works
pytest

# 2. Benchmark on your GPU
python -m benchmarks.bench_throughput
python -m benchmarks.bench_memory

# 3. Run a quick CIFAR-10 experiment
python experiments/cifar10.py --method resu --sparsity 0.7 --cycles 5

# 4. Compare with baseline
python experiments/cifar10.py --method rigl --sparsity 0.7

# 5. Plot results
python scripts/plot_results.py
```

---

Good luck! You have a solid foundation. The implementation is correct, tested, and ready for experiments.
