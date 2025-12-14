# RESU: Resurrection of Sparse Units
## Full Implementation Architecture

### Vision
Not a minimal prototype. A production-ready, SOTA sparse training system with:
- Fused Triton kernels for all hot paths
- Clean PyTorch abstractions
- Drop-in module replacements
- Full training infrastructure

---

## Project Structure

```
resu/
├── kernels/                    # Triton kernels (bottom layer)
│   ├── __init__.py
│   ├── masked_ops.py          # Masked multiply, add, fused ops
│   ├── embedding.py           # Φ and Φ⁻¹ scatter/gather
│   ├── update.py              # RESU gradient update
│   ├── consistency.py         # EMA tracking, consistency compute
│   ├── selection.py           # TopK, filtering, combined selection
│   └── pruning.py             # Importance scoring, tournament
│
├── core/                       # Core abstractions (middle layer)
│   ├── __init__.py
│   ├── mask.py                # SparseMask class - the partition (A, P)
│   ├── resurrection.py        # ResurrectionEmbedding - Φ, Φ⁻¹
│   ├── effective.py           # EffectiveWeight computation
│   ├── updater.py             # RESUUpdater - basic update rule
│   └── selective.py           # RESUSelective - with consistency
│
├── modules/                    # Drop-in replacements (user-facing)
│   ├── __init__.py
│   ├── linear.py              # RESULinear
│   ├── conv.py                # RESUConv2d
│   ├── attention.py           # RESUMultiheadAttention
│   └── wrapper.py             # wrap_module() utility
│
├── pruning/                    # Pruning algorithms
│   ├── __init__.py
│   ├── magnitude.py           # Basic magnitude pruning
│   ├── wanda.py               # Wanda: W * activation magnitude
│   ├── dsnot.py               # DSNOT stabilization
│   └── amnesty.py             # Amnesty with relative tournament
│
├── training/                   # Training infrastructure
│   ├── __init__.py
│   ├── config.py              # RESUConfig dataclass
│   ├── scheduler.py           # Sparsity schedule, r(c) schedule
│   ├── cycle.py               # Single Train→Prune→Stabilize→RESU→Commit
│   ├── trainer.py             # Full training loop
│   └── callbacks.py           # Logging, checkpointing
│
├── optim/                      # Custom optimizers
│   ├── __init__.py
│   └── resu_adamw.py          # AdamW variant aware of RESU phases
│
└── utils/
    ├── __init__.py
    ├── sparse_utils.py        # Sparsity computation, mask ops
    ├── metrics.py             # Training metrics
    └── visualization.py       # Sparsity pattern viz
```

---

## Implementation Phases

### Phase 1: Triton Kernels - Core Operations
**Goal**: Efficient primitives for masked operations and embedding

#### 1.1 `kernels/masked_ops.py`
```python
# Kernels to implement:

@triton.jit
def masked_mul_kernel(X, M, Out, ...)
    """Out = M ⊙ X"""

@triton.jit  
def inv_masked_mul_kernel(X, M, Out, ...)
    """Out = (1-M) ⊙ X"""

@triton.jit
def fused_effective_weight_kernel(W, theta_embedded, M, Out, ...)
    """Out = M⊙W + (1-M)⊙theta_embedded
    Single pass, no intermediate allocations"""

@triton.jit
def masked_add_kernel(X, Y, M, Out, ...)
    """Out = X + M⊙Y (for gradient accumulation)"""
```

#### 1.2 `kernels/embedding.py`
```python
# The heart of RESU

@triton.jit
def phi_embed_kernel(theta, indices, Out, ...)
    """Φ(θ): Scatter θ values to pruned positions
    
    indices: flattened positions in P (precomputed from mask)
    theta: p-dimensional vector
    Out: dout × din matrix (sparse, only P positions filled)
    """

@triton.jit
def phi_inverse_kernel(G_P, indices, out_theta, ...)
    """Φ⁻¹(G_P): Gather gradients from pruned positions
    
    G_P: gradient matrix (only P positions matter)
    indices: flattened positions in P
    out_theta: p-dimensional gradient vector
    """

@triton.jit
def fused_effective_and_embed_kernel(W, theta, mask, indices, Out, ...)
    """Combined: compute Φ(θ) and W_eff in one kernel
    Most common operation during RESU forward pass"""
```

#### 1.3 `kernels/update.py`
```python
@triton.jit
def resu_update_kernel(theta, grad_theta, lr, ...)
    """Basic: θ ← θ - η·∇θ"""

@triton.jit
def resu_selective_update_kernel(
    theta, grad_theta, 
    selection_mask, consistency_weights,
    lr, ...
)
    """Selective: θ ← θ - η·M_sel⊙C_t⊙∇θ"""

@triton.jit
def resu_adamw_update_kernel(
    theta, grad_theta,
    m, v,  # momentum, variance
    selection_mask, consistency_weights,
    lr, beta1, beta2, weight_decay,
    step, ...
)
    """Full AdamW update for θ with selective masking"""
```

### Phase 2: Triton Kernels - RESU-Selective

#### 2.1 `kernels/consistency.py`
```python
@triton.jit
def ema_update_kernel(m, v, g, beta, ...)
    """Update EMAs:
    m = β·m + (1-β)·g
    v = β·v + (1-β)·|g|
    """

@triton.jit
def consistency_compute_kernel(m, v, C, delta, ...)
    """C = |m| / (v + δ)"""

@triton.jit
def fused_ema_consistency_kernel(m, v, g, C, beta, delta, ...)
    """All-in-one: update EMAs and compute consistency"""
```

#### 2.2 `kernels/selection.py`
```python
@triton.jit
def topk_indices_kernel(values, k, out_indices, out_values, ...)
    """GPU TopK - return indices and values
    Uses parallel reduction + heap"""

@triton.jit
def threshold_filter_kernel(C, tau, out_mask, ...)
    """P_con = {(i,j) : C(i,j) > τ}"""

@triton.jit
def intersection_topk_kernel(
    values, mask1, mask2, 
    k, out_indices, ...
)
    """TopK over intersection of two masks
    For P_select = TopK(P_mag ∩ P_con)"""
```

### Phase 3: Triton Kernels - Pruning

#### 3.1 `kernels/pruning.py`
```python
@triton.jit
def wanda_score_kernel(W, activation_norms, scores, ...)
    """Wanda importance: |W| · ||X||"""

@triton.jit
def magnitude_score_kernel(W, scores, ...)
    """Simple |W| scoring"""

@triton.jit
def relative_tournament_kernel(
    scores, current_mask,
    n_active, n_resurrection,
    new_mask, ...
)
    """Amnesty: TopK among active, TopK among resurrected
    No direct competition between groups"""

@triton.jit
def dsnot_update_kernel(W, M, grad, threshold, ...)
    """DSNOT mask update step"""
```

---

## Phase 4: Core Abstractions

### 4.1 `core/mask.py`
```python
class SparseMask:
    """Represents the partition (A, P) induced by pruning.
    
    Precomputes and caches:
    - mask: the binary mask M
    - active_indices: flattened indices of A
    - pruned_indices: flattened indices of P (needed for Φ, Φ⁻¹)
    - sparsity: current sparsity level
    """
    
    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        self._precompute_indices()
    
    def _precompute_indices(self):
        flat = self.mask.flatten()
        self.active_indices = torch.nonzero(flat, as_tuple=True)[0]
        self.pruned_indices = torch.nonzero(1 - flat, as_tuple=True)[0]
        self.n_active = len(self.active_indices)
        self.n_pruned = len(self.pruned_indices)
    
    @property
    def sparsity(self) -> float:
        return self.n_pruned / self.mask.numel()
    
    def update(self, new_mask: torch.Tensor):
        """Update mask and recompute indices"""
        ...
```

### 4.2 `core/resurrection.py`
```python
class ResurrectionEmbedding:
    """Implements Φ: ℝᵖ → S_P and Φ⁻¹: S_P → ℝᵖ
    
    The core abstraction of RESU.
    """
    
    def __init__(self, sparse_mask: SparseMask, device: torch.device):
        self.mask = sparse_mask
        self.device = device
        
        # θ ∈ ℝᵖ - the learnable resurrection parameters
        self.theta = None  # Initialized during RESU phase
    
    def initialize(self, active_std: float, epsilon: float = 0.1):
        """Initialize θ ~ N(0, ε·σ_A)"""
        p = self.mask.n_pruned
        self.theta = torch.randn(p, device=self.device) * (epsilon * active_std)
        self.theta.requires_grad_(True)
    
    def phi(self) -> torch.Tensor:
        """Φ(θ): Embed θ into pruned subspace S_P
        Returns: sparse matrix with θ values at pruned positions
        """
        # Uses phi_embed_kernel
        ...
    
    def phi_inverse(self, G_P: torch.Tensor) -> torch.Tensor:
        """Φ⁻¹(G_P): Extract gradient vector from pruned subspace
        Returns: p-dimensional gradient for θ
        """
        # Uses phi_inverse_kernel
        ...
    
    def commit(self) -> torch.Tensor:
        """Return final Φ(θ) for weight commitment"""
        return self.phi()
```

### 4.3 `core/effective.py`
```python
class EffectiveWeight(torch.autograd.Function):
    """Custom autograd for W_eff = M⊙W + (1-M)⊙Φ(θ)
    
    Forward: Compute effective weights
    Backward: Split gradients to W and θ
    """
    
    @staticmethod
    def forward(ctx, W, theta, mask, pruned_indices):
        # Uses fused_effective_weight_kernel
        ctx.save_for_backward(mask, pruned_indices)
        ...
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, pruned_indices = ctx.saved_tensors
        
        # G_A = M ⊙ G (for active weights, if training them)
        # G_P = (1-M) ⊙ G → Φ⁻¹ → grad_theta
        ...
```

### 4.4 `core/updater.py`
```python
class RESUUpdater:
    """Basic RESU update: θ ← θ - η·Φ⁻¹(G_P)"""
    
    def __init__(
        self,
        resurrection: ResurrectionEmbedding,
        lr: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.resurrection = resurrection
        self.optimizer = torch.optim.AdamW(
            [resurrection.theta], 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def step(self, grad_full: torch.Tensor):
        """Perform one RESU update step"""
        grad_theta = self.resurrection.phi_inverse(grad_full)
        self.resurrection.theta.grad = grad_theta
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 4.5 `core/selective.py`
```python
class RESUSelective:
    """RESU-Selective with directional consistency filtering.
    
    Tracks:
    - m_t: EMA of gradients (direction)
    - v_t: EMA of |gradients| (magnitude)
    - C_t: consistency = |m_t| / (v_t + δ)
    
    Filters:
    - P_mag: TopK by gradient magnitude
    - P_con: Above consistency threshold
    - P_select: TopK of intersection
    """
    
    def __init__(
        self,
        resurrection: ResurrectionEmbedding,
        beta: float = 0.9,
        tau_stable: float = 0.5,
        k_screen: int = None,  # default: 50% of |P|
        k_select: int = None,  # default: 20% of |P|
        lr: float = 1e-4,
    ):
        self.resurrection = resurrection
        self.beta = beta
        self.tau_stable = tau_stable
        
        p = resurrection.mask.n_pruned
        self.k_screen = k_screen or p // 2
        self.k_select = k_select or p // 5
        
        # EMA state
        self.m = torch.zeros(p, device=resurrection.device)
        self.v = torch.zeros(p, device=resurrection.device)
        self.step_count = 0
    
    def compute_consistency(self) -> torch.Tensor:
        """C_t = |m_t| / (v_t + δ)"""
        ...
    
    def select_coordinates(self, grad_theta: torch.Tensor) -> torch.Tensor:
        """Returns selection mask for update"""
        # 1. P_mag = TopK by |grad|
        # 2. P_con = threshold by C_t
        # 3. P_select = TopK of intersection
        ...
    
    def step(self, grad_full: torch.Tensor):
        """Selective RESU update"""
        grad_theta = self.resurrection.phi_inverse(grad_full)
        
        # Update EMAs
        self.m = self.beta * self.m + (1 - self.beta) * grad_theta
        self.v = self.beta * self.v + (1 - self.beta) * grad_theta.abs()
        
        # Get selection and consistency
        C = self.compute_consistency()
        selection = self.select_coordinates(grad_theta)
        
        # Weighted update: θ -= η · selection · C · grad
        ...
```

---

## Phase 5: Pruning Infrastructure

### 5.1 `pruning/wanda.py`
```python
class WandaPruner:
    """Wanda: Pruning by Weights and Activations
    
    Score = |W| · ||X||₂ (activation L2 norm)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_norms = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activation norms"""
        ...
    
    def compute_scores(self, layer_name: str) -> torch.Tensor:
        """Compute Wanda importance scores"""
        ...
    
    def prune(self, sparsity: float) -> Dict[str, SparseMask]:
        """Prune model to target sparsity, return masks"""
        ...
```

### 5.2 `pruning/dsnot.py`
```python
class DSNOTStabilizer:
    """DSNOT: Dynamic Sparse Network Optimization with Threshold
    
    Iteratively adjusts mask based on gradient information
    to find stable sparse structure.
    """
    
    def __init__(
        self,
        model: nn.Module,
        masks: Dict[str, SparseMask],
        threshold: float = 0.01
    ):
        ...
    
    def step(self, loss: torch.Tensor):
        """One DSNOT stabilization step"""
        ...
    
    def stabilize(self, dataloader, num_steps: int):
        """Run full stabilization phase"""
        ...
```

### 5.3 `pruning/amnesty.py`
```python
class AmnestyMechanism:
    """Relative Tournament Pruning with Resurrection Budget
    
    r(c) = r_start - (r_start - r_end) · (c/C)
    
    Separate competitions:
    - Best active weights compete among themselves
    - Best resurrected weights compete among themselves
    """
    
    def __init__(
        self,
        r_start: float = 0.10,
        r_end: float = 0.02,
        total_cycles: int = 5
    ):
        self.r_start = r_start
        self.r_end = r_end
        self.total_cycles = total_cycles
    
    def resurrection_budget(self, cycle: int) -> float:
        """r(c) - fraction of budget reserved for resurrections"""
        return self.r_start - (self.r_start - self.r_end) * (cycle / self.total_cycles)
    
    def relative_tournament(
        self,
        scores: torch.Tensor,
        current_mask: SparseMask,
        sparsity: float,
        cycle: int
    ) -> SparseMask:
        """Perform relative tournament pruning"""
        r = self.resurrection_budget(cycle)
        n_keep = int((1 - sparsity) * scores.numel())
        n_resurrection = int(r * n_keep)
        n_active = n_keep - n_resurrection
        
        # TopK among currently active
        # TopK among currently pruned (resurrected)
        # Union → new mask
        ...
```

---

## Phase 6: Modules and Training

### 6.1 `modules/linear.py`
```python
class RESULinear(nn.Module):
    """Drop-in replacement for nn.Linear with RESU support.
    
    Modes:
    - DENSE: Normal forward pass
    - SPARSE: Forward with mask (standard sparse training)
    - RESU: Forward with effective weights (resurrection phase)
    """
    
    class Mode(Enum):
        DENSE = auto()
        SPARSE = auto()
        RESU = auto()
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        self.mask: Optional[SparseMask] = None
        self.resurrection: Optional[ResurrectionEmbedding] = None
        self.mode = RESULinear.Mode.DENSE
        
        self.reset_parameters()
    
    def set_mask(self, mask: torch.Tensor):
        """Set pruning mask and initialize SparseMask"""
        self.mask = SparseMask(mask)
    
    def enter_resu_mode(self, epsilon: float = 0.1):
        """Initialize resurrection parameters and enter RESU mode"""
        assert self.mask is not None
        active_std = self.weight[self.mask.mask.bool()].std().item()
        self.resurrection = ResurrectionEmbedding(self.mask, self.weight.device)
        self.resurrection.initialize(active_std, epsilon)
        self.mode = RESULinear.Mode.RESU
    
    def exit_resu_mode(self, commit: bool = True):
        """Exit RESU mode, optionally committing resurrected weights"""
        if commit and self.resurrection is not None:
            # W = M⊙W + (1-M)⊙Φ(θ)
            with torch.no_grad():
                self.weight.data = (
                    self.mask.mask * self.weight + 
                    self.resurrection.commit()
                )
        self.resurrection = None
        self.mode = RESULinear.Mode.SPARSE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == RESULinear.Mode.DENSE:
            return F.linear(x, self.weight, self.bias)
        
        elif self.mode == RESULinear.Mode.SPARSE:
            W_sparse = self.mask.mask * self.weight
            return F.linear(x, W_sparse, self.bias)
        
        elif self.mode == RESULinear.Mode.RESU:
            W_eff = EffectiveWeight.apply(
                self.weight,
                self.resurrection.theta,
                self.mask.mask,
                self.mask.pruned_indices
            )
            return F.linear(x, W_eff, self.bias)
```

### 6.2 `training/config.py`
```python
@dataclass
class RESUConfig:
    """Full configuration for RESU training"""
    
    # Sparsity
    initial_sparsity: float = 0.0
    target_sparsity: float = 0.7
    sparsity_schedule: str = "linear"  # linear, exponential, cosine
    
    # Cycles
    num_cycles: int = 5
    steps_per_cycle: int = 1000
    
    # Phase lengths (as fraction of steps_per_cycle)
    train_fraction: float = 0.6
    dsnot_fraction: float = 0.1
    resu_fraction: float = 0.3
    
    # RESU parameters
    resu_lr: float = 1e-4
    resu_epsilon: float = 0.1  # initialization scale
    use_selective: bool = True
    selective_beta: float = 0.9
    selective_tau: float = 0.5
    selective_k_screen_ratio: float = 0.5
    selective_k_select_ratio: float = 0.2
    
    # Amnesty
    amnesty_r_start: float = 0.10
    amnesty_r_end: float = 0.02
    
    # Pruning
    pruning_method: str = "wanda"  # magnitude, wanda
    
    # Densification mode
    densify: bool = False  # If True, decrease sparsity each cycle
    
    # Training
    base_lr: float = 1e-3
    weight_decay: float = 0.01
```

### 6.3 `training/cycle.py`
```python
class RESUCycle:
    """One complete RESU training cycle:
    Train → Prune → Stabilize → RESU → Commit
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RESUConfig,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        cycle_num: int,
    ):
        ...
    
    def train_phase(self):
        """Standard training with current mask"""
        for step in range(self.train_steps):
            batch = next(self.data_iter)
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def prune_phase(self):
        """Prune to target sparsity"""
        if self.config.pruning_method == "wanda":
            self.pruner.prune(self.current_sparsity)
        ...
    
    def stabilize_phase(self):
        """DSNOT stabilization"""
        self.stabilizer.stabilize(self.dataloader, self.dsnot_steps)
    
    def resu_phase(self):
        """Resurrect pruned weights"""
        # Enter RESU mode for all layers
        for module in self.resu_modules:
            module.enter_resu_mode(self.config.resu_epsilon)
        
        # Create updaters
        if self.config.use_selective:
            updaters = {
                name: RESUSelective(module.resurrection, ...)
                for name, module in self.resu_modules.items()
            }
        else:
            updaters = {
                name: RESUUpdater(module.resurrection, ...)
                for name, module in self.resu_modules.items()
            }
        
        # RESU training loop
        for step in range(self.resu_steps):
            batch = next(self.data_iter)
            loss = self.compute_loss(batch)
            loss.backward()
            
            for name, module in self.resu_modules.items():
                # Get full gradient (includes both active and pruned)
                grad_full = module.weight.grad
                updaters[name].step(grad_full)
            
            # Note: we don't step the main optimizer during RESU
    
    def commit_phase(self):
        """Commit with amnesty"""
        for name, module in self.resu_modules.items():
            # Get scores for all weights
            W_eff = module.mask.mask * module.weight + module.resurrection.commit()
            scores = self.compute_importance(W_eff)
            
            # Relative tournament
            new_mask = self.amnesty.relative_tournament(
                scores, module.mask, self.current_sparsity, self.cycle_num
            )
            
            # Commit and update mask
            module.exit_resu_mode(commit=True)
            module.set_mask(new_mask.mask)
    
    def run(self):
        """Execute full cycle"""
        self.train_phase()
        self.prune_phase()
        self.stabilize_phase()
        self.resu_phase()
        self.commit_phase()
```

---

## Kernel Fusion Strategy

### Hot Paths to Fuse:

1. **RESU Forward** (every forward pass during RESU):
   ```
   W_eff = M⊙W + (1-M)⊙Φ(θ)
   ```
   → Single kernel: `fused_effective_weight_kernel`

2. **RESU Backward + Update** (every backward during RESU):
   ```
   G_P = (1-M)⊙G
   grad_θ = Φ⁻¹(G_P)
   Update EMAs: m, v
   Compute C
   Apply selective update
   ```
   → Fuse: `fused_resu_backward_update_kernel`

3. **Pruning Score + Mask** (once per cycle):
   ```
   scores = |W| · ||X||
   mask = TopK(scores)
   ```
   → Fuse: `fused_wanda_prune_kernel`

4. **Amnesty Tournament** (once per cycle):
   ```
   Split scores by current mask
   TopK each group
   Merge into new mask
   ```
   → Fuse: `fused_amnesty_kernel`

---

## Memory Layout Considerations

### θ Storage Options:

1. **Separate Vector** (current design):
   - θ ∈ ℝᵖ stored as contiguous 1D tensor
   - Pro: Natural for optimizer states
   - Con: Extra scatter/gather

2. **In-Place in W** (paper's claim):
   - Store θ directly in pruned positions of W
   - Pro: Zero memory overhead
   - Con: Tricky gradient handling, can't have W.requires_grad during RESU

3. **Hybrid**:
   - θ stored separately during RESU phase
   - Committed in-place after RESU
   - Best of both worlds

**Decision**: Start with (1) for correctness, optimize to (3) later.

### Index Precomputation:

- `pruned_indices`: Precomputed once per mask update
- Stored as `torch.int32` to save memory
- Consider CSR format for very sparse masks

---

## Testing Strategy

### Unit Tests:
- Each kernel against PyTorch reference
- Gradient checks for custom autograd
- Mask operations correctness

### Integration Tests:
- Full RESU cycle on small model
- Sparsity tracking through training
- Resurrection actually happens (weights change)

### Benchmarks:
- Kernel latency vs PyTorch ops
- Full training throughput
- Memory usage vs dense training

---

## Implementation Order

### Day 1: Core Kernels
1. `kernels/masked_ops.py` - all masked operations
2. `kernels/embedding.py` - Φ and Φ⁻¹
3. `core/mask.py` - SparseMask class
4. `core/resurrection.py` - ResurrectionEmbedding

### Day 2: RESU Update
5. `kernels/update.py` - update kernels
6. `core/updater.py` - RESUUpdater
7. `core/effective.py` - EffectiveWeight autograd
8. Test: Basic RESU step works

### Day 3: RESU-Selective
9. `kernels/consistency.py` - EMA kernels
10. `kernels/selection.py` - TopK, filtering
11. `core/selective.py` - RESUSelective
12. Test: Selective filtering works

### Day 4: Modules
13. `modules/linear.py` - RESULinear
14. `modules/conv.py` - RESUConv2d
15. Test: Drop-in replacement works

### Day 5: Pruning
16. `pruning/wanda.py`
17. `pruning/dsnot.py`
18. `pruning/amnesty.py`
19. Test: Full prune→stabilize→resurrect→commit

### Day 6: Training Loop
20. `training/config.py`
21. `training/scheduler.py`
22. `training/cycle.py`
23. `training/trainer.py`
24. Test: Full training on MNIST

### Day 7: Optimization & Polish
25. Kernel fusion
26. Benchmarking
27. Documentation
28. Example scripts

---

## Let's Begin

Starting with `kernels/masked_ops.py` - the foundation.
