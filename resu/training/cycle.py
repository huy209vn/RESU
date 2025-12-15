"""
RESU Training Cycle

Orchestrates the complete RESU training loop:
Train → Prune → Stabilize → RESU → Commit (with Amnesty)

Each cycle takes a model through all phases, progressively
refining the sparse structure while allowing resurrection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import time

from .config import RESUConfig
from ..core.mask import SparseMask
from ..core.selective import SelectionConfig
from ..modules.linear import RESULinear, get_resu_modules, find_layers_qwen2
from ..pruning.amnesty import Amnesty, AmnestyConfig, AmnestyResult

# Optional: Wanda++/DSNoT integration (if prune.py available)
try:
    from ..pruning.integration import (
        WandaPlusPruner,
        DSNoTStabilizer,
        WandaDSNoTConfig,
        create_pruner_and_stabilizer,
    )
    HAS_WANDA_DSNOT = True
except ImportError:
    HAS_WANDA_DSNOT = False


class Phase(Enum):
    """Training phases within a cycle."""
    TRAIN = auto()
    PRUNE = auto()
    STABILIZE = auto()
    RESU = auto()
    COMMIT = auto()


@dataclass
class CycleStats:
    """Statistics from a training cycle."""
    cycle: int
    target_sparsity: float
    actual_sparsity: float
    
    # Per-phase metrics
    train_loss: float
    train_steps: int
    stabilize_steps: int
    resu_steps: int
    
    # RESU metrics
    resu_updates: int
    mean_consistency: float
    mean_selection_ratio: float
    
    # Amnesty metrics
    resurrection_budget: float
    n_resurrected: int
    n_active_kept: int
    resurrection_rate: float
    
    # Timing
    duration_seconds: float


@dataclass
class PhaseResult:
    """Result from a single phase."""
    phase: Phase
    steps: int
    metrics: Dict[str, float]
    duration: float


class RESUCycle:
    """Executes one complete RESU training cycle.
    
    Phases:
    1. TRAIN: Standard training with current mask
    2. PRUNE: Prune to target sparsity (if needed)
    3. STABILIZE: DSNOT stabilization (optional)
    4. RESU: Train resurrection parameters
    5. COMMIT: Merge θ with amnesty tournament
    
    Example:
        cycle = RESUCycle(
            model=model,
            config=config,
            optimizer=optimizer,
            train_fn=train_step,
            cycle_num=0,
        )
        
        stats = cycle.run(train_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RESUConfig,
        optimizer: torch.optim.Optimizer,
        train_fn: Callable[[nn.Module, Any], torch.Tensor],
        cycle_num: int,
        pruner: Optional[Any] = None,  # Your Wanda/DSNOT pruner
        stabilizer: Optional[Any] = None,  # Your DSNOT stabilizer
        device: Optional[torch.device] = None,
        logger: Optional[Callable] = None,
    ):
        """
        Args:
            model: Model with RESULinear layers
            config: RESU configuration
            optimizer: Main optimizer
            train_fn: Function(model, batch) -> loss
            cycle_num: Current cycle number (0-indexed)
            pruner: Pruning implementation (Wanda)
            stabilizer: DSNOT stabilizer
            device: Target device
            logger: Optional logging function
        """
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_fn = train_fn
        self.cycle_num = cycle_num
        self.pruner = pruner
        self.stabilizer = stabilizer
        self.device = device or next(model.parameters()).device
        self.logger = logger or print
        
        # Get RESU modules
        self.resu_modules = get_resu_modules(model)
        if not self.resu_modules:
            raise ValueError("No RESULinear modules found in model")
        
        # Amnesty
        self.amnesty = Amnesty(AmnestyConfig(
            r_start=config.amnesty_r_start,
            r_end=config.amnesty_r_end,
            total_cycles=config.num_cycles,
            score_type=config.amnesty_score_type,
        ))
        
        # Target sparsity for this cycle
        self.target_sparsity = config.get_sparsity_for_cycle(cycle_num)
        
        # Metrics accumulation
        self._metrics: Dict[str, List[float]] = {}
    
    def _log(self, msg: str):
        """Log a message."""
        self.logger(f"[Cycle {self.cycle_num}] {msg}")
    
    def _accumulate(self, key: str, value: float):
        """Accumulate a metric."""
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)
    
    def _mean_metric(self, key: str) -> float:
        """Get mean of accumulated metric."""
        if key not in self._metrics or not self._metrics[key]:
            return 0.0
        return sum(self._metrics[key]) / len(self._metrics[key])
    
    # =========================================================================
    # Phase Implementations
    # =========================================================================
    
    def train_phase(self, dataloader: DataLoader) -> PhaseResult:
        """Standard training with current sparse mask."""
        self._log(f"TRAIN phase: {self.config.train_steps} steps")
        start_time = time.time()
        
        self.model.train()
        data_iter = iter(dataloader)
        
        total_loss = 0.0
        for step in range(self.config.train_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            if isinstance(batch, (tuple, list)):
                batch = tuple(b.to(self.device) if hasattr(b, 'to') else b for b in batch)
            elif hasattr(batch, 'to'):
                batch = batch.to(self.device)
            
            # Forward + backward
            self.optimizer.zero_grad()
            loss = self.train_fn(self.model, batch)
            loss.backward()
            self.optimizer.step()
            
            # Re-apply masks after optimizer step
            for module in self.resu_modules.values():
                if module.is_sparse and not module.is_resu_active:
                    module.apply_mask()
            
            total_loss += loss.item()
            
            if (step + 1) % self.config.log_interval == 0:
                self._log(f"  Step {step+1}/{self.config.train_steps}, Loss: {loss.item():.4f}")
        
        duration = time.time() - start_time
        avg_loss = total_loss / self.config.train_steps
        
        return PhaseResult(
            phase=Phase.TRAIN,
            steps=self.config.train_steps,
            metrics={"loss": avg_loss},
            duration=duration,
        )
    
    def prune_phase(self, dataloader: Optional[DataLoader] = None) -> PhaseResult:
        """Prune to target sparsity using Wanda++ or magnitude."""
        self._log(f"PRUNE phase: target sparsity = {self.target_sparsity:.1%}")
        start_time = time.time()
        
        # Check if pruning needed
        current_sparsity = self._get_model_sparsity()
        if abs(current_sparsity - self.target_sparsity) < 0.01:
            self._log("  Sparsity already at target, skipping")
            return PhaseResult(Phase.PRUNE, 0, {"sparsity": current_sparsity}, 0)
        
        if self.pruner is not None:
            # Use provided pruner (WandaPlusPruner or similar)
            if hasattr(self.pruner, 'calibrate'):
                self.pruner.calibrate(self.target_sparsity)
            if hasattr(self.pruner, 'prune'):
                self.pruner.prune(self.target_sparsity)
            elif hasattr(self.pruner, 'get_masks'):
                masks = self.pruner.get_masks(self.target_sparsity)
                self.pruner.apply_masks(masks)
        else:
            # Fallback: magnitude pruning on RESULinear layers
            for name, module in self.resu_modules.items():
                module.prune_by_magnitude(self.target_sparsity)
        
        actual_sparsity = self._get_model_sparsity()
        duration = time.time() - start_time
        
        self._log(f"  Pruned to {actual_sparsity:.1%} sparsity")
        
        return PhaseResult(
            phase=Phase.PRUNE,
            steps=0,
            metrics={"sparsity": actual_sparsity},
            duration=duration,
        )
    
    def stabilize_phase(self, dataloader: DataLoader) -> PhaseResult:
        """DSNoT stabilization phase."""
        if self.config.stabilize_steps == 0 and self.stabilizer is None:
            return PhaseResult(Phase.STABILIZE, 0, {}, 0)
        
        self._log(f"STABILIZE phase: DSNoT refinement")
        start_time = time.time()
        
        if self.stabilizer is not None:
            # Use DSNoTStabilizer
            if hasattr(self.stabilizer, 'stabilize'):
                masks = self.stabilizer.stabilize(self.target_sparsity)
                # Masks are already applied by run_dsnot
                self._log(f"  DSNoT refined {len(masks)} layers")
        else:
            # Minimal stabilization: just train with mask
            self.model.train()
            data_iter = iter(dataloader)
            
            for step in range(self.config.stabilize_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                if isinstance(batch, (tuple, list)):
                    batch = tuple(b.to(self.device) if hasattr(b, 'to') else b for b in batch)
                elif hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.train_fn(self.model, batch)
                loss.backward()
                self.optimizer.step()
                
                for module in self.resu_modules.values():
                    module.apply_mask()
        
        duration = time.time() - start_time
        return PhaseResult(Phase.STABILIZE, self.config.stabilize_steps, {}, duration)
    
    def resu_phase(self, dataloader: DataLoader) -> PhaseResult:
        """RESU resurrection phase."""
        self._log(f"RESU phase: {self.config.resu_steps} steps")
        start_time = time.time()
        
        # Enter RESU mode for all layers
        selective_config = SelectionConfig(
            beta=self.config.selective_beta,
            delta=self.config.selective_delta,
            tau_stable=self.config.selective_tau,
            k_screen_ratio=self.config.selective_k_screen_ratio,
            k_select_ratio=self.config.selective_k_select_ratio,
        ) if self.config.use_selective else None
        
        # CRITICAL FIX: Clear optimizer states for W before entering RESU
        # This frees memory from momentum/variance that was tracking W
        # During RESU, only θ needs optimizer states (managed internally)
        for name, module in self.resu_modules.items():
            if module.weight in self.optimizer.state:
                del self.optimizer.state[module.weight]
                self._log(f"  Cleared optimizer state for {name} (freed memory)")

        for name, module in self.resu_modules.items():
            if module.mask is not None:
                module.enter_resu_mode(
                    epsilon=self.config.resu_epsilon,
                    use_selective=self.config.use_selective,
                    selective_config=selective_config,
                    lr=self.config.resu_lr,
                    weight_decay=self.config.weight_decay,
                )
        
        # RESU training loop
        self.model.train()
        data_iter = iter(dataloader)
        
        total_updates = 0
        consistency_sum = 0.0
        selection_sum = 0.0
        
        for step in range(self.config.resu_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            if isinstance(batch, (tuple, list)):
                batch = tuple(b.to(self.device) if hasattr(b, 'to') else b for b in batch)
            elif hasattr(batch, 'to'):
                batch = batch.to(self.device)
            
            # Forward (with effective weights)
            self.optimizer.zero_grad()
            loss = self.train_fn(self.model, batch)
            loss.backward()
            
            # RESU update (update θ, not W)
            for name, module in self.resu_modules.items():
                if module.is_resu_active and module.weight.grad is not None:
                    stats = module.resu_step(module.weight.grad)
                    if stats:
                        total_updates += 1
                        consistency_sum += stats.get("mean_consistency", 0)
                        selection_sum += stats.get("selection_ratio", 0)
            
            # Don't step main optimizer during RESU
            # self.optimizer.step()  # Commented out intentionally
            
            if (step + 1) % self.config.log_interval == 0:
                self._log(f"  RESU step {step+1}/{self.config.resu_steps}, Loss: {loss.item():.4f}")
        
        duration = time.time() - start_time
        
        metrics = {
            "updates": total_updates,
            "mean_consistency": consistency_sum / max(total_updates, 1),
            "mean_selection_ratio": selection_sum / max(total_updates, 1),
        }
        
        return PhaseResult(Phase.RESU, self.config.resu_steps, metrics, duration)
    
    def commit_phase(self) -> PhaseResult:
        """Commit RESU and update mask based on strategy."""
        self._log(f"COMMIT phase (strategy: {self.config.commit_strategy})")
        start_time = time.time()

        total_resurrected = 0
        total_active_kept = 0

        for name, module in self.resu_modules.items():
            if not module.is_resu_active:
                continue

            # Get effective weights and old mask
            W_eff = module.get_effective_weight()
            old_mask = module.mask

            # Choose commit strategy
            if self.config.commit_strategy == "amnesty":
                # Original amnesty tournament
                if self.config.use_amnesty and old_mask is not None:
                    new_mask, result = self.amnesty.commit_with_amnesty(
                        W_eff=W_eff,
                        old_mask=old_mask,
                        target_sparsity=self.target_sparsity,
                        cycle=self.cycle_num,
                    )

                    total_resurrected += result.n_resurrected_kept
                    total_active_kept += result.n_active_kept

                    self._log(f"  {name}: {result.n_resurrected_kept} resurrected, "
                             f"{result.n_active_kept} active kept")

                    # Exit RESU and update mask
                    module.exit_resu_mode(commit=True)
                    module.set_mask(new_mask)

                    # Zero out newly pruned positions
                    with torch.no_grad():
                        module.weight.data *= new_mask.mask
                else:
                    # Simple commit without amnesty
                    module.exit_resu_mode(commit=True)

            elif self.config.commit_strategy == "wanda_reprune":
                # RESU+Wanda strategy: Merge all θ → W, then re-prune with Wanda++
                self._log(f"  {name}: Merging all resurrections, will re-prune with Wanda++")

                # 1. Count pruned positions before merge
                old_pruned = 0
                if old_mask is not None:
                    old_pruned = int((~old_mask.mask.bool()).sum().item())

                # 2. Merge ALL θ into W (no tournament)
                module.exit_resu_mode(commit=True)

                # 3. Re-prune with Wanda++ (structure-aware)
                # This will be done in the next cycle's prune phase
                # After merge, W_eff has values at all positions
                total_resurrected += old_pruned  # All pruned positions get values

                self._log(f"  {name}: All {old_pruned} pruned positions merged")

            elif self.config.commit_strategy == "simple":
                # Just merge θ → W without re-pruning
                self._log(f"  {name}: Simple merge without re-pruning")
                module.exit_resu_mode(commit=True)

            else:
                raise ValueError(f"Unknown commit_strategy: {self.config.commit_strategy}")
        
        duration = time.time() - start_time

        metrics = {
            "total_resurrected": float(total_resurrected),
            "total_active_kept": float(total_active_kept),
        }

        return PhaseResult(Phase.COMMIT, 0, metrics, duration)
    
    # =========================================================================
    # Main Execution
    # =========================================================================
    
    def run(self, dataloader: DataLoader) -> CycleStats:
        """Execute complete cycle.
        
        Args:
            dataloader: Training data
            
        Returns:
            CycleStats with all metrics
        """
        start_time = time.time()
        self._log(f"Starting cycle {self.cycle_num}, target sparsity: {self.target_sparsity:.1%}")
        
        # Execute phases
        train_result = self.train_phase(dataloader)
        prune_result = self.prune_phase(dataloader)
        stabilize_result = self.stabilize_phase(dataloader)
        resu_result = self.resu_phase(dataloader)
        commit_result = self.commit_phase()
        
        # Compute final stats
        total_duration = time.time() - start_time
        actual_sparsity = self._get_model_sparsity()
        
        # Resurrection rate
        total_params = sum(m.weight.numel() for m in self.resu_modules.values())
        n_resurrected = commit_result.metrics.get("total_resurrected", 0)
        resurrection_rate = n_resurrected / max(int(total_params * self.target_sparsity), 1)
        
        stats = CycleStats(
            cycle=self.cycle_num,
            target_sparsity=self.target_sparsity,
            actual_sparsity=actual_sparsity,
            train_loss=train_result.metrics.get("loss", 0),
            train_steps=train_result.steps,
            stabilize_steps=stabilize_result.steps,
            resu_steps=resu_result.steps,
            resu_updates=int(resu_result.metrics.get("updates", 0)),
            mean_consistency=resu_result.metrics.get("mean_consistency", 0),
            mean_selection_ratio=resu_result.metrics.get("mean_selection_ratio", 0),
            resurrection_budget=self.amnesty.resurrection_budget(self.cycle_num),
            n_resurrected=n_resurrected,
            n_active_kept=commit_result.metrics.get("total_active_kept", 0),
            resurrection_rate=resurrection_rate,
            duration_seconds=total_duration,
        )
        
        self._log(f"Cycle complete: sparsity={actual_sparsity:.1%}, "
                 f"resurrected={n_resurrected}, duration={total_duration:.1f}s")
        
        return stats
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _get_model_sparsity(self) -> float:
        """Compute overall model sparsity."""
        total_params = 0
        total_zeros = 0
        
        for module in self.resu_modules.values():
            W = module.weight.data
            total_params += W.numel()
            total_zeros += (W == 0).sum().item()
        
        return total_zeros / total_params if total_params > 0 else 0.0


# =============================================================================
# Full Training Loop
# =============================================================================

class RESUTrainer:
    """Full RESU training across multiple cycles.
    
    Example:
        trainer = RESUTrainer(model, config, optimizer, train_fn)
        final_stats = trainer.train(train_loader, num_cycles=5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RESUConfig,
        optimizer: torch.optim.Optimizer,
        train_fn: Callable[[nn.Module, Any], torch.Tensor],
        pruner: Optional[Any] = None,
        stabilizer: Optional[Any] = None,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_fn = train_fn
        self.pruner = pruner
        self.stabilizer = stabilizer
        self.eval_fn = eval_fn
        self.device = device or next(model.parameters()).device
        self.logger = logger or print
        
        self.cycle_stats: List[CycleStats] = []
    
    def train(
        self,
        dataloader: DataLoader,
        num_cycles: Optional[int] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> List[CycleStats]:
        """Run full RESU training.
        
        Args:
            dataloader: Training data
            num_cycles: Override config num_cycles
            eval_dataloader: Optional evaluation data
            
        Returns:
            List of CycleStats for each cycle
        """
        num_cycles = num_cycles or self.config.num_cycles
        
        for cycle in range(num_cycles):
            # Create cycle
            resu_cycle = RESUCycle(
                model=self.model,
                config=self.config,
                optimizer=self.optimizer,
                train_fn=self.train_fn,
                cycle_num=cycle,
                pruner=self.pruner,
                stabilizer=self.stabilizer,
                device=self.device,
                logger=self.logger,
            )
            
            # Run cycle
            stats = resu_cycle.run(dataloader)
            self.cycle_stats.append(stats)
            
            # Evaluation
            if self.eval_fn is not None and (cycle + 1) % max(1, self.config.checkpoint_interval) == 0:
                eval_metrics = self.eval_fn(self.model)
                self.logger(f"[Eval] Cycle {cycle}: {eval_metrics}")
        
        return self.cycle_stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training."""
        if not self.cycle_stats:
            return {}
        
        return {
            "num_cycles": len(self.cycle_stats),
            "final_sparsity": self.cycle_stats[-1].actual_sparsity,
            "total_resurrected": sum(s.n_resurrected for s in self.cycle_stats),
            "total_duration": sum(s.duration_seconds for s in self.cycle_stats),
            "avg_resurrection_rate": sum(s.resurrection_rate for s in self.cycle_stats) / len(self.cycle_stats),
        }
