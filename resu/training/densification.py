"""
Densification Training with RL Pauses

RESU densification: progressively decrease sparsity from high to zero,
with pause points for reinforcement learning or other training modalities.

Example schedule:
    Cycle 0: 70% sparse → train → RESU
    [PAUSE for RL training]
    Cycle 1: 50% sparse → train → RESU
    [PAUSE for RL training]
    Cycle 2: 30% sparse → train → RESU
    [PAUSE for RL training]
    Cycle 3: 10% sparse → train → RESU
    [PAUSE for RL training]
    Cycle 4: 0% sparse (fully dense)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from .config import RESUConfig
from .cycle import RESUCycle, RESUTrainer, CycleStats
from ..modules.linear import get_resu_modules


class PauseReason(Enum):
    """Reason for pause."""
    RL_TRAINING = auto()
    EVALUATION = auto()
    CHECKPOINT = auto()
    CUSTOM = auto()


@dataclass
class PauseConfig:
    """Configuration for a pause point."""
    after_cycle: int
    """Pause after this cycle number (0-indexed)."""

    reason: PauseReason = PauseReason.RL_TRAINING
    """Why we're pausing."""

    duration_steps: Optional[int] = None
    """Suggested duration in steps (for RL training, etc)."""

    callback: Optional[Callable[[nn.Module, int], None]] = None
    """Callback function(model, cycle) → None executed during pause."""

    metadata: Dict[str, Any] = None
    """Additional metadata for this pause."""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DensificationSchedule:
    """Sparsity schedule for densification.

    Example:
        # Linear densification over 5 cycles
        schedule = DensificationSchedule.linear(
            start_sparsity=0.7,
            end_sparsity=0.0,
            num_cycles=5,
        )

        # Custom stepped schedule
        schedule = DensificationSchedule(
            sparsities=[0.8, 0.6, 0.4, 0.2, 0.0],
            pauses=[
                PauseConfig(after_cycle=0, reason=PauseReason.RL_TRAINING),
                PauseConfig(after_cycle=2, reason=PauseReason.RL_TRAINING),
                PauseConfig(after_cycle=4, reason=PauseReason.EVALUATION),
            ]
        )
    """

    sparsities: List[float]
    """Sparsity for each cycle (decreasing)."""

    pauses: List[PauseConfig]
    """Pause points during training."""

    @classmethod
    def linear(
        cls,
        start_sparsity: float,
        end_sparsity: float,
        num_cycles: int,
        pause_every: int = 1,
        pause_reason: PauseReason = PauseReason.RL_TRAINING,
    ) -> "DensificationSchedule":
        """Create linear densification schedule.

        Args:
            start_sparsity: Initial sparsity (e.g., 0.7)
            end_sparsity: Final sparsity (e.g., 0.0 for fully dense)
            num_cycles: Number of cycles
            pause_every: Pause after every N cycles
            pause_reason: Reason for pauses

        Returns:
            DensificationSchedule with linear spacing
        """
        sparsities = []
        for i in range(num_cycles):
            t = i / max(num_cycles - 1, 1)
            s = start_sparsity + t * (end_sparsity - start_sparsity)
            sparsities.append(s)

        pauses = [
            PauseConfig(after_cycle=i, reason=pause_reason)
            for i in range(0, num_cycles - 1, pause_every)
        ]

        return cls(sparsities=sparsities, pauses=pauses)

    @classmethod
    def stepped(
        cls,
        steps: List[float],
        pause_after_each: bool = True,
        pause_reason: PauseReason = PauseReason.RL_TRAINING,
    ) -> "DensificationSchedule":
        """Create stepped densification schedule.

        Args:
            steps: Explicit sparsity values (e.g., [0.7, 0.5, 0.3, 0.1, 0.0])
            pause_after_each: Add pause after each step
            pause_reason: Reason for pauses

        Returns:
            DensificationSchedule with explicit steps
        """
        pauses = []
        if pause_after_each:
            pauses = [
                PauseConfig(after_cycle=i, reason=pause_reason)
                for i in range(len(steps) - 1)
            ]

        return cls(sparsities=steps, pauses=pauses)

    def get_sparsity(self, cycle: int) -> float:
        """Get sparsity for given cycle."""
        idx = min(cycle, len(self.sparsities) - 1)
        return self.sparsities[idx]

    def should_pause(self, cycle: int) -> Optional[PauseConfig]:
        """Check if should pause after this cycle."""
        for pause in self.pauses:
            if pause.after_cycle == cycle:
                return pause
        return None


class DensificationTrainer(RESUTrainer):
    """RESU trainer with densification schedule and pause support.

    Extends RESUTrainer to support:
    - Decreasing sparsity schedule
    - Pause points for RL training
    - Custom callbacks during pauses

    Example:
        # Create schedule
        schedule = DensificationSchedule.linear(
            start_sparsity=0.7,
            end_sparsity=0.0,
            num_cycles=5,
            pause_every=1,
        )

        # Define RL callback
        def rl_training_callback(model, cycle):
            print(f"Running RL training after cycle {cycle}")
            # Your RL training code here
            run_ppo_training(model, num_steps=10000)

        # Add callbacks to pauses
        for pause in schedule.pauses:
            pause.callback = rl_training_callback

        # Create trainer
        trainer = DensificationTrainer(
            model=model,
            config=resu_config,
            optimizer=optimizer,
            train_fn=supervised_train_fn,
            schedule=schedule,
        )

        # Train (will pause automatically for RL)
        stats = trainer.train_with_densification(train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: RESUConfig,
        optimizer: torch.optim.Optimizer,
        train_fn: Callable[[nn.Module, Any], torch.Tensor],
        schedule: DensificationSchedule,
        pruner: Optional[Any] = None,
        stabilizer: Optional[Any] = None,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Callable] = None,
    ):
        """
        Args:
            model: Model with RESULinear layers
            config: RESU configuration
            optimizer: Main optimizer
            train_fn: Training step function
            schedule: Densification schedule with pauses
            pruner: Pruning implementation
            stabilizer: DSNOT stabilizer
            eval_fn: Evaluation function
            device: Target device
            logger: Logging function
        """
        super().__init__(
            model=model,
            config=config,
            optimizer=optimizer,
            train_fn=train_fn,
            pruner=pruner,
            stabilizer=stabilizer,
            eval_fn=eval_fn,
            device=device,
            logger=logger,
        )

        self.schedule = schedule
        self.pause_callbacks: Dict[int, List[Callable]] = {}

    def add_pause_callback(
        self,
        cycle: int,
        callback: Callable[[nn.Module, int], None],
    ):
        """Add a callback to execute during pause after given cycle.

        Args:
            cycle: Cycle number after which to pause
            callback: Function(model, cycle) to execute
        """
        if cycle not in self.pause_callbacks:
            self.pause_callbacks[cycle] = []
        self.pause_callbacks[cycle].append(callback)

    def _execute_pause(self, pause_config: PauseConfig, cycle: int):
        """Execute pause and run callbacks.

        Args:
            pause_config: Pause configuration
            cycle: Current cycle number
        """
        self.logger(f"\n{'='*60}")
        self.logger(f"PAUSE after cycle {cycle}: {pause_config.reason.name}")
        self.logger(f"{'='*60}")

        # Execute pause callback
        if pause_config.callback is not None:
            self.logger(f"Executing pause callback...")
            pause_config.callback(self.model, cycle)

        # Execute any additional callbacks
        if cycle in self.pause_callbacks:
            for callback in self.pause_callbacks[cycle]:
                self.logger(f"Executing additional callback...")
                callback(self.model, cycle)

        self.logger(f"{'='*60}")
        self.logger(f"Resuming RESU training...")
        self.logger(f"{'='*60}\n")

    def train_with_densification(
        self,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> List[CycleStats]:
        """Run RESU training with densification schedule.

        Automatically pauses at configured points for RL training or
        other custom callbacks.

        Args:
            dataloader: Training data
            eval_dataloader: Optional evaluation data

        Returns:
            List of CycleStats for each cycle
        """
        num_cycles = len(self.schedule.sparsities)

        for cycle in range(num_cycles):
            # Override config target sparsity from schedule
            target_sparsity = self.schedule.get_sparsity(cycle)
            self.config.target_sparsity = target_sparsity

            self.logger(f"\n{'='*60}")
            self.logger(f"CYCLE {cycle}/{num_cycles}: Target sparsity = {target_sparsity:.1%}")
            self.logger(f"{'='*60}")

            # Create and run cycle
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

            stats = resu_cycle.run(dataloader)
            self.cycle_stats.append(stats)

            # Evaluation
            if self.eval_fn is not None and (cycle + 1) % max(1, self.config.checkpoint_interval) == 0:
                eval_metrics = self.eval_fn(self.model)
                self.logger(f"[Eval] Cycle {cycle}: {eval_metrics}")

            # Check for pause
            pause_config = self.schedule.should_pause(cycle)
            if pause_config is not None:
                self._execute_pause(pause_config, cycle)

        return self.cycle_stats


# =============================================================================
# Utility Functions
# =============================================================================

def create_rl_training_schedule(
    initial_sparsity: float = 0.7,
    rl_cycles: int = 4,
    final_dense: bool = True,
) -> DensificationSchedule:
    """Create a typical schedule for RL training with RESU.

    Starts sparse, progressively densifies with RL training at each stage.

    Args:
        initial_sparsity: Starting sparsity (e.g., 0.7 = 70% weights pruned)
        rl_cycles: Number of RL training cycles
        final_dense: Whether to end fully dense

    Returns:
        DensificationSchedule configured for RL training
    """
    # Create stepped sparsity schedule
    if final_dense:
        # e.g., [0.7, 0.5, 0.3, 0.1, 0.0]
        sparsities = []
        for i in range(rl_cycles + 1):
            s = initial_sparsity * (1 - i / rl_cycles)
            sparsities.append(s)
    else:
        # e.g., [0.7, 0.55, 0.4, 0.25, 0.1]
        sparsities = []
        for i in range(rl_cycles):
            s = initial_sparsity * (1 - i / (rl_cycles - 1))
            sparsities.append(s)

    # Pause after each cycle for RL
    pauses = [
        PauseConfig(
            after_cycle=i,
            reason=PauseReason.RL_TRAINING,
            duration_steps=10000,  # Suggested: 10k RL steps
            metadata={"rl_phase": i + 1},
        )
        for i in range(len(sparsities) - 1)
    ]

    return DensificationSchedule(sparsities=sparsities, pauses=pauses)


def get_current_model_sparsity(model: nn.Module) -> float:
    """Compute overall model sparsity."""
    resu_modules = get_resu_modules(model)
    if not resu_modules:
        return 0.0

    total_params = 0
    total_zeros = 0

    for module in resu_modules.values():
        W = module.weight.data
        total_params += W.numel()
        total_zeros += (W == 0).sum().item()

    return total_zeros / total_params if total_params > 0 else 0.0


def print_densification_summary(stats_list: List[CycleStats]):
    """Print summary of densification training."""
    print("\n" + "=" * 70)
    print("DENSIFICATION TRAINING SUMMARY")
    print("=" * 70)

    print(f"{'Cycle':<6} {'Sparsity':<10} {'Loss':<10} {'Resurrected':<12} {'Duration':<10}")
    print("-" * 70)

    for stats in stats_list:
        print(
            f"{stats.cycle:<6} "
            f"{stats.actual_sparsity:>7.1%}   "
            f"{stats.train_loss:>9.4f} "
            f"{stats.n_resurrected:>11} "
            f"{stats.duration_seconds:>9.1f}s"
        )

    print("-" * 70)

    total_resurrected = sum(s.n_resurrected for s in stats_list)
    total_duration = sum(s.duration_seconds for s in stats_list)
    final_sparsity = stats_list[-1].actual_sparsity if stats_list else 0.0

    print(f"Total resurrected: {total_resurrected}")
    print(f"Final sparsity: {final_sparsity:.1%}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print("=" * 70 + "\n")
