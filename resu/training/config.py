"""
RESU Training Configuration

Centralizes all hyperparameters for RESU training cycles.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum, auto


class SparsitySchedule(Enum):
    """How sparsity changes across cycles."""
    CONSTANT = auto()       # Same sparsity each cycle
    LINEAR = auto()         # Linear increase to target
    EXPONENTIAL = auto()    # Exponential increase
    COSINE = auto()         # Cosine annealing
    CUSTOM = auto()         # User-provided schedule


class DensificationSchedule(Enum):
    """For densification mode (decreasing sparsity)."""
    LINEAR = auto()
    STEPPED = auto()        # Discrete steps: [0.7, 0.5, 0.3, 0.1, 0.0]
    CUSTOM = auto()


@dataclass
class RESUConfig:
    """Full configuration for RESU training.
    
    Organized into logical groups:
    - Sparsity settings
    - Cycle structure
    - RESU phase parameters
    - Selective update parameters
    - Amnesty parameters
    - Pruning method
    - Optimizer settings
    """
    
    # =========================================================================
    # Sparsity
    # =========================================================================
    
    initial_sparsity: float = 0.0
    """Starting sparsity (0 = dense start)."""
    
    target_sparsity: float = 0.7
    """Final target sparsity."""
    
    sparsity_schedule: SparsitySchedule = SparsitySchedule.LINEAR
    """How to interpolate sparsity across cycles."""
    
    custom_sparsities: Optional[List[float]] = None
    """Custom sparsity per cycle (if schedule=CUSTOM)."""
    
    # =========================================================================
    # Densification Mode
    # =========================================================================
    
    densify: bool = False
    """If True, decrease sparsity each cycle (recover from pruning)."""
    
    densification_schedule: DensificationSchedule = DensificationSchedule.STEPPED
    """How to decrease sparsity in densification mode."""
    
    densification_steps: List[float] = field(
        default_factory=lambda: [0.7, 0.5, 0.3, 0.1, 0.0]
    )
    """Sparsity steps for STEPPED densification."""
    
    # =========================================================================
    # Cycle Structure
    # =========================================================================
    
    num_cycles: int = 5
    """Number of RESU cycles."""
    
    steps_per_cycle: int = 1000
    """Total steps per cycle."""
    
    train_fraction: float = 0.6
    """Fraction of cycle for training phase."""
    
    stabilize_fraction: float = 0.1
    """Fraction of cycle for DSNOT stabilization."""
    
    resu_fraction: float = 0.3
    """Fraction of cycle for RESU phase."""
    
    # Computed properties
    @property
    def train_steps(self) -> int:
        return int(self.steps_per_cycle * self.train_fraction)
    
    @property
    def stabilize_steps(self) -> int:
        return int(self.steps_per_cycle * self.stabilize_fraction)
    
    @property
    def resu_steps(self) -> int:
        return int(self.steps_per_cycle * self.resu_fraction)
    
    # =========================================================================
    # RESU Phase
    # =========================================================================
    
    resu_lr: float = 1e-4
    """Learning rate for θ updates."""
    
    resu_epsilon: float = 0.1
    """Initialization scale: θ ~ N(0, ε·σ_A)."""
    
    resu_init_type: Literal["normal", "uniform", "zero"] = "normal"
    """Initialization distribution for θ."""
    
    freeze_active_during_resu: bool = True
    """Whether to freeze active weights during RESU phase."""
    
    # =========================================================================
    # RESU-Selective
    # =========================================================================
    
    use_selective: bool = True
    """Whether to use RESU-Selective filtering."""
    
    selective_beta: float = 0.9
    """EMA coefficient for gradient tracking."""
    
    selective_delta: float = 1e-8
    """Stability constant for consistency computation."""
    
    selective_tau: float = 0.5
    """Consistency threshold for filtering."""
    
    selective_k_screen_ratio: float = 0.5
    """Fraction for magnitude screening (P_mag)."""
    
    selective_k_select_ratio: float = 0.2
    """Fraction for final selection (P_select)."""
    
    # =========================================================================
    # Amnesty
    # =========================================================================
    
    use_amnesty: bool = True
    """Whether to use amnesty mechanism."""

    commit_strategy: Literal["amnesty", "wanda_reprune", "simple"] = "amnesty"
    """Commit strategy after RESU phase:
    - 'amnesty': Tournament between active and resurrected (original)
    - 'wanda_reprune': Merge all θ → W, then re-prune with Wanda++/DSNoT
    - 'simple': Just merge θ → W without re-pruning
    """

    amnesty_r_start: float = 0.10
    """Initial resurrection budget."""

    amnesty_r_end: float = 0.02
    """Final resurrection budget."""

    amnesty_score_type: Literal["magnitude", "gradient", "wanda"] = "magnitude"
    """How to score weights for amnesty tournament."""
    
    # =========================================================================
    # Pruning
    # =========================================================================
    
    pruning_method: Literal["magnitude", "wanda", "random"] = "wanda"
    """Pruning method for initial/re-pruning."""
    
    pruning_granularity: Literal["element", "row", "column"] = "element"
    """Granularity of pruning."""
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    
    base_lr: float = 1e-3
    """Base learning rate for main optimizer."""
    
    weight_decay: float = 0.01
    """Weight decay."""
    
    warmup_steps: int = 100
    """Warmup steps per cycle."""
    
    # =========================================================================
    # Misc
    # =========================================================================
    
    seed: int = 42
    """Random seed."""
    
    log_interval: int = 50
    """Steps between logging."""
    
    eval_interval: int = 200
    """Steps between evaluation."""
    
    checkpoint_interval: int = 1
    """Cycles between checkpoints (0 = no checkpoints)."""
    
    # =========================================================================
    # Methods
    # =========================================================================
    
    def get_sparsity_for_cycle(self, cycle: int) -> float:
        """Get target sparsity for a given cycle.
        
        Args:
            cycle: Cycle number (0-indexed)
            
        Returns:
            Target sparsity for this cycle
        """
        if self.densify:
            return self._get_densification_sparsity(cycle)
        
        if self.sparsity_schedule == SparsitySchedule.CONSTANT:
            return self.target_sparsity
        
        elif self.sparsity_schedule == SparsitySchedule.CUSTOM:
            if self.custom_sparsities is None:
                raise ValueError("custom_sparsities required for CUSTOM schedule")
            return self.custom_sparsities[min(cycle, len(self.custom_sparsities) - 1)]
        
        # Interpolation schedules
        t = cycle / max(self.num_cycles - 1, 1)  # 0 to 1
        
        if self.sparsity_schedule == SparsitySchedule.LINEAR:
            return self.initial_sparsity + t * (self.target_sparsity - self.initial_sparsity)
        
        elif self.sparsity_schedule == SparsitySchedule.EXPONENTIAL:
            # Exponential: faster increase at start
            return self.initial_sparsity + (1 - (1 - t) ** 2) * (self.target_sparsity - self.initial_sparsity)
        
        elif self.sparsity_schedule == SparsitySchedule.COSINE:
            # Cosine: slower at extremes
            import math
            cos_t = (1 - math.cos(t * math.pi)) / 2
            return self.initial_sparsity + cos_t * (self.target_sparsity - self.initial_sparsity)
        
        else:
            return self.target_sparsity
    
    def _get_densification_sparsity(self, cycle: int) -> float:
        """Get sparsity for densification mode."""
        if self.densification_schedule == DensificationSchedule.STEPPED:
            idx = min(cycle, len(self.densification_steps) - 1)
            return self.densification_steps[idx]
        
        elif self.densification_schedule == DensificationSchedule.LINEAR:
            t = cycle / max(self.num_cycles - 1, 1)
            return self.initial_sparsity * (1 - t)
        
        else:
            return 0.0
    
    def validate(self):
        """Validate configuration."""
        assert 0 <= self.initial_sparsity <= 1
        assert 0 <= self.target_sparsity <= 1
        assert self.num_cycles >= 1
        assert self.steps_per_cycle >= 1
        assert abs(self.train_fraction + self.stabilize_fraction + self.resu_fraction - 1.0) < 0.01
        assert 0 < self.resu_lr
        assert 0 < self.resu_epsilon < 1
        assert 0 < self.selective_beta < 1
        assert 0 < self.selective_tau < 1
        assert 0 <= self.amnesty_r_end <= self.amnesty_r_start <= 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            # Sparsity
            "initial_sparsity": self.initial_sparsity,
            "target_sparsity": self.target_sparsity,
            "sparsity_schedule": self.sparsity_schedule.name,
            "custom_sparsities": self.custom_sparsities,
            # Densification
            "densify": self.densify,
            "densification_schedule": self.densification_schedule.name,
            "densification_steps": self.densification_steps,
            # Cycles
            "num_cycles": self.num_cycles,
            "steps_per_cycle": self.steps_per_cycle,
            "train_fraction": self.train_fraction,
            "stabilize_fraction": self.stabilize_fraction,
            "resu_fraction": self.resu_fraction,
            # RESU
            "resu_lr": self.resu_lr,
            "resu_epsilon": self.resu_epsilon,
            "resu_init_type": self.resu_init_type,
            "freeze_active_during_resu": self.freeze_active_during_resu,
            # Selective
            "use_selective": self.use_selective,
            "selective_beta": self.selective_beta,
            "selective_delta": self.selective_delta,
            "selective_tau": self.selective_tau,
            "selective_k_screen_ratio": self.selective_k_screen_ratio,
            "selective_k_select_ratio": self.selective_k_select_ratio,
            # Amnesty
            "use_amnesty": self.use_amnesty,
            "amnesty_r_start": self.amnesty_r_start,
            "amnesty_r_end": self.amnesty_r_end,
            "amnesty_score_type": self.amnesty_score_type,
            # Pruning
            "pruning_method": self.pruning_method,
            "pruning_granularity": self.pruning_granularity,
            # Optimizer
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            # Misc
            "seed": self.seed,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RESUConfig":
        """Create from dictionary."""
        d = d.copy()
        if "sparsity_schedule" in d:
            d["sparsity_schedule"] = SparsitySchedule[d["sparsity_schedule"]]
        if "densification_schedule" in d:
            d["densification_schedule"] = DensificationSchedule[d["densification_schedule"]]
        return cls(**d)


# =============================================================================
# Preset Configurations
# =============================================================================

def default_config() -> RESUConfig:
    """Default RESU configuration."""
    return RESUConfig()


def aggressive_pruning_config() -> RESUConfig:
    """Configuration for aggressive pruning (high sparsity)."""
    return RESUConfig(
        target_sparsity=0.9,
        num_cycles=7,
        steps_per_cycle=2000,
        amnesty_r_start=0.15,
        amnesty_r_end=0.05,
    )


def conservative_pruning_config() -> RESUConfig:
    """Configuration for conservative pruning."""
    return RESUConfig(
        target_sparsity=0.5,
        num_cycles=3,
        steps_per_cycle=500,
        use_selective=False,
        amnesty_r_start=0.20,
    )


def densification_config() -> RESUConfig:
    """Configuration for densification (recovering from pruning)."""
    return RESUConfig(
        densify=True,
        initial_sparsity=0.7,
        num_cycles=5,
        densification_steps=[0.7, 0.5, 0.3, 0.1, 0.0],
        amnesty_r_start=0.30,  # More generous resurrection budget
        amnesty_r_end=0.10,
    )


def quick_test_config() -> RESUConfig:
    """Minimal configuration for quick testing."""
    return RESUConfig(
        target_sparsity=0.5,
        num_cycles=2,
        steps_per_cycle=100,
        train_fraction=0.5,
        stabilize_fraction=0.1,
        resu_fraction=0.4,
        log_interval=10,
    )
