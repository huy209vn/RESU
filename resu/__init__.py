"""
RESU: Resurrection of Sparse Units

A sparse neural network training framework that allows pruned weights
to be resurrected through learned parameters.

Key Components:
- RESULinear: Drop-in replacement for nn.Linear
- ResurrectionEmbedding: Core Φ/Φ⁻¹ operations
- RESUSelective: Intelligent update filtering
- Amnesty: Fair resurrection tournament
- RESUTrainer: Full training orchestration

Example:
    from resu import RESULinear, RESUConfig, RESUTrainer, convert_to_resu
    
    # Convert model
    model = convert_to_resu(model)
    
    # Configure training
    config = RESUConfig(
        target_sparsity=0.7,
        num_cycles=5,
    )
    
    # Train with your existing train_fn
    trainer = RESUTrainer(model, config, optimizer, train_fn)
    trainer.train(dataloader)
"""

__version__ = "0.1.0"

# Core
from .core.mask import SparseMask, MaskStats
from .core.resurrection import ResurrectionEmbedding, StorageMode
from .core.effective import effective_weight, EffectiveWeightFunction
from .core.selective import (
    RESUSelective,
    SelectionConfig,
    SelectionResult,
)

# Modules
from .modules.linear import (
    RESULinear,
    RESUMode,
    convert_to_resu,
    get_resu_modules,
)

# Pruning
from .pruning.amnesty import (
    Amnesty,
    AmnestyConfig,
    AmnestyResult,
)

# Training
from .training.config import (
    RESUConfig,
    SparsitySchedule,
    DensificationSchedule,
    default_config,
    aggressive_pruning_config,
    conservative_pruning_config,
    densification_config,
    quick_test_config,
)
from .training.cycle import (
    RESUCycle,
    RESUTrainer,
    CycleStats,
    Phase,
)

__all__ = [
    # Version
    "__version__",
    
    # Core
    "SparseMask",
    "MaskStats",
    "ResurrectionEmbedding",
    "StorageMode",
    "effective_weight",
    "EffectiveWeightFunction",
    "RESUSelective",
    "SelectionConfig",
    "SelectionResult",
    
    # Modules
    "RESULinear",
    "RESUMode",
    "convert_to_resu",
    "get_resu_modules",
    
    # Pruning
    "Amnesty",
    "AmnestyConfig",
    "AmnestyResult",
    
    # Training
    "RESUConfig",
    "SparsitySchedule",
    "DensificationSchedule",
    "default_config",
    "aggressive_pruning_config",
    "conservative_pruning_config",
    "densification_config",
    "quick_test_config",
    "RESUCycle",
    "RESUTrainer",
    "CycleStats",
    "Phase",
]
