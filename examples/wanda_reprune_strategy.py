"""
Example: Using RESU+Wanda+DSNoT commit strategy

Instead of amnesty tournament, merge ALL resurrections and re-prune with Wanda++.
This leverages structure-aware pruning after resurrection.
"""

import torch
import torch.nn as nn
from resu.training.config import RESUConfig
from resu.training.cycle import RESUCycle
from resu.modules.linear import convert_to_resu


def simple_train_fn(model, batch):
    """Simple training function."""
    x, y = batch
    logits = model(x)
    return nn.functional.cross_entropy(logits, y)


# Create model
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# Convert to RESU
model = convert_to_resu(model)

# Create config with wanda_reprune strategy
config = RESUConfig(
    target_sparsity=0.5,
    num_cycles=3,
    steps_per_cycle=100,

    # KEY: Use wanda_reprune instead of amnesty
    commit_strategy="wanda_reprune",  # Options: "amnesty", "wanda_reprune", "simple"

    # RESU settings
    use_selective=True,
    resu_epsilon=0.1,

    # Pruning (will be used for re-pruning after merge)
    pruning_method="wanda",
)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
train_data = [(torch.randn(32, 128), torch.randint(0, 10, (32,))) for _ in range(10)]
dataloader = train_data

# Create cycle
cycle = RESUCycle(
    model=model,
    config=config,
    optimizer=optimizer,
    train_fn=simple_train_fn,
    cycle_num=0,
)

# Run one cycle
print("Running RESU cycle with wanda_reprune strategy...")
stats = cycle.run(dataloader)

print(f"\nCycle complete!")
print(f"  Sparsity: {stats.actual_sparsity:.1%}")
print(f"  Resurrections: {stats.n_resurrected}")
print(f"  Strategy: wanda_reprune (merged all θ, will re-prune next cycle)")
