"""
RESU Training Example with Qwen2.5

Shows how to use RESU with your existing Wanda++ and DSNoT.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your existing pruning code
from prune import PruneConfig

# RESU imports
from resu import (
    RESUConfig,
    RESUTrainer,
    convert_to_resu,
)
from resu.pruning.integration import (
    WandaPlusPruner,
    DSNoTStabilizer,
    WandaDSNoTConfig,
    create_pruner_and_stabilizer,
)


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    
    device = torch.device("cuda")
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.seqlen = 2048
    
    # =========================================================================
    # Convert to RESU
    # =========================================================================
    
    print("Converting model to RESU...")
    model = convert_to_resu(model)
    
    # =========================================================================
    # Create Pruner and Stabilizer
    # =========================================================================
    
    wanda_config = WandaDSNoTConfig(
        nsamples=128,
        sparsity=0.5,
        wanda_alpha=100.0,
        dsnot_gamma=1.0,
        dsnot_cycles=50,
    )
    
    pruner, make_stabilizer = create_pruner_and_stabilizer(
        model, tokenizer, wanda_config, device
    )
    
    # =========================================================================
    # Configure RESU Training
    # =========================================================================
    
    resu_config = RESUConfig(
        # Sparsity
        initial_sparsity=0.0,
        target_sparsity=0.5,
        
        # Cycles
        num_cycles=3,
        steps_per_cycle=500,
        train_fraction=0.5,
        stabilize_fraction=0.1,  # DSNoT handles this
        resu_fraction=0.4,
        
        # RESU
        resu_lr=1e-4,
        resu_epsilon=0.1,
        use_selective=True,
        
        # Amnesty
        use_amnesty=True,
        amnesty_r_start=0.10,
        amnesty_r_end=0.02,
    )
    
    # =========================================================================
    # Define Training Function
    # =========================================================================
    
    def train_fn(model, batch):
        """Your training step - returns loss."""
        input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        
        outputs = model(input_ids, labels=input_ids)
        return outputs.loss
    
    # =========================================================================
    # Create Optimizer
    # =========================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resu_config.base_lr,
        weight_decay=resu_config.weight_decay,
    )
    
    # =========================================================================
    # Run RESU Training
    # =========================================================================
    
    # First cycle: calibrate pruner
    pruner.calibrate(resu_config.target_sparsity)
    stabilizer = make_stabilizer()
    
    trainer = RESUTrainer(
        model=model,
        config=resu_config,
        optimizer=optimizer,
        train_fn=train_fn,
        pruner=pruner,
        stabilizer=stabilizer,
        device=device,
    )
    
    # You need a dataloader here
    # train_loader = ...
    
    # stats = trainer.train(train_loader)
    # print(trainer.get_training_summary())
    
    print("Setup complete! Provide train_loader to run training.")


# =========================================================================
# Simpler Usage: Just RESU without full training loop
# =========================================================================

def simple_resu_example():
    """Minimal example of using RESULinear directly."""
    
    from resu import RESULinear, SparseMask
    
    device = torch.device("cuda")
    
    # Create layer
    layer = RESULinear(768, 3072, device=device)
    
    # Prune to 50%
    layer.prune_by_magnitude(0.5)
    print(f"After pruning: {layer}")
    
    # Enter RESU mode
    layer.enter_resu_mode(epsilon=0.1, use_selective=True, lr=1e-4)
    print(f"In RESU mode: {layer}")
    
    # Forward pass
    x = torch.randn(4, 128, 768, device=device)
    y = layer(x)
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    # RESU step (updates θ)
    stats = layer.resu_step()
    print(f"RESU step: {stats}")
    
    # Commit θ to weights
    layer.exit_resu_mode(commit=True)
    print(f"After commit: {layer}")


if __name__ == "__main__":
    # main()
    simple_resu_example()
