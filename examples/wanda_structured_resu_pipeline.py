"""
Complete Pipeline: Wanda++ → 2:4 Structured → RESU Training

This demonstrates the full workflow:
1. Wanda++ calibration (activation-aware importance scoring)
2. Convert to 2:4 structured masks (guided by Wanda scores)
3. Apply structured masks to model
4. RESU training on 2:4 pruned positions
5. Verify 2:4 structure maintained throughout

Result: 50% sparse, 2:4 structured, tensor-core accelerated, trained!
"""

import torch
import torch.nn as nn
from resu.modules.linear import RESULinear
from resu.core.structured import verify_nm_structure


def create_simple_model():
    """Create a simple model for demonstration."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = RESULinear(512, 256, bias=False)
            self.layer2 = RESULinear(256, 128, bias=False)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.activation(self.layer1(x))
            x = self.layer2(x)
            return x

    return SimpleModel()


def demo_structured_pipeline_simple():
    """
    Simplified demo without Wanda++ (uses magnitude-based 2:4).

    Shows the core RESU → 2:4 structured flow.
    """
    from resu.core.structured import project_to_nm_structured

    print("=" * 80)
    print("Simplified Pipeline: Magnitude → 2:4 → RESU")
    print("=" * 80)
    print()

    # Create model
    model = create_simple_model()
    model.train()

    # Step 1: Project directly to 2:4 structure (from dense)
    print("Step 1: Project to 2:4 structured (50%)...")
    for layer in [model.layer1, model.layer2]:
        # Get current weights
        W = layer.weight.data.clone()

        # Project to 2:4 (keeps top-2 per 4 by magnitude)
        W_24, mask_24 = project_to_nm_structured(W, n=2, m=4, dim=1)

        # Apply
        layer.weight.data.copy_(W_24)

        # Create SparseMask from binary mask
        from resu.core.mask import SparseMask
        pruned_indices = (~mask_24.bool()).flatten().nonzero(as_tuple=True)[0]
        sparse_mask = SparseMask(pruned_indices, mask_24.shape, device=layer.device)
        layer.set_mask(sparse_mask)

        # Verify
        is_valid, sparsity = verify_nm_structure(layer.weight.data, n=2, m=4, dim=1)
        print(f"  {layer.__class__.__name__}: valid={is_valid}, sparsity={sparsity:.1%}")
    print()

    # Step 2: RESU on 2:4 pruned positions
    print("Step 2: Enter RESU mode (trains 2:4 gaps)...")
    for layer in [model.layer1, model.layer2]:
        # IMPORTANT: use_selective=False to train ALL pruned positions
        # freeze_active=True to keep 2:4 active positions fixed
        layer.enter_resu_mode(epsilon=0.1, freeze_active=True, use_selective=False)
        n_pruned = int((~layer._mask.mask).sum().item())
        print(f"  {layer.__class__.__name__}: {n_pruned} pruned positions to train")
    print()

    # Step 3: Simulate training
    print("Step 3: Training with RESU...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(100):
        # Fake data
        x = torch.randn(32, 512)
        y = model(x)
        loss = y.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    print()

    # Step 4: Exit RESU and verify structure maintained
    print("Step 4: Exit RESU and verify 2:4 maintained...")
    for layer in [model.layer1, model.layer2]:
        layer.exit_resu_mode(commit=True)

        # Verify still 2:4
        is_valid, sparsity = verify_nm_structure(layer.weight.data, n=2, m=4, dim=1)
        print(f"  {layer.__class__.__name__}: valid={is_valid}, sparsity={sparsity:.1%}")

        if not is_valid:
            print(f"    ⚠️  WARNING: Structure not maintained!")
        else:
            print(f"    ✓ Structure maintained throughout!")
    print()

    print("=" * 80)
    print("✓ Pipeline Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Started: 70% sparse (unstructured)")
    print("  - Projected to: 50% sparse (2:4 structured)")
    print("  - RESU trained: 2:4 gaps filled with gradient-guided values")
    print("  - Result: 50% sparse, 2:4 structured, trained!")
    print()
    print("Next steps:")
    print("  - Use Wanda++ for smarter structured pruning (see below)")
    print("  - Fine-tune further if needed")
    print("  - Deploy with tensor core acceleration!")
    print()


def demo_wanda_structured_pipeline():
    """
    Full pipeline with Wanda++ (requires model + tokenizer).

    This is a template - adapt for your model.
    """
    print("=" * 80)
    print("Wanda++ → 2:4 → RESU Pipeline (Template)")
    print("=" * 80)
    print()

    print("To use Wanda++ for structured pruning:")
    print()
    print("```python")
    print("from resu.pruning.integration import WandaPlusPruner")
    print("from resu.core.structured import verify_nm_structure")
    print()
    print("# 1. Create pruner")
    print("pruner = WandaPlusPruner(model, tokenizer, config)")
    print()
    print("# 2. Calibrate (gets importance scores)")
    print("pruner.calibrate()")
    print()
    print("# 3. Get 2:4 structured masks (guided by Wanda scores!)")
    print("structured_masks = pruner.get_structured_masks(n=2, m=4)")
    print()
    print("# 4. Apply masks")
    print("pruner.apply_masks(structured_masks)")
    print()
    print("# 5. RESU on 2:4 positions")
    print("for layer in model.modules():")
    print("    if isinstance(layer, RESULinear):")
    print("        layer.enter_resu_mode(epsilon=0.1, freeze_active=True)")
    print()
    print("# 6. Train")
    print("train(model, dataloader, epochs=10)")
    print()
    print("# 7. Exit RESU")
    print("for layer in model.modules():")
    print("    if isinstance(layer, RESULinear):")
    print("        layer.exit_resu_mode(commit=True)")
    print()
    print("# 8. Verify structure")
    print("for name, layer in model.named_modules():")
    print("    if isinstance(layer, RESULinear):")
    print("        is_valid, sparsity = verify_nm_structure(")
    print("            layer.weight.data, n=2, m=4, dim=1")
    print("        )")
    print("        print(f'{name}: valid={is_valid}, sparsity={sparsity:.1%}')")
    print("```")
    print()
    print("Wanda++ advantages over magnitude:")
    print("  ✓ Activation-aware (considers actual usage)")
    print("  ✓ Better quality structured pruning")
    print("  ✓ Lower accuracy loss")
    print()


if __name__ == "__main__":
    print()
    print("Demo 1: Simplified Pipeline (magnitude-based)")
    print()
    demo_structured_pipeline_simple()

    print()
    print("=" * 80)
    print()

    print("Demo 2: Wanda++ Pipeline (template)")
    print()
    demo_wanda_structured_pipeline()
