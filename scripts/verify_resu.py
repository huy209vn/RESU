"""
End-to-end verification that RESU works.

This script:
1. Trains a small model on a toy task
2. Prunes it aggressively
3. Runs RESU to resurrect weights
4. Verifies that resurrection actually happens
5. Checks that performance recovers
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resu.modules.linear import RESULinear, convert_to_resu, get_resu_modules
from resu.training.config import RESUConfig
from resu.training.cycle import RESUCycle
from resu.pruning.amnesty import Amnesty, AmnestyConfig


def create_toy_dataset(n_samples=1000, n_features=64, n_classes=10, seed=42):
    """Create a simple classification dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)

    # Create true weights for classes
    W_true = torch.randn(n_classes, n_features)

    # Generate labels
    logits = X @ W_true.T
    y = logits.argmax(dim=1)

    return TensorDataset(X, y), W_true


def create_model(n_features, n_hidden, n_classes):
    """Create a simple 2-layer model."""
    return nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_classes),
    )


def train_model(model, dataloader, optimizer, device, epochs=10):
    """Train model for a few epochs."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(dataloader):.4f}, Acc = {acc:.1f}%")

    return acc


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100 * correct / total


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"RESU End-to-End Verification")
    print(f"{'='*70}")
    print(f"Device: {device}\n")

    # Hyperparameters
    n_features = 64
    n_hidden = 128
    n_classes = 10
    sparsity = 0.7
    batch_size = 32

    # Create dataset
    print("Creating toy dataset...")
    dataset, _ = create_toy_dataset(n_samples=1000, n_features=n_features, n_classes=n_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 1: Train dense model
    print("\n[Step 1] Training dense model...")
    model = create_model(n_features, n_hidden, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dense_acc = train_model(model, dataloader, optimizer, device, epochs=20)
    print(f"✓ Dense model accuracy: {dense_acc:.1f}%")

    # Step 2: Convert to RESU and prune
    print(f"\n[Step 2] Converting to RESU and pruning to {sparsity:.0%} sparsity...")
    model = convert_to_resu(model)
    resu_modules = get_resu_modules(model)

    for module in resu_modules.values():
        module.prune_by_magnitude(sparsity)

    # Evaluate sparse model
    sparse_acc = evaluate_model(model, dataloader, device)
    print(f"✓ Sparse model accuracy: {sparse_acc:.1f}%")
    print(f"  Performance drop: {dense_acc - sparse_acc:.1f}%")

    # Step 3: Run RESU
    print(f"\n[Step 3] Running RESU resurrection...")

    # Enter RESU mode
    for module in resu_modules.values():
        module.enter_resu_mode(epsilon=0.2, use_selective=True, lr=0.01)

    # Track which weights were pruned
    original_masks = {name: m.mask.mask.clone() for name, m in resu_modules.items()}

    # RESU training
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # RESU step
            for module in resu_modules.values():
                if module.is_resu_active:
                    grad = module.weight.grad if module.weight.grad is not None else torch.zeros_like(module.weight)
                    module.resu_step(grad)

            model.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"  RESU epoch {epoch+1}/30")

    # Commit RESU
    for module in resu_modules.values():
        module.exit_resu_mode(commit=True)

    print("✓ RESU phase complete")

    # Step 4: Apply amnesty
    print(f"\n[Step 4] Applying amnesty mechanism...")
    amnesty = Amnesty(AmnestyConfig(r_start=0.30, r_end=0.10, total_cycles=1))

    total_resurrected = 0
    for name, module in resu_modules.items():
        old_mask = module.mask
        W_eff = module.weight.data

        new_mask, result = amnesty.commit_with_amnesty(
            W_eff=W_eff,
            old_mask=old_mask,
            target_sparsity=sparsity,
            cycle=0,
        )

        module.set_mask(new_mask)
        total_resurrected += result.n_resurrected_kept

        print(f"  {name}: {result.n_resurrected_kept} weights resurrected")

    print(f"✓ Total resurrected weights: {total_resurrected}")

    # Verify resurrection happened
    resurrection_happened = total_resurrected > 0

    if resurrection_happened:
        print("  ✓ VERIFICATION PASSED: Weights were resurrected!")
    else:
        print("  ✗ VERIFICATION FAILED: No weights resurrected")
        return False

    # Step 5: Fine-tune and evaluate
    print(f"\n[Step 5] Fine-tuning after resurrection...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    _ = train_model(model, dataloader, optimizer, device, epochs=10)

    final_acc = evaluate_model(model, dataloader, device)
    print(f"✓ Final accuracy: {final_acc:.1f}%")

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Dense model:           {dense_acc:.1f}%")
    print(f"Sparse (no RESU):      {sparse_acc:.1f}% (Δ = {sparse_acc - dense_acc:+.1f}%)")
    print(f"After RESU + Amnesty:  {final_acc:.1f}% (Δ = {final_acc - dense_acc:+.1f}%)")
    print(f"\nRecovery: {final_acc - sparse_acc:.1f}% improvement")
    print(f"Weights resurrected: {total_resurrected}")

    # Final verification
    recovery = final_acc - sparse_acc
    success = resurrection_happened and recovery > 0

    if success:
        print(f"\n{'='*70}")
        print("✓ VERIFICATION SUCCESSFUL!")
        print(f"{'='*70}")
        print("RESU successfully:")
        print(f"  1. Resurrected {total_resurrected} pruned weights")
        print(f"  2. Improved accuracy by {recovery:.1f}%")
        print(f"  3. Recovered {100*recovery/(dense_acc - sparse_acc):.1f}% of lost performance")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"\n{'='*70}")
        print("✗ VERIFICATION FAILED")
        print(f"{'='*70}\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
