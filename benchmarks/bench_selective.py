"""
Benchmark: RESU-Selective vs Vanilla RESU

Compares directional consistency filtering vs standard resurrection.

Metrics:
- Resurrection quality (how many survive amnesty)
- Training stability (loss variance)
- Final accuracy
- Memory overhead (should be similar)
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resu.modules.linear import RESULinear


def create_synthetic_task(
    in_features=128,
    out_features=64,
    hidden_dim=256,
    num_samples=1000,
    device=torch.device("cuda"),
):
    """Create synthetic classification task."""

    # Simple 2-layer network
    true_model = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_features),
    ).to(device)

    # Generate data
    X = torch.randn(num_samples, in_features, device=device)
    with torch.no_grad():
        y = true_model(X).argmax(dim=1)

    return X, y, true_model


def train_resu_layer(
    layer: RESULinear,
    X: torch.Tensor,
    y_target: torch.Tensor,
    use_selective: bool,
    num_epochs=50,
    batch_size=32,
) -> Dict[str, List[float]]:
    """Train a single RESU layer and track metrics."""

    # Prune to 50%
    layer.prune_by_magnitude(0.5)

    # Enter RESU mode
    layer.enter_resu_mode(
        epsilon=0.1,
        use_selective=use_selective,
        lr=0.001,
    )

    # Track metrics
    losses = []
    grad_norms = []
    consistency_scores = []

    num_samples = len(X)
    num_batches = num_samples // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        # Shuffle data
        perm = torch.randperm(num_samples)
        X_shuffled = X[perm]
        y_shuffled = y_target[perm]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward
            logits = layer(x_batch)
            loss = nn.functional.cross_entropy(logits, y_batch)

            # Backward
            loss.backward()

            # RESU step
            if layer.weight.grad is not None:
                grad_matrix = layer.weight.grad
                stats = layer.resu_step(grad_matrix)

                # Track consistency if selective
                if use_selective and stats:
                    consistency_scores.append(stats.get("mean_consistency", 0))

                # Track gradient norm
                grad_norm = grad_matrix.norm().item()
                epoch_grad_norm += grad_norm

            epoch_loss += loss.item()

            # Zero gradients
            layer.weight.grad = None

        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches

        losses.append(avg_loss)
        grad_norms.append(avg_grad_norm)

    # Exit RESU
    layer.exit_resu_mode(commit=True)

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "consistency_scores": consistency_scores,
        "final_loss": losses[-1] if losses else float('inf'),
        "loss_std": torch.tensor(losses).std().item() if losses else 0.0,
    }


def benchmark_selective(
    in_features=128,
    out_features=64,
    sparsity=0.5,
    num_runs=3,
    device=torch.device("cuda"),
):
    """Compare selective vs non-selective RESU."""

    print(f"\n{'='*70}")
    print(f"RESU-Selective vs Vanilla RESU Benchmark")
    print(f"{'='*70}")
    print(f"Features: {in_features} → {out_features}")
    print(f"Sparsity: {sparsity:.0%}")
    print(f"Runs: {num_runs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Create synthetic task
    X, y, true_model = create_synthetic_task(in_features, out_features, device=device)

    results = {
        "vanilla": [],
        "selective": [],
    }

    # Run multiple times for statistical significance
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")

        # 1. Vanilla RESU (no selective)
        print("  Training: Vanilla RESU...")
        layer_vanilla = RESULinear(in_features, out_features, device=device)
        vanilla_metrics = train_resu_layer(
            layer_vanilla,
            X,
            y,
            use_selective=False,
            num_epochs=50,
        )
        results["vanilla"].append(vanilla_metrics)
        print(f"    Final loss: {vanilla_metrics['final_loss']:.4f}")

        # 2. RESU-Selective
        print("  Training: RESU-Selective...")
        layer_selective = RESULinear(in_features, out_features, device=device)
        selective_metrics = train_resu_layer(
            layer_selective,
            X,
            y,
            use_selective=True,
            num_epochs=50,
        )
        results["selective"].append(selective_metrics)
        print(f"    Final loss: {selective_metrics['final_loss']:.4f}")
        print(f"    Avg consistency: {torch.tensor(selective_metrics['consistency_scores']).mean():.3f}")

        print()

    # Aggregate results
    print(f"{'='*70}")
    print(f"RESULTS (averaged over {num_runs} runs)")
    print(f"{'='*70}\n")

    # Vanilla RESU
    vanilla_final_losses = [r["final_loss"] for r in results["vanilla"]]
    vanilla_loss_stds = [r["loss_std"] for r in results["vanilla"]]

    print(f"Vanilla RESU:")
    print(f"  Final loss:       {torch.tensor(vanilla_final_losses).mean():.4f} ± {torch.tensor(vanilla_final_losses).std():.4f}")
    print(f"  Training stability: {torch.tensor(vanilla_loss_stds).mean():.4f} (lower = more stable)")

    # RESU-Selective
    selective_final_losses = [r["final_loss"] for r in results["selective"]]
    selective_loss_stds = [r["loss_std"] for r in results["selective"]]
    all_consistency = [c for r in results["selective"] for c in r["consistency_scores"]]

    print(f"\nRESU-Selective:")
    print(f"  Final loss:       {torch.tensor(selective_final_losses).mean():.4f} ± {torch.tensor(selective_final_losses).std():.4f}")
    print(f"  Training stability: {torch.tensor(selective_loss_stds).mean():.4f}")
    print(f"  Avg consistency:   {torch.tensor(all_consistency).mean():.3f}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}\n")

    vanilla_mean = torch.tensor(vanilla_final_losses).mean()
    selective_mean = torch.tensor(selective_final_losses).mean()
    improvement = ((vanilla_mean - selective_mean) / vanilla_mean * 100).item()

    if improvement > 0:
        print(f"✓ RESU-Selective achieves {improvement:.1f}% better final loss")
    else:
        print(f"⚠ Vanilla RESU achieves {-improvement:.1f}% better final loss")

    vanilla_std_mean = torch.tensor(vanilla_loss_stds).mean()
    selective_std_mean = torch.tensor(selective_loss_stds).mean()
    stability_improvement = ((vanilla_std_mean - selective_std_mean) / vanilla_std_mean * 100).item()

    if stability_improvement > 0:
        print(f"✓ RESU-Selective is {stability_improvement:.1f}% more stable (lower variance)")
    else:
        print(f"⚠ Vanilla RESU is {-stability_improvement:.1f}% more stable")

    print(f"\n{'='*70}\n")

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU (will be slower).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Run benchmark
    results = benchmark_selective(
        in_features=128,
        out_features=64,
        sparsity=0.5,
        num_runs=3,
        device=device,
    )
