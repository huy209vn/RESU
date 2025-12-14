"""
Unit tests for Amnesty mechanism.
"""

import pytest
import torch
from resu.core.mask import SparseMask
from resu.pruning.amnesty import Amnesty, AmnestyConfig


class TestAmnesty:
    """Test Amnesty mechanism."""

    def test_resurrection_budget_schedule(self):
        """Test resurrection budget r(c) schedule."""
        config = AmnestyConfig(
            r_start=0.10,
            r_end=0.02,
            total_cycles=5,
        )
        amnesty = Amnesty(config)

        # First cycle
        r0 = amnesty.resurrection_budget(0)
        assert r0 == 0.10

        # Last cycle
        r5 = amnesty.resurrection_budget(5)
        assert abs(r5 - 0.02) < 1e-9

        # Middle cycle
        r2 = amnesty.resurrection_budget(2)
        assert 0.02 < r2 < 0.10

        # Should be linear
        r1 = amnesty.resurrection_budget(1)
        r3 = amnesty.resurrection_budget(3)
        r4 = amnesty.resurrection_budget(4)

        # Check linear spacing
        assert r1 > r2 > r3 > r4 > r5

    def test_magnitude_scoring(self, device):
        """Test magnitude-based importance scoring."""
        config = AmnestyConfig(score_type="magnitude")
        amnesty = Amnesty(config)

        W = torch.randn(128, 64, device=device)
        scores = amnesty.compute_scores(W)

        assert scores.shape == W.shape
        assert torch.allclose(scores, W.abs())

    def test_gradient_scoring(self, device):
        """Test gradient-based importance scoring."""
        config = AmnestyConfig(score_type="gradient")
        amnesty = Amnesty(config)

        W = torch.randn(128, 64, device=device)
        gradients = torch.randn(128, 64, device=device)

        scores = amnesty.compute_scores(W, gradients=gradients)

        expected = W.abs() * gradients.abs()
        assert torch.allclose(scores, expected)

    def test_wanda_scoring(self, device):
        """Test Wanda importance scoring."""
        config = AmnestyConfig(score_type="wanda")
        amnesty = Amnesty(config)

        W = torch.randn(128, 64, device=device)
        activation_norms = torch.rand(64, device=device)

        scores = amnesty.compute_scores(W, activation_norms=activation_norms)

        expected = W.abs() * activation_norms.unsqueeze(0)
        assert torch.allclose(scores, expected)

    def test_relative_tournament_basic(self, device):
        """Test basic relative tournament."""
        amnesty = Amnesty(AmnestyConfig(r_start=0.2, r_end=0.1, total_cycles=5))

        # Create weights and old mask
        shape = (128, 64)
        W_eff = torch.randn(shape, device=device)

        # Old mask: 50% sparsity
        old_mask = SparseMask.from_magnitude(W_eff, sparsity=0.5)

        # Now all weights have values (active kept their values, pruned got θ)
        # We want to keep same sparsity
        scores = W_eff.abs()

        result = amnesty.relative_tournament(scores, old_mask, target_sparsity=0.5, cycle=0)

        # Check sparsity
        assert abs(result.new_mask.sparsity - 0.5) < 0.01

        # Check budget allocation
        total_kept = result.n_active_kept + result.n_resurrected_kept
        expected_resurrection = int(0.2 * total_kept)  # r(0) = 0.2

        assert abs(result.n_resurrected_kept - expected_resurrection) <= 1

    def test_resurrection_actually_happens(self, device):
        """Test that some pruned weights can get resurrected."""
        amnesty = Amnesty(AmnestyConfig(r_start=0.3, r_end=0.1, total_cycles=5))

        shape = (256, 128)
        W = torch.randn(shape, device=device)

        # Create initial mask (50% sparse)
        old_mask = SparseMask.from_magnitude(W, sparsity=0.5)

        # Simulate RESU: give high scores to some pruned weights
        W_eff = W.clone()
        pruned_indices = old_mask.pruned_indices

        # Make first quarter of pruned weights have very high magnitude
        n_high = len(pruned_indices) // 4
        W_eff.view(-1)[pruned_indices[:n_high]] = 10.0

        scores = W_eff.abs()
        result = amnesty.relative_tournament(scores, old_mask, target_sparsity=0.5, cycle=0)

        # At least some should be resurrected
        assert result.n_resurrected_kept > 0

        # Check that high-magnitude pruned weights were resurrected
        new_active = result.new_mask.active_indices
        new_active_set = set(new_active.cpu().numpy())

        resurrected_count = sum(1 for idx in pruned_indices[:n_high].cpu().numpy() if idx in new_active_set)
        assert resurrected_count > 0, "Some high-magnitude pruned weights should be resurrected"

    def test_active_weights_can_be_pruned(self, device):
        """Test that low-importance active weights get pruned."""
        amnesty = Amnesty(AmnestyConfig(r_start=0.2))

        shape = (256, 128)
        W = torch.randn(shape, device=device)

        old_mask = SparseMask.from_magnitude(W, sparsity=0.5)

        # Simulate scenario: active weights lose importance, pruned weights gain it
        W_eff = W.clone()

        # Make active weights small
        active_indices = old_mask.active_indices
        W_eff.view(-1)[active_indices] = torch.randn(len(active_indices), device=device) * 0.01

        # Make pruned weights large
        pruned_indices = old_mask.pruned_indices
        W_eff.view(-1)[pruned_indices] = torch.randn(len(pruned_indices), device=device) * 5.0

        scores = W_eff.abs()
        result = amnesty.relative_tournament(scores, old_mask, target_sparsity=0.5, cycle=0)

        # Some active weights should be pruned
        assert result.n_active_pruned > 0

        # Many pruned weights should be resurrected
        assert result.n_resurrected_kept > len(pruned_indices) * 0.1

    def test_different_sparsities(self, device):
        """Test tournament with changing sparsity."""
        amnesty = Amnesty(AmnestyConfig(r_start=0.15))

        shape = (128, 64)
        W_eff = torch.randn(shape, device=device)

        old_mask = SparseMask.from_magnitude(W_eff, sparsity=0.7)
        scores = W_eff.abs()

        # Decrease sparsity (densify)
        result_dense = amnesty.relative_tournament(scores, old_mask, target_sparsity=0.5, cycle=0)
        assert abs(result_dense.new_mask.sparsity - 0.5) < 0.01
        assert result_dense.n_active_kept + result_dense.n_resurrected_kept > old_mask.n_active

        # Increase sparsity (more aggressive pruning)
        result_sparse = amnesty.relative_tournament(scores, old_mask, target_sparsity=0.9, cycle=0)
        assert abs(result_sparse.new_mask.sparsity - 0.9) < 0.01
        assert result_sparse.n_active_kept + result_sparse.n_resurrected_kept < old_mask.n_active

    def test_commit_with_amnesty(self, device):
        """Test full commit_with_amnesty flow."""
        amnesty = Amnesty(AmnestyConfig(score_type="magnitude", r_start=0.2))

        shape = (256, 128)
        W_eff = torch.randn(shape, device=device)
        old_mask = SparseMask.from_magnitude(W_eff, sparsity=0.5)

        new_mask, result = amnesty.commit_with_amnesty(
            W_eff=W_eff,
            old_mask=old_mask,
            target_sparsity=0.5,
            cycle=2,
        )

        assert isinstance(new_mask, SparseMask)
        assert abs(new_mask.sparsity - 0.5) < 0.01
        assert result.n_resurrected_kept >= 0
        assert result.n_active_kept >= 0


class TestAmnestyUtilities:
    """Test amnesty utility functions."""

    def test_resurrection_rate(self, device):
        """Test resurrection rate computation."""
        from resu.pruning.amnesty import compute_resurrection_rate

        shape = (128, 64)
        old_mask = SparseMask.from_random(shape, 0.5, device)

        # New mask resurrects some
        new_mask_tensor = old_mask.mask.clone()
        pruned_indices = old_mask.pruned_indices

        # Resurrect 20% of pruned
        n_resurrect = len(pruned_indices) // 5
        new_mask_tensor.view(-1)[pruned_indices[:n_resurrect]] = 1.0

        new_mask = SparseMask(new_mask_tensor)

        rate = compute_resurrection_rate(old_mask, new_mask)
        assert 0.15 < rate < 0.25  # Should be around 20%

    def test_mask_churn(self, device):
        """Test mask change statistics."""
        from resu.pruning.amnesty import compute_mask_churn

        shape = (128, 64)
        mask1 = SparseMask.from_random(shape, 0.5, device)
        mask2 = SparseMask.from_random(shape, 0.5, device)

        churn = compute_mask_churn(mask1, mask2)

        assert "stable_active" in churn
        assert "stable_pruned" in churn
        assert "changed" in churn
        assert "jaccard" in churn

        # Should sum to 1
        total = churn["stable_active"] + churn["stable_pruned"] + churn["changed"]
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
