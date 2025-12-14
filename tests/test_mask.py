"""
Unit tests for SparseMask.
"""

import pytest
import torch
from resu.core.mask import SparseMask, MaskStats


class TestSparseMask:
    """Test SparseMask functionality."""

    def test_basic_creation(self, device, small_shape, seed):
        """Test basic mask creation."""
        mask_tensor = (torch.rand(small_shape, device=device) > 0.5).float()
        mask = SparseMask(mask_tensor)

        assert mask.shape == small_shape
        assert mask.device.type == device.type
        assert mask.n_active + mask.n_pruned == mask_tensor.numel()
        assert 0 <= mask.sparsity <= 1

    def test_indices_correctness(self, device, small_shape, seed):
        """Test that precomputed indices are correct."""
        mask_tensor = (torch.rand(small_shape, device=device) > 0.3).float()
        mask = SparseMask(mask_tensor)

        # Verify active indices
        flat = mask_tensor.view(-1)
        for idx in mask.active_indices:
            assert flat[idx] == 1, f"Active index {idx} should be 1"

        # Verify pruned indices
        for idx in mask.pruned_indices:
            assert flat[idx] == 0, f"Pruned index {idx} should be 0"

        # Verify counts
        assert len(mask.active_indices) == mask.n_active
        assert len(mask.pruned_indices) == mask.n_pruned

    def test_apply_operations(self, device, small_shape, seed):
        """Test mask application operations."""
        mask_tensor = (torch.rand(small_shape, device=device) > 0.5).float()
        mask = SparseMask(mask_tensor)

        X = torch.randn(small_shape, device=device)

        # Test apply (M ⊙ X)
        masked = mask.apply(X)
        expected = mask_tensor * X
        assert torch.allclose(masked, expected)

        # Test apply_inverse ((1-M) ⊙ X)
        inv_masked = mask.apply_inverse(X)
        expected_inv = (1 - mask_tensor) * X
        assert torch.allclose(inv_masked, expected_inv)

        # Verify orthogonality
        assert torch.allclose(masked + inv_masked, X)

    def test_where_operation(self, device, small_shape, seed):
        """Test where operation."""
        mask_tensor = (torch.rand(small_shape, device=device) > 0.5).float()
        mask = SparseMask(mask_tensor)

        X = torch.randn(small_shape, device=device)
        Y = torch.randn(small_shape, device=device)

        result = mask.where(X, Y)
        expected = torch.where(mask_tensor.bool(), X, Y)

        assert torch.allclose(result, expected)

    def test_from_magnitude(self, device, small_shape, seed):
        """Test magnitude-based pruning."""
        weights = torch.randn(small_shape, device=device)
        sparsity = 0.5

        mask = SparseMask.from_magnitude(weights, sparsity)

        assert abs(mask.sparsity - sparsity) < 0.01

        # Verify small weights are pruned
        pruned_weights = weights.view(-1)[mask.pruned_indices]
        active_weights = weights.view(-1)[mask.active_indices]

        if len(pruned_weights) > 0 and len(active_weights) > 0:
            max_pruned = pruned_weights.abs().max()
            min_active = active_weights.abs().min()
            assert max_pruned <= min_active or torch.isclose(max_pruned, min_active)

    def test_random_mask(self, device, small_shape, seed):
        """Test random mask creation."""
        sparsity = 0.7
        mask = SparseMask.from_random(small_shape, sparsity, device)

        assert mask.shape == small_shape
        assert abs(mask.sparsity - sparsity) < 0.1  # Allow 10% tolerance for randomness

    def test_ones_zeros(self, device, small_shape):
        """Test all-ones and all-zeros masks."""
        ones = SparseMask.ones(small_shape, device)
        assert ones.sparsity == 0.0
        assert ones.n_active == small_shape[0] * small_shape[1]
        assert ones.n_pruned == 0

        zeros = SparseMask.zeros(small_shape, device)
        assert zeros.sparsity == 1.0
        assert zeros.n_active == 0
        assert zeros.n_pruned == small_shape[0] * small_shape[1]

    def test_overlap(self, device, small_shape, seed):
        """Test mask overlap computation."""
        mask1 = SparseMask.from_random(small_shape, 0.5, device)
        mask2 = SparseMask.from_random(small_shape, 0.5, device)

        both_active, both_pruned, different = mask1.overlap_with(mask2)

        total = small_shape[0] * small_shape[1]
        assert both_active + both_pruned + different == total
        assert both_active >= 0
        assert both_pruned >= 0
        assert different >= 0

    def test_jaccard_similarity(self, device, small_shape, seed):
        """Test Jaccard similarity."""
        mask = SparseMask.from_random(small_shape, 0.5, device)

        # Same mask should have similarity 1.0
        assert mask.jaccard_similarity(mask) == 1.0

        # Different masks should have similarity between 0 and 1
        mask2 = SparseMask.from_random(small_shape, 0.5, device)
        sim = mask.jaccard_similarity(mask2)
        assert 0 <= sim <= 1

    def test_update(self, device, small_shape, seed):
        """Test mask update."""
        mask = SparseMask.from_random(small_shape, 0.5, device)
        old_sparsity = mask.sparsity

        new_tensor = (torch.rand(small_shape, device=device) > 0.7).float()
        new_mask = mask.update(new_tensor)

        # Original should be unchanged
        assert mask.sparsity == old_sparsity

        # New mask should have new sparsity
        assert abs(new_mask.sparsity - 0.7) < 0.1

    def test_update_inplace(self, device, small_shape, seed):
        """Test in-place mask update."""
        mask = SparseMask.from_random(small_shape, 0.5, device)

        new_tensor = (torch.rand(small_shape, device=device) > 0.3).float()
        mask.update_inplace(new_tensor)

        assert abs(mask.sparsity - 0.3) < 0.1

    def test_state_dict(self, device, small_shape, seed):
        """Test serialization."""
        mask = SparseMask.from_random(small_shape, 0.6, device)

        state = mask.state_dict()
        loaded = SparseMask.from_state_dict(state, device)

        assert torch.equal(mask.mask, loaded.mask)
        assert mask.n_active == loaded.n_active
        assert mask.n_pruned == loaded.n_pruned

    def test_to_device(self, device, small_shape, seed):
        """Test device transfer."""
        mask = SparseMask.from_random(small_shape, 0.5, torch.device("cpu"))

        if device.type == "cuda":
            mask_cuda = mask.to(device)
            assert mask_cuda.device.type == "cuda"
            assert torch.equal(mask.mask.cpu(), mask_cuda.mask.cpu())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
