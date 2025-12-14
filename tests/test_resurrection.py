"""
Unit tests for ResurrectionEmbedding.
"""

import pytest
import torch
from resu.core.mask import SparseMask
from resu.core.resurrection import ResurrectionEmbedding, StorageMode


class TestResurrectionEmbedding:
    """Test ResurrectionEmbedding functionality."""

    def test_initialization(self, sparse_mask_50, device):
        """Test basic initialization."""
        embed = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)

        assert embed.p == sparse_mask_50.n_pruned
        assert embed.shape == sparse_mask_50.shape
        assert not embed._initialized

        # Initialize
        embed.initialize(active_std=1.0, epsilon=0.1)
        assert embed._initialized
        assert embed.theta.shape == (embed.p,)
        assert embed.theta.requires_grad

    def test_compact_mode_phi(self, resurrection_embedding, sparse_mask_50):
        """Test Φ operation in compact mode."""
        phi_theta = resurrection_embedding.phi()

        assert phi_theta.shape == resurrection_embedding.shape

        # Active positions should be 0
        active_vals = phi_theta.view(-1)[sparse_mask_50.active_indices]
        assert (active_vals == 0).all()

        # Pruned positions should match theta
        pruned_vals = phi_theta.view(-1)[sparse_mask_50.pruned_indices]
        assert torch.allclose(pruned_vals, resurrection_embedding.theta)

    def test_phi_inverse(self, resurrection_embedding, sparse_mask_50, device):
        """Test Φ⁻¹ operation."""
        matrix = torch.randn(resurrection_embedding.shape, device=device)

        gathered = resurrection_embedding.phi_inverse(matrix)
        expected = matrix.view(-1)[sparse_mask_50.pruned_indices]

        assert gathered.shape == (resurrection_embedding.p,)
        assert torch.allclose(gathered, expected)

    def test_phi_phi_inverse_round_trip(self, resurrection_embedding):
        """Test Φ and Φ⁻¹ are inverses."""
        theta = resurrection_embedding.theta.clone()

        # Φ(θ) → Φ⁻¹ should give back θ
        phi_theta = resurrection_embedding.phi()
        recovered = resurrection_embedding.phi_inverse(phi_theta)

        assert torch.allclose(recovered, theta)

    def test_effective_weights(self, resurrection_embedding, sparse_mask_50, device):
        """Test effective weight computation."""
        W = torch.randn(resurrection_embedding.shape, device=device)

        W_eff = resurrection_embedding.effective_weights(W, detach_active=False)

        assert W_eff.shape == W.shape

        # Active positions should have W
        W_active = W_eff.view(-1)[sparse_mask_50.active_indices]
        W_expected = W.view(-1)[sparse_mask_50.active_indices]
        assert torch.allclose(W_active, W_expected)

        # Pruned positions should have theta
        W_pruned = W_eff.view(-1)[sparse_mask_50.pruned_indices]
        assert torch.allclose(W_pruned, resurrection_embedding.theta)

    def test_sgd_update(self, resurrection_embedding, device):
        """Test SGD update."""
        grad_matrix = torch.randn(resurrection_embedding.shape, device=device)
        old_theta = resurrection_embedding.theta.clone()

        lr = 0.01
        resurrection_embedding.update_sgd(grad_matrix, lr)

        new_theta = resurrection_embedding.theta

        # Theta should have changed
        assert not torch.allclose(old_theta, new_theta)

        # Verify update direction
        grad_theta = resurrection_embedding.phi_inverse(grad_matrix)
        expected = old_theta - lr * grad_theta
        assert torch.allclose(new_theta, expected, atol=1e-5)

    def test_momentum_update(self, resurrection_embedding, device):
        """Test momentum update."""
        # Multiple updates
        old_theta = resurrection_embedding.theta.clone()

        for _ in range(5):
            grad_matrix = torch.randn(resurrection_embedding.shape, device=device)
            resurrection_embedding.update_momentum(grad_matrix, lr=0.01, beta=0.9)

        # Theta should have changed
        assert not torch.allclose(old_theta, resurrection_embedding.theta)
        assert resurrection_embedding._momentum is not None

    def test_adam_update(self, resurrection_embedding, device):
        """Test Adam update."""
        old_theta = resurrection_embedding.theta.clone()

        for step in range(10):
            grad_matrix = torch.randn(resurrection_embedding.shape, device=device)
            resurrection_embedding.update_adam(grad_matrix, lr=0.001)

        # Theta should have changed
        assert not torch.allclose(old_theta, resurrection_embedding.theta)
        assert resurrection_embedding._step_count == 10
        assert resurrection_embedding._momentum is not None
        assert resurrection_embedding._variance is not None

    def test_dense_mode(self, sparse_mask_50, device):
        """Test dense storage mode."""
        embed = ResurrectionEmbedding(sparse_mask_50, StorageMode.DENSE, device)
        embed.initialize(active_std=1.0, epsilon=0.1)

        assert embed._initialized

        # Dense buffer should exist
        assert embed._dense_buffer is not None

        # Phi should work
        phi = embed.phi()
        assert phi.shape == embed.shape

        # Compact form should work
        theta_compact = embed.theta
        assert theta_compact.shape == (embed.p,)

    def test_dense_compact_equivalence(self, sparse_mask_50, device):
        """Test that dense and compact modes give same results."""
        # Compact mode
        embed_compact = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
        embed_compact.initialize(active_std=1.0, epsilon=0.1, init_type="zero")

        # Dense mode
        embed_dense = ResurrectionEmbedding(sparse_mask_50, StorageMode.DENSE, device)
        embed_dense.initialize(active_std=1.0, epsilon=0.1, init_type="zero")

        # Set same theta values
        theta_init = torch.randn(embed_compact.p, device=device)
        embed_compact.theta = theta_init
        embed_dense.theta = theta_init

        # Phi should match
        phi_compact = embed_compact.phi()
        phi_dense = embed_dense.phi()
        assert torch.allclose(phi_compact, phi_dense)

        # Updates should match
        grad = torch.randn(embed_compact.shape, device=device)
        embed_compact.update_sgd(grad, lr=0.01)
        embed_dense.update_sgd(grad, lr=0.01)

        assert torch.allclose(embed_compact.theta, embed_dense.theta, atol=1e-5)

    def test_state_dict(self, resurrection_embedding, sparse_mask_50, device):
        """Test serialization."""
        # Do some updates
        for _ in range(5):
            grad = torch.randn(resurrection_embedding.shape, device=device)
            resurrection_embedding.update_adam(grad, lr=0.001)

        state = resurrection_embedding.state_dict()

        # Create new embedding and load
        embed2 = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
        embed2.initialize(active_std=1.0, epsilon=0.1)
        embed2.load_state_dict(state)

        assert torch.allclose(resurrection_embedding.theta, embed2.theta)
        assert resurrection_embedding._step_count == embed2._step_count

    def test_initialization_types(self, sparse_mask_50, device):
        """Test different initialization types."""
        # Normal
        embed_normal = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
        embed_normal.initialize(active_std=1.0, epsilon=0.1, init_type="normal")
        assert embed_normal.theta.abs().max() > 0

        # Uniform
        embed_uniform = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
        embed_uniform.initialize(active_std=1.0, epsilon=0.1, init_type="uniform")
        assert embed_uniform.theta.abs().max() > 0

        # Zero
        embed_zero = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
        embed_zero.initialize(active_std=1.0, epsilon=0.1, init_type="zero")
        assert (embed_zero.theta == 0).all()

    def test_gradient_flow(self, resurrection_embedding, device):
        """Test that gradients flow through phi."""
        W = torch.randn(resurrection_embedding.shape, device=device, requires_grad=True)

        # Create effective weights
        W_eff = resurrection_embedding.effective_weights(W, detach_active=False)

        # Compute loss and backward
        loss = W_eff.sum()
        loss.backward()

        # W should have gradients at active positions
        assert W.grad is not None

        # Resurrection theta should NOT have grad (we're not using PyTorch autograd for it)
        # (RESU uses manual gradient extraction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
