"""
Unit tests for RESU-Selective.
"""

import pytest
import torch
from resu.core.mask import SparseMask
from resu.core.resurrection import ResurrectionEmbedding, StorageMode
from resu.core.selective import (
    RESUSelective,
    SelectionConfig,
    select_coordinates,
    update_ema_and_consistency,
)


class TestRESUSelective:
    """Test RESU-Selective functionality."""

    def test_ema_update(self, device):
        """Test EMA update correctness."""
        from resu.core.selective import update_ema

        p = 128
        m = torch.zeros(p, device=device)
        v = torch.zeros(p, device=device)
        g = torch.randn(p, device=device)

        beta = 0.9
        update_ema(m, v, g, beta)

        # Check first step
        expected_m = (1 - beta) * g
        expected_v = (1 - beta) * g.abs()

        assert torch.allclose(m, expected_m, atol=1e-5)
        assert torch.allclose(v, expected_v, atol=1e-5)

    def test_consistency_computation(self, device):
        """Test consistency computation."""
        from resu.core.selective import compute_consistency

        p = 128
        m = torch.randn(p, device=device)
        v = torch.rand(p, device=device) + 0.1
        delta = 1e-8

        c = compute_consistency(m, v, delta)

        expected = m.abs() / (v + delta)
        assert torch.allclose(c, expected, atol=1e-5)
        assert (c >= 0).all()
        assert (c <= 1).all() or (c > 1).any()  # Can be > 1

    def test_fused_ema_consistency(self, device):
        """Test fused EMA+consistency computation."""
        p = 128
        m = torch.zeros(p, device=device)
        v = torch.zeros(p, device=device)
        g = torch.randn(p, device=device)

        beta = 0.9
        delta = 1e-8

        c = update_ema_and_consistency(m, v, g, beta, delta)

        # Verify EMA was updated
        expected_m = (1 - beta) * g
        expected_v = (1 - beta) * g.abs()
        assert torch.allclose(m, expected_m, atol=1e-5)
        assert torch.allclose(v, expected_v, atol=1e-5)

        # Verify consistency
        expected_c = expected_m.abs() / (expected_v + delta)
        assert torch.allclose(c, expected_c, atol=1e-5)

    def test_selection_algorithm(self, device):
        """Test coordinate selection."""
        p = 256
        config = SelectionConfig(
            tau_stable=0.3,
            k_screen_ratio=0.5,
            k_select_ratio=0.2,
        )

        grad_theta = torch.randn(p, device=device)

        # Create consistency with high values for first half
        consistency = torch.rand(p, device=device) * 0.2
        consistency[:p//2] = 0.8 + torch.rand(p//2, device=device) * 0.2

        result = select_coordinates(grad_theta, consistency, config)

        assert result.n_selected <= config.k_select(p) + 1
        assert result.n_selected >= 0
        assert result.mask.shape == (p,)
        assert (result.mask >= 0).all()
        assert (result.mask <= 1).all()

        # Most selected should be from high-consistency region
        selected_indices = torch.nonzero(result.mask.bool(), as_tuple=True)[0]
        if len(selected_indices) > 0:
            high_con_selected = (selected_indices < p//2).sum()
            assert high_con_selected > len(selected_indices) * 0.3  # At least 30% from high-con

    def test_resu_selective_step(self, resurrection_embedding, device):
        """Test full selective update step."""
        config = SelectionConfig(
            beta=0.9,
            tau_stable=0.5,
            k_screen_ratio=0.5,
            k_select_ratio=0.2,
        )

        selective = RESUSelective(
            resurrection_embedding,
            config=config,
            lr=0.01,
        )

        assert selective.step_count == 0

        grad_matrix = torch.randn(resurrection_embedding.shape, device=device)
        old_theta = resurrection_embedding.theta.clone()

        stats = selective.step(grad_matrix)

        assert selective.step_count == 1
        assert "n_selected" in stats
        assert "mean_consistency" in stats
        assert stats["n_selected"] <= resurrection_embedding.p

        # Theta should have changed (unless nothing selected)
        if stats["n_selected"] > 0:
            assert not torch.allclose(old_theta, resurrection_embedding.theta)

    def test_consistency_buildup(self, resurrection_embedding, device):
        """Test that consistency builds up over steps."""
        config = SelectionConfig(beta=0.9, tau_stable=0.5)
        selective = RESUSelective(resurrection_embedding, config, lr=0.01)

        # Generate consistent gradients for first half
        p = resurrection_embedding.p

        for step in range(20):
            grad_matrix = torch.randn(resurrection_embedding.shape, device=device)

            # Make pruned coordinates have consistent direction
            grad_flat = grad_matrix.view(-1)
            pruned_indices = resurrection_embedding.mask.pruned_indices
            grad_flat[pruned_indices[:p//2]] = grad_flat[pruned_indices[:p//2]].abs()

            stats = selective.step(grad_matrix)

            if step > 5:  # After some warmup
                # Check that consistency is building up
                mean_c = stats["mean_consistency"]
                assert mean_c > 0, f"Step {step}: consistency should be > 0"

        # After 20 steps with consistent gradients, consistency should be meaningful
        assert selective.consistency.mean() > 0.1

    def test_selection_quality(self, resurrection_embedding, device):
        """Test that selection picks high-gradient coordinates."""
        config = SelectionConfig(
            beta=0.9,
            tau_stable=0.3,
            k_screen_ratio=0.5,
            k_select_ratio=0.2,
        )
        selective = RESUSelective(resurrection_embedding, config, lr=0.01)

        # Warm up consistency
        for _ in range(10):
            grad = torch.randn(resurrection_embedding.shape, device=device)
            selective.step(grad)

        # Now test with known high-gradient region
        grad_matrix = torch.randn(resurrection_embedding.shape, device=device)
        pruned_indices = resurrection_embedding.mask.pruned_indices
        p = len(pruned_indices)

        # Make first quarter have very high gradients
        grad_matrix.view(-1)[pruned_indices[:p//4]] *= 10.0

        stats = selective.step(grad_matrix)

        # Check if selection captured high-gradient region
        selection = selective.last_selection
        if selection and selection.n_selected > 0:
            # Selection should have picked from high-gradient region
            selected_mask = selection.mask.bool()
            # At least some should be from the high-gradient quarter
            assert selected_mask[:p//4].sum() > 0

    def test_state_dict(self, resurrection_embedding):
        """Test selective state serialization."""
        config = SelectionConfig(beta=0.9)
        selective = RESUSelective(resurrection_embedding, config, lr=0.01)

        # Do some steps
        for _ in range(5):
            grad = torch.randn(resurrection_embedding.shape, device=resurrection_embedding.device)
            selective.step(grad)

        state = selective.state_dict()

        # Create new selective and load
        selective2 = RESUSelective(resurrection_embedding, config, lr=0.01)
        selective2.load_state_dict(state)

        assert selective.step_count == selective2.step_count
        assert torch.allclose(selective.m, selective2.m)
        assert torch.allclose(selective.v, selective2.v)
        assert torch.allclose(selective.consistency, selective2.consistency)

    def test_reset_state(self, resurrection_embedding):
        """Test state reset."""
        selective = RESUSelective(resurrection_embedding, lr=0.01)

        # Do some steps
        for _ in range(5):
            grad = torch.randn(resurrection_embedding.shape, device=resurrection_embedding.device)
            selective.step(grad)

        assert selective.step_count > 0
        assert selective.m.abs().sum() > 0

        # Reset
        selective.reset_state()

        assert selective.step_count == 0
        assert (selective.m == 0).all()
        assert (selective.v == 0).all()
        assert (selective.consistency == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
