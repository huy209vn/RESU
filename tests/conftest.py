"""
PyTest configuration and fixtures for RESU tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value


@pytest.fixture
def small_shape():
    """Small shape for fast tests."""
    return (128, 64)


@pytest.fixture
def medium_shape():
    """Medium shape for standard tests."""
    return (512, 256)


@pytest.fixture
def large_shape():
    """Large shape for stress tests."""
    return (2048, 2048)


@pytest.fixture
def sparse_mask_50(device, small_shape, seed):
    """Create a 50% sparse mask."""
    from resu.core.mask import SparseMask
    mask_tensor = (torch.rand(small_shape, device=device) > 0.5).float()
    return SparseMask(mask_tensor)


@pytest.fixture
def sparse_mask_70(device, small_shape, seed):
    """Create a 70% sparse mask."""
    from resu.core.mask import SparseMask
    mask_tensor = (torch.rand(small_shape, device=device) > 0.7).float()
    return SparseMask(mask_tensor)


@pytest.fixture
def resurrection_embedding(sparse_mask_50, device):
    """Create a basic ResurrectionEmbedding."""
    from resu.core.resurrection import ResurrectionEmbedding, StorageMode
    embed = ResurrectionEmbedding(sparse_mask_50, StorageMode.COMPACT, device)
    embed.initialize(active_std=1.0, epsilon=0.1)
    return embed


@pytest.fixture
def resu_linear(device, seed):
    """Create a RESULinear layer."""
    from resu.modules.linear import RESULinear
    layer = RESULinear(128, 64, device=device)
    return layer


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
