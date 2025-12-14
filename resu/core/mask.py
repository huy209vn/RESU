"""
SparseMask: Represents the partition (A, P) induced by pruning.

A = active coordinates (mask = 1)
P = pruned coordinates (mask = 0)

Precomputes and caches indices for efficient Φ, Φ⁻¹ operations.
"""

import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class MaskStats:
    """Statistics about a sparse mask."""
    total: int
    n_active: int
    n_pruned: int
    sparsity: float
    
    def __repr__(self) -> str:
        return (f"MaskStats(total={self.total}, active={self.n_active}, "
                f"pruned={self.n_pruned}, sparsity={self.sparsity:.2%})")


class SparseMask:
    """Represents the (A, P) partition induced by pruning.
    
    Maintains:
    - Binary mask tensor (1=active, 0=pruned)
    - Precomputed flat indices for A and P
    - Sparsity statistics
    
    Thread-safe for reads after initialization.
    """
    
    def __init__(
        self, 
        mask: torch.Tensor,
        precompute_indices: bool = True,
    ):
        """
        Args:
            mask: Binary mask (1=active, 0=pruned). Can be bool or float.
            precompute_indices: Whether to precompute index tensors
        """
        # Normalize to float mask
        if mask.dtype == torch.bool:
            self._mask = mask.float()
        else:
            self._mask = mask.clone()
        
        self._shape = mask.shape
        self._device = mask.device
        self._dtype = mask.dtype
        
        # Index caches (lazily computed or precomputed)
        self._active_indices: Optional[torch.Tensor] = None
        self._pruned_indices: Optional[torch.Tensor] = None
        self._stats: Optional[MaskStats] = None
        
        if precompute_indices:
            self._compute_indices()
            self._compute_stats()
    
    def _compute_indices(self):
        """Precompute flat indices for active and pruned sets."""
        mask_flat = self._mask.view(-1)
        
        # Active: where mask == 1
        self._active_indices = torch.nonzero(mask_flat == 1, as_tuple=True)[0].to(torch.int64)
        
        # Pruned: where mask == 0
        self._pruned_indices = torch.nonzero(mask_flat == 0, as_tuple=True)[0].to(torch.int64)
    
    def _compute_stats(self):
        """Compute mask statistics."""
        total = self._mask.numel()
        n_active = int((self._mask == 1).sum().item())
        n_pruned = total - n_active
        sparsity = n_pruned / total if total > 0 else 0.0
        
        self._stats = MaskStats(
            total=total,
            n_active=n_active,
            n_pruned=n_pruned,
            sparsity=sparsity,
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def mask(self) -> torch.Tensor:
        """The binary mask tensor."""
        return self._mask
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the mask."""
        return self._shape
    
    @property
    def device(self) -> torch.device:
        """Device of the mask."""
        return self._device
    
    @property
    def active_indices(self) -> torch.Tensor:
        """Flat indices of active positions (A)."""
        if self._active_indices is None:
            self._compute_indices()
        return self._active_indices
    
    @property
    def pruned_indices(self) -> torch.Tensor:
        """Flat indices of pruned positions (P)."""
        if self._pruned_indices is None:
            self._compute_indices()
        return self._pruned_indices
    
    @property
    def n_active(self) -> int:
        """Number of active parameters |A|."""
        if self._stats is None:
            self._compute_stats()
        return self._stats.n_active
    
    @property
    def n_pruned(self) -> int:
        """Number of pruned parameters |P| = p."""
        if self._stats is None:
            self._compute_stats()
        return self._stats.n_pruned
    
    @property
    def sparsity(self) -> float:
        """Sparsity ratio |P| / (|A| + |P|)."""
        if self._stats is None:
            self._compute_stats()
        return self._stats.sparsity
    
    @property
    def stats(self) -> MaskStats:
        """Full statistics object."""
        if self._stats is None:
            self._compute_stats()
        return self._stats
    
    # =========================================================================
    # Mask Operations
    # =========================================================================
    
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mask: M ⊙ X (zero out pruned positions)."""
        return self._mask * tensor
    
    def apply_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse mask: (1-M) ⊙ X (zero out active positions)."""
        return (1 - self._mask) * tensor
    
    def where(self, active_vals: torch.Tensor, pruned_vals: torch.Tensor) -> torch.Tensor:
        """Select values: active_vals where M=1, pruned_vals where M=0."""
        return self._mask * active_vals + (1 - self._mask) * pruned_vals
    
    # =========================================================================
    # Update Operations
    # =========================================================================
    
    def update(self, new_mask: torch.Tensor) -> "SparseMask":
        """Create new SparseMask with updated mask.
        
        Returns new object (immutable pattern).
        """
        return SparseMask(new_mask, precompute_indices=True)
    
    def update_inplace(self, new_mask: torch.Tensor):
        """Update mask in-place (use with caution).
        
        Invalidates cached indices.
        """
        if new_mask.dtype == torch.bool:
            self._mask = new_mask.float()
        else:
            self._mask.copy_(new_mask)
        
        # Invalidate caches
        self._active_indices = None
        self._pruned_indices = None
        self._stats = None
        
        # Recompute
        self._compute_indices()
        self._compute_stats()
    
    # =========================================================================
    # I/O
    # =========================================================================
    
    def to(self, device: Union[str, torch.device]) -> "SparseMask":
        """Move to device."""
        if self._device == torch.device(device):
            return self
        
        new_mask = SparseMask.__new__(SparseMask)
        new_mask._mask = self._mask.to(device)
        new_mask._shape = self._shape
        new_mask._device = torch.device(device)
        new_mask._dtype = self._dtype
        new_mask._stats = self._stats  # Stats are device-independent
        
        # Move indices if they exist
        if self._active_indices is not None:
            new_mask._active_indices = self._active_indices.to(device)
            new_mask._pruned_indices = self._pruned_indices.to(device)
        else:
            new_mask._active_indices = None
            new_mask._pruned_indices = None
        
        return new_mask
    
    def state_dict(self) -> dict:
        """Serialize mask state."""
        return {
            "mask": self._mask.cpu(),
            "shape": self._shape,
        }
    
    @classmethod
    def from_state_dict(cls, state: dict, device: Union[str, torch.device] = "cpu") -> "SparseMask":
        """Deserialize mask state."""
        mask = state["mask"].to(device)
        return cls(mask, precompute_indices=True)
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_magnitude(
        cls,
        weights: torch.Tensor,
        sparsity: float,
        granularity: str = "element",
    ) -> "SparseMask":
        """Create mask by pruning smallest magnitude weights.
        
        Args:
            weights: Weight tensor to prune
            sparsity: Fraction to prune (0 = no pruning, 1 = all pruned)
            granularity: 'element', 'row', or 'column'
            
        Returns:
            SparseMask with smallest weights pruned
        """
        if granularity == "element":
            # Element-wise pruning
            flat = weights.view(-1).abs()
            k = int(sparsity * len(flat))
            if k == 0:
                mask = torch.ones_like(weights)
            else:
                threshold = torch.kthvalue(flat, k).values
                mask = (weights.abs() > threshold).float()
        
        elif granularity == "row":
            # Row-wise pruning (prune entire rows)
            row_norms = weights.abs().sum(dim=1)
            k = int(sparsity * len(row_norms))
            if k == 0:
                mask = torch.ones_like(weights)
            else:
                threshold = torch.kthvalue(row_norms, k).values
                row_mask = (row_norms > threshold).float()
                mask = row_mask.unsqueeze(1).expand_as(weights)
        
        elif granularity == "column":
            # Column-wise pruning
            col_norms = weights.abs().sum(dim=0)
            k = int(sparsity * len(col_norms))
            if k == 0:
                mask = torch.ones_like(weights)
            else:
                threshold = torch.kthvalue(col_norms, k).values
                col_mask = (col_norms > threshold).float()
                mask = col_mask.unsqueeze(0).expand_as(weights)
        
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        return cls(mask, precompute_indices=True)
    
    @classmethod
    def from_random(
        cls,
        shape: Tuple[int, ...],
        sparsity: float,
        device: Union[str, torch.device] = "cpu",
    ) -> "SparseMask":
        """Create random mask with given sparsity."""
        mask = (torch.rand(shape, device=device) >= sparsity).float()
        return cls(mask, precompute_indices=True)
    
    @classmethod
    def ones(
        cls,
        shape: Tuple[int, ...],
        device: Union[str, torch.device] = "cpu",
    ) -> "SparseMask":
        """Create all-active mask (no pruning)."""
        mask = torch.ones(shape, device=device)
        return cls(mask, precompute_indices=True)
    
    @classmethod
    def zeros(
        cls,
        shape: Tuple[int, ...],
        device: Union[str, torch.device] = "cpu",
    ) -> "SparseMask":
        """Create all-pruned mask (100% sparsity)."""
        mask = torch.zeros(shape, device=device)
        return cls(mask, precompute_indices=True)
    
    # =========================================================================
    # Comparison / Analysis
    # =========================================================================
    
    def overlap_with(self, other: "SparseMask") -> Tuple[int, int, int]:
        """Compute overlap with another mask.
        
        Returns:
            (both_active, both_pruned, different)
        """
        assert self._shape == other._shape
        
        both_active = int(((self._mask == 1) & (other._mask == 1)).sum().item())
        both_pruned = int(((self._mask == 0) & (other._mask == 0)).sum().item())
        different = self._mask.numel() - both_active - both_pruned
        
        return both_active, both_pruned, different
    
    def jaccard_similarity(self, other: "SparseMask") -> float:
        """Jaccard similarity of active sets."""
        both_active, _, _ = self.overlap_with(other)
        union = self.n_active + other.n_active - both_active
        return both_active / union if union > 0 else 1.0
    
    # =========================================================================
    # Repr
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"SparseMask(shape={list(self._shape)}, "
                f"sparsity={self.sparsity:.2%}, "
                f"active={self.n_active}, pruned={self.n_pruned})")


# =============================================================================
# Testing
# =============================================================================

def _test_sparse_mask():
    """Test SparseMask functionality."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on {device}...")
    
    # Test basic creation
    shape = (128, 64)
    mask_tensor = (torch.rand(shape, device=device) > 0.3).float()
    mask = SparseMask(mask_tensor)
    
    print(f"Created: {mask}")
    assert mask.n_active + mask.n_pruned == mask_tensor.numel()
    assert abs(mask.sparsity - 0.3) < 0.1  # Approximately 30% sparse
    
    # Test indices
    assert len(mask.active_indices) == mask.n_active
    assert len(mask.pruned_indices) == mask.n_pruned
    
    # Verify indices are correct
    flat = mask_tensor.view(-1)
    for idx in mask.active_indices[:10]:  # Check first 10
        assert flat[idx] == 1
    for idx in mask.pruned_indices[:10]:
        assert flat[idx] == 0
    
    # Test apply operations
    X = torch.randn(shape, device=device)
    
    masked = mask.apply(X)
    assert torch.allclose(masked, mask_tensor * X)
    
    inv_masked = mask.apply_inverse(X)
    assert torch.allclose(inv_masked, (1 - mask_tensor) * X)
    
    # Test where
    Y = torch.randn(shape, device=device)
    result = mask.where(X, Y)
    expected = torch.where(mask_tensor.bool(), X, Y)
    assert torch.allclose(result, expected)
    
    # Test factory methods
    mag_mask = SparseMask.from_magnitude(X, sparsity=0.5)
    assert abs(mag_mask.sparsity - 0.5) < 0.01
    
    rand_mask = SparseMask.from_random(shape, sparsity=0.7, device=device)
    assert abs(rand_mask.sparsity - 0.7) < 0.05
    
    ones_mask = SparseMask.ones(shape, device=device)
    assert ones_mask.sparsity == 0.0
    
    zeros_mask = SparseMask.zeros(shape, device=device)
    assert zeros_mask.sparsity == 1.0
    
    # Test overlap
    mask1 = SparseMask.from_random(shape, 0.5, device)
    mask2 = SparseMask.from_random(shape, 0.5, device)
    both_active, both_pruned, different = mask1.overlap_with(mask2)
    assert both_active + both_pruned + different == shape[0] * shape[1]
    
    # Test serialization
    state = mask.state_dict()
    loaded = SparseMask.from_state_dict(state, device=device)
    assert torch.equal(mask.mask, loaded.mask)
    
    # Test update
    new_tensor = (torch.rand(shape, device=device) > 0.6).float()
    new_mask = mask.update(new_tensor)
    assert abs(new_mask.sparsity - 0.6) < 0.1
    assert mask.sparsity != new_mask.sparsity  # Original unchanged
    
    print("✓ All SparseMask tests passed!")


if __name__ == "__main__":
    _test_sparse_mask()
