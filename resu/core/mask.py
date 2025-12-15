"""
Minimal SparseMask: Only stores pruned indices.

Matches the paper's storage model: no additional memory beyond dense baseline.
"""

import torch
from typing import Optional, Tuple
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
    """Minimal sparse mask with int32 indices and adaptive storage.

    Storage: O(min(p, a)) where p = pruned, a = active parameters
    Uses int32 for 50% memory savings vs int64.

    The paper expects RESU to add NO memory beyond dense storage.
    This implementation achieves that by:
    1. Using int32 indices (4 bytes vs 8 bytes)
    2. Storing whichever is smaller: active or pruned indices
    3. Computing everything else on-demand
    """

    def __init__(
        self,
        pruned_indices: torch.Tensor,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        adaptive: bool = True,
    ):
        """
        Args:
            pruned_indices: Flat indices of pruned positions
            shape: Shape of the weight matrix
            device: Device (derived from pruned_indices if None)
            adaptive: If True, store whichever is smaller (active or pruned)
        """
        self._shape = shape
        self._device = device or pruned_indices.device

        # Compute stats once
        self._total = int(torch.prod(torch.tensor(shape)).item())
        n_pruned = len(pruned_indices)
        n_active = self._total - n_pruned
        self._n_pruned = n_pruned
        self._n_active = n_active
        self._sparsity = self._n_pruned / self._total if self._total > 0 else 0.0

        # ADAPTIVE STORAGE: Store whichever is smaller
        if adaptive and n_active < n_pruned:
            # Store ACTIVE indices (complement of pruned)
            all_indices = torch.arange(self._total, device=self._device, dtype=torch.int64)
            mask = torch.ones(self._total, dtype=torch.bool, device=self._device)
            mask[pruned_indices] = False
            active_indices = all_indices[mask]
            self._indices = active_indices.to(torch.int32)
            self._stores_active = True
        else:
            # Store PRUNED indices (default)
            self._indices = pruned_indices.to(torch.int32)
            self._stores_active = False

        # Lazy caches
        self._mask_cache: Optional[torch.Tensor] = None
        self._active_indices_cache: Optional[torch.Tensor] = None
        self._pruned_indices_cache: Optional[torch.Tensor] = None

    @classmethod
    def from_dense_mask(cls, mask: torch.Tensor) -> 'SparseMask':
        """Create from dense boolean/float mask.

        Args:
            mask: Dense mask (1=active, 0=pruned)

        Returns:
            SparseMask with only pruned indices stored
        """
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        pruned_indices = torch.nonzero(~mask_bool.view(-1), as_tuple=True)[0]
        return cls(pruned_indices, mask.shape, mask.device)

    # =========================================================================
    # Core Properties (Cheap)
    # =========================================================================

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the weight matrix."""
        return self._shape

    @property
    def device(self) -> torch.device:
        """Device."""
        return self._device

    @property
    def n_active(self) -> int:
        """Number of active parameters."""
        return self._n_active

    @property
    def n_pruned(self) -> int:
        """Number of pruned parameters."""
        return self._n_pruned

    @property
    def sparsity(self) -> float:
        """Sparsity ratio."""
        return self._sparsity

    @property
    def pruned_indices(self) -> torch.Tensor:
        """Flat indices of pruned positions.

        Returns int64 tensor for compatibility with indexing operations.
        If storing active indices, computes pruned as complement (cached).
        """
        if self._stores_active:
            # Compute pruned as complement of active (cached)
            if self._pruned_indices_cache is None:
                all_indices = torch.arange(self._total, device=self._device, dtype=torch.int64)
                mask = torch.ones(self._total, dtype=torch.bool, device=self._device)
                mask[self._indices.long()] = False
                self._pruned_indices_cache = all_indices[mask]
            return self._pruned_indices_cache
        else:
            # Direct access - convert int32 to int64 for indexing
            return self._indices.long()

    @property
    def stats(self) -> MaskStats:
        """Mask statistics."""
        return MaskStats(
            total=self._total,
            n_active=self._n_active,
            n_pruned=self._n_pruned,
            sparsity=self._sparsity,
        )

    # =========================================================================
    # Computed Properties (Expensive - use sparingly!)
    # =========================================================================

    @property
    def mask(self) -> torch.Tensor:
        """Dense boolean mask (EXPENSIVE - computed on-demand).

        Returns:
            Boolean tensor of shape self.shape (1=active, 0=pruned)

        Note: This creates a full dense tensor! Use sparingly.
        Prefer boolean indexing with pruned_indices when possible.
        """
        if self._mask_cache is None:
            # Create dense boolean mask
            if self._stores_active:
                # Start with all False, set active to True
                mask_flat = torch.zeros(self._total, dtype=torch.bool, device=self._device)
                mask_flat[self._indices.long()] = True
            else:
                # Start with all True, set pruned to False
                mask_flat = torch.ones(self._total, dtype=torch.bool, device=self._device)
                mask_flat[self._indices.long()] = False
            self._mask_cache = mask_flat.view(self._shape)
        return self._mask_cache

    @property
    def active_indices(self) -> torch.Tensor:
        """Flat indices of active positions.

        Returns int64 tensor for compatibility with indexing operations.
        If storing pruned indices, computes active as complement (cached).
        """
        if self._stores_active:
            # Direct access - convert int32 to int64 for indexing
            return self._indices.long()
        else:
            # Compute active as complement of pruned (cached)
            if self._active_indices_cache is None:
                all_indices = torch.arange(self._total, device=self._device, dtype=torch.int64)
                mask = torch.ones(self._total, dtype=torch.bool, device=self._device)
                mask[self._indices.long()] = False
                self._active_indices_cache = all_indices[mask]
            return self._active_indices_cache

    # =========================================================================
    # Operations (Optimized for minimal storage)
    # =========================================================================

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mask to tensor (zero out pruned positions).

        Args:
            tensor: Tensor to mask (must match self.shape)

        Returns:
            Masked tensor (pruned positions = 0)
        """
        result = tensor.clone()
        result.view(-1)[self.pruned_indices] = 0
        return result

    def apply_inplace(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mask in-place (zero out pruned positions).

        Args:
            tensor: Tensor to mask (must match self.shape)

        Returns:
            Same tensor (modified in-place)
        """
        tensor.view(-1)[self.pruned_indices] = 0
        return tensor

    def where(self, true_val: torch.Tensor, false_val: torch.Tensor) -> torch.Tensor:
        """Equivalent to torch.where(mask, true_val, false_val).

        Returns true_val at active positions, false_val at pruned.
        """
        result = true_val.clone()
        pruned = self.pruned_indices
        result.view(-1)[pruned] = false_val.view(-1)[pruned]
        return result

    def get_active(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract active elements.

        Args:
            tensor: Tensor to extract from

        Returns:
            1D tensor of active elements
        """
        mask_flat = torch.ones(self._total, dtype=torch.bool, device=self._device)
        mask_flat[self.pruned_indices] = False
        return tensor.view(-1)[mask_flat]

    def get_pruned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract pruned elements.

        Args:
            tensor: Tensor to extract from

        Returns:
            1D tensor of pruned elements
        """
        return tensor.view(-1)[self.pruned_indices]

    def update(self, new_mask: torch.Tensor, inplace: bool = True, adaptive: bool = True) -> 'SparseMask':
        """Update mask with new pruning pattern.

        Args:
            new_mask: New dense boolean mask
            inplace: If True, update this object. Otherwise create new.
            adaptive: If True, use adaptive storage for new mask

        Returns:
            Updated mask
        """
        new_pruned_indices = torch.nonzero(~new_mask.view(-1).bool(), as_tuple=True)[0]

        if inplace:
            # Re-initialize with new indices (will apply adaptive logic)
            n_pruned = len(new_pruned_indices)
            n_active = self._total - n_pruned
            self._n_pruned = n_pruned
            self._n_active = n_active
            self._sparsity = n_pruned / self._total

            # ADAPTIVE STORAGE: Store whichever is smaller
            if adaptive and n_active < n_pruned:
                # Store ACTIVE indices
                all_indices = torch.arange(self._total, device=self._device, dtype=torch.int64)
                mask = torch.ones(self._total, dtype=torch.bool, device=self._device)
                mask[new_pruned_indices] = False
                active_indices = all_indices[mask]
                self._indices = active_indices.to(torch.int32)
                self._stores_active = True
            else:
                # Store PRUNED indices
                self._indices = new_pruned_indices.to(torch.int32)
                self._stores_active = False

            # Clear caches
            self._mask_cache = None
            self._active_indices_cache = None
            self._pruned_indices_cache = None
            return self
        else:
            return SparseMask(new_pruned_indices, self._shape, self._device, adaptive=adaptive)

    def to(self, device: torch.device) -> 'SparseMask':
        """Move to device.

        Args:
            device: Target device

        Returns:
            New mask on target device
        """
        if device == self._device:
            return self
        return SparseMask(
            self.pruned_indices.to(device),
            self._shape,
            device,
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'indices': self._indices,
            'stores_active': self._stores_active,
            'shape': self._shape,
            'n_pruned': self._n_pruned,
            'n_active': self._n_active,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: Optional[torch.device] = None) -> 'SparseMask':
        """Load from state dict."""
        # Reconstruct pruned_indices from stored state
        indices = state['indices']
        stores_active = state['stores_active']
        shape = tuple(state['shape'])
        n_pruned = state['n_pruned']
        n_active = state['n_active']
        total = n_pruned + n_active

        if stores_active:
            # Convert active indices back to pruned indices
            all_indices = torch.arange(total, device=device or indices.device, dtype=torch.int64)
            mask = torch.ones(total, dtype=torch.bool, device=device or indices.device)
            mask[indices.long()] = False
            pruned_indices = all_indices[mask]
        else:
            pruned_indices = indices.long()

        return cls(
            pruned_indices=pruned_indices,
            shape=shape,
            device=device,
            adaptive=True,
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @staticmethod
    def from_magnitude(
        weights: torch.Tensor,
        sparsity: float,
    ) -> 'SparseMask':
        """Create mask by magnitude pruning.

        Args:
            weights: Weight tensor
            sparsity: Fraction to prune

        Returns:
            Mask with smallest magnitude weights pruned
        """
        flat = weights.view(-1)
        n_prune = int(sparsity * len(flat))

        if n_prune == 0:
            # No pruning
            return SparseMask(
                torch.tensor([], dtype=torch.int64, device=weights.device),
                weights.shape,
                weights.device,
            )

        # Get indices of smallest magnitudes
        _, pruned_indices = torch.topk(flat.abs(), n_prune, largest=False)

        return SparseMask(pruned_indices, weights.shape, weights.device)

    @staticmethod
    def random(
        shape: Tuple[int, ...],
        sparsity: float,
        device: Optional[torch.device] = None,
    ) -> 'SparseMask':
        """Create random mask.

        Args:
            shape: Weight shape
            sparsity: Fraction to prune
            device: Device

        Returns:
            Random mask
        """
        device = device or torch.device('cpu')
        total = int(torch.prod(torch.tensor(shape)).item())
        n_prune = int(sparsity * total)

        # Random sample of indices
        all_indices = torch.randperm(total, device=device)
        pruned_indices = all_indices[:n_prune]

        return SparseMask(pruned_indices, shape, device)

    def __repr__(self) -> str:
        storage_type = "active" if self._stores_active else "pruned"
        n_stored = len(self._indices)
        return (f"SparseMask(shape={self.shape}, "
                f"sparsity={self.sparsity:.2%}, "
                f"storing={storage_type}, "
                f"n_stored={n_stored})")
