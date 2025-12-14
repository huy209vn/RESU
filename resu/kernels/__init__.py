"""Triton kernels for RESU operations."""
from .masked_ops import (
    masked_mul,
    inv_masked_mul,
    fused_effective_weight,
    split_gradient,
    extract_pruned_gradient,
    apply_mask_inplace,
    mask_where,
)
from .embedding import (
    phi_scatter,
    phi_inverse_gather,
    resu_update_indexed,
    resu_update_with_momentum,
    compute_pruned_indices,
    compute_active_indices,
    DenseResurrectionBuffer,
)
