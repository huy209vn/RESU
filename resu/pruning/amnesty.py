"""
Amnesty: Relative Tournament Pruning for RESU

The key insight: resurrected weights shouldn't compete directly with
established active weights. They need a protected pathway to prove
themselves.

r(c) = r_start - (r_start - r_end) · (c/C)

Resurrection budget decreases over cycles as the network stabilizes.

Algorithm:
1. Split weights into Active (A) and Resurrected (R) groups
2. Run separate tournaments: TopK within A, TopK within R
3. Merge winners into new active set
4. No direct competition between groups
"""

import torch
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from ..core.mask import SparseMask


@dataclass
class AmnestyConfig:
    """Configuration for Amnesty mechanism."""
    r_start: float = 0.10       # Initial resurrection budget (10%)
    r_end: float = 0.02         # Final resurrection budget (2%)
    total_cycles: int = 5       # Total number of cycles
    score_type: str = "magnitude"  # 'magnitude', 'gradient', 'wanda'


class AmnestyResult(NamedTuple):
    """Result of amnesty pruning."""
    new_mask: SparseMask        # Updated mask
    n_active_kept: int          # Active weights kept
    n_resurrected_kept: int     # Resurrected weights kept
    n_active_pruned: int        # Active weights pruned
    n_resurrected_pruned: int   # Resurrected weights that didn't make it
    resurrection_budget: float  # r(c) used


class Amnesty:
    """Amnesty mechanism for fair resurrection competition.
    
    Implements relative tournament pruning where:
    - Active weights compete among themselves
    - Resurrected weights compete among themselves
    - A protected budget ensures resurrections get a chance
    
    Example:
        amnesty = Amnesty(config)
        
        # After RESU phase, commit with amnesty
        result = amnesty.commit(
            W_eff=layer.effective_weight,
            old_mask=layer.mask,
            target_sparsity=0.5,
            cycle=2,
        )
        
        layer.set_mask(result.new_mask)
    """
    
    def __init__(self, config: Optional[AmnestyConfig] = None):
        self.config = config or AmnestyConfig()
    
    def resurrection_budget(self, cycle: int) -> float:
        """Compute resurrection budget r(c) for given cycle.
        
        Linear decay from r_start to r_end over total_cycles.
        
        Args:
            cycle: Current cycle (0-indexed)
            
        Returns:
            Resurrection budget as fraction of keep budget
        """
        c = min(cycle, self.config.total_cycles)
        C = self.config.total_cycles
        
        r = self.config.r_start - (self.config.r_start - self.config.r_end) * (c / C)
        return r
    
    def compute_scores(
        self,
        W_eff: torch.Tensor,
        score_type: Optional[str] = None,
        activation_norms: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance scores for weights.
        
        Args:
            W_eff: Effective weights (committed)
            score_type: Override config score type
            activation_norms: For Wanda scoring
            gradients: For gradient-based scoring
            
        Returns:
            Importance scores (same shape as W_eff)
        """
        score_type = score_type or self.config.score_type
        
        if score_type == "magnitude":
            return W_eff.abs()
        
        elif score_type == "gradient":
            if gradients is None:
                raise ValueError("Gradients required for gradient scoring")
            # Fisher-like: |W| * |grad|
            return W_eff.abs() * gradients.abs()
        
        elif score_type == "wanda":
            if activation_norms is None:
                raise ValueError("Activation norms required for Wanda scoring")
            # |W| * ||X||
            return W_eff.abs() * activation_norms.unsqueeze(0)
        
        else:
            raise ValueError(f"Unknown score type: {score_type}")
    
    def relative_tournament(
        self,
        scores: torch.Tensor,
        old_mask: SparseMask,
        target_sparsity: float,
        cycle: int,
    ) -> AmnestyResult:
        """Perform relative tournament pruning.
        
        Args:
            scores: Importance scores for all weights
            old_mask: Previous mask (defines A vs R groups)
            target_sparsity: Target sparsity after pruning
            cycle: Current cycle number
            
        Returns:
            AmnestyResult with new mask and statistics
        """
        device = scores.device
        flat_scores = scores.view(-1)
        flat_old_mask = old_mask.mask.view(-1)
        
        n_total = len(flat_scores)
        n_keep = int((1 - target_sparsity) * n_total)
        
        # Compute resurrection budget
        r = self.resurrection_budget(cycle)
        n_resurrection_slots = int(r * n_keep)
        n_active_slots = n_keep - n_resurrection_slots
        
        # Identify groups
        # Active: positions that were active in old mask (mask == 1)
        # Resurrected: positions that were pruned but now have θ values
        active_indices = old_mask.active_indices
        resurrected_indices = old_mask.pruned_indices
        
        # Get scores for each group
        active_scores = flat_scores[active_indices]
        resurrected_scores = flat_scores[resurrected_indices]
        
        # Tournament within Active group
        n_active_keep = min(n_active_slots, len(active_indices))
        if n_active_keep > 0 and len(active_scores) > 0:
            _, active_topk_local = torch.topk(active_scores, n_active_keep)
            active_winners = active_indices[active_topk_local]
        else:
            active_winners = torch.tensor([], dtype=torch.int64, device=device)
        
        # Tournament within Resurrected group
        n_resurrection_keep = min(n_resurrection_slots, len(resurrected_indices))
        if n_resurrection_keep > 0 and len(resurrected_scores) > 0:
            _, resurrection_topk_local = torch.topk(resurrected_scores, n_resurrection_keep)
            resurrection_winners = resurrected_indices[resurrection_topk_local]
        else:
            resurrection_winners = torch.tensor([], dtype=torch.int64, device=device)
        
        # If we have leftover slots, fill from the other group
        total_kept = len(active_winners) + len(resurrection_winners)
        if total_kept < n_keep:
            # Need more - take from whichever group has more capacity
            remaining = n_keep - total_kept
            
            # Try to get more from resurrected first (they need the chance)
            if len(resurrection_winners) < len(resurrected_indices):
                # Get indices not already selected
                resurrection_mask = torch.ones(len(resurrected_indices), dtype=torch.bool, device=device)
                if len(resurrection_winners) > 0:
                    local_selected = torch.searchsorted(
                        resurrected_indices, resurrection_winners
                    )
                    valid = local_selected < len(resurrected_indices)
                    resurrection_mask[local_selected[valid]] = False
                
                remaining_resurrection_scores = resurrected_scores.clone()
                remaining_resurrection_scores[~resurrection_mask] = float('-inf')
                
                n_extra = min(remaining, resurrection_mask.sum().item())
                if n_extra > 0:
                    _, extra_local = torch.topk(remaining_resurrection_scores, n_extra)
                    extra_winners = resurrected_indices[extra_local]
                    resurrection_winners = torch.cat([resurrection_winners, extra_winners])
                    remaining -= n_extra
            
            # If still need more, get from active
            if remaining > 0 and len(active_winners) < len(active_indices):
                active_mask = torch.ones(len(active_indices), dtype=torch.bool, device=device)
                if len(active_winners) > 0:
                    local_selected = torch.searchsorted(active_indices, active_winners)
                    valid = local_selected < len(active_indices)
                    active_mask[local_selected[valid]] = False
                
                remaining_active_scores = active_scores.clone()
                remaining_active_scores[~active_mask] = float('-inf')
                
                n_extra = min(remaining, active_mask.sum().item())
                if n_extra > 0:
                    _, extra_local = torch.topk(remaining_active_scores, n_extra)
                    extra_winners = active_indices[extra_local]
                    active_winners = torch.cat([active_winners, extra_winners])
        
        # Build new mask
        new_mask_flat = torch.zeros(n_total, device=device)
        if len(active_winners) > 0:
            new_mask_flat[active_winners] = 1.0
        if len(resurrection_winners) > 0:
            new_mask_flat[resurrection_winners] = 1.0
        
        new_mask = SparseMask(new_mask_flat.view(scores.shape))
        
        # Compute statistics
        n_active_kept = len(active_winners)
        n_resurrected_kept = len(resurrection_winners)
        n_active_pruned = len(active_indices) - n_active_kept
        n_resurrected_pruned = len(resurrected_indices) - n_resurrected_kept
        
        return AmnestyResult(
            new_mask=new_mask,
            n_active_kept=n_active_kept,
            n_resurrected_kept=n_resurrected_kept,
            n_active_pruned=n_active_pruned,
            n_resurrected_pruned=n_resurrected_pruned,
            resurrection_budget=r,
        )
    
    def commit_with_amnesty(
        self,
        W_eff: torch.Tensor,
        old_mask: SparseMask,
        target_sparsity: float,
        cycle: int,
        score_type: Optional[str] = None,
        activation_norms: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
    ) -> Tuple[SparseMask, AmnestyResult]:
        """Full amnesty commit: compute scores and run tournament.
        
        Args:
            W_eff: Effective weights after RESU
            old_mask: Mask before RESU (defines groups)
            target_sparsity: Target sparsity
            cycle: Current cycle
            score_type: Override score type
            activation_norms: For Wanda scoring
            gradients: For gradient scoring
            
        Returns:
            (new_mask, result)
        """
        scores = self.compute_scores(
            W_eff, score_type, activation_norms, gradients
        )
        result = self.relative_tournament(scores, old_mask, target_sparsity, cycle)
        return result.new_mask, result


# =============================================================================
# Utilities
# =============================================================================

def compute_resurrection_rate(
    old_mask: SparseMask,
    new_mask: SparseMask,
) -> float:
    """Compute fraction of pruned weights that got resurrected.
    
    Returns:
        Fraction of old pruned positions that are now active
    """
    old_pruned = old_mask.pruned_indices
    new_active = new_mask.active_indices
    
    # Count old pruned that are now active
    new_active_set = set(new_active.cpu().numpy())
    resurrected = sum(1 for idx in old_pruned.cpu().numpy() if idx in new_active_set)
    
    return resurrected / len(old_pruned) if len(old_pruned) > 0 else 0.0


def compute_mask_churn(
    old_mask: SparseMask,
    new_mask: SparseMask,
) -> Dict[str, float]:
    """Compute mask change statistics.
    
    Returns:
        Dict with change statistics
    """
    both_active, both_pruned, different = old_mask.overlap_with(new_mask)
    total = old_mask.mask.numel()
    
    return {
        "stable_active": both_active / total,
        "stable_pruned": both_pruned / total,
        "changed": different / total,
        "jaccard": old_mask.jaccard_similarity(new_mask),
    }


def visualize_amnesty_result(result: AmnestyResult) -> str:
    """Create text visualization of amnesty result."""
    total_kept = result.n_active_kept + result.n_resurrected_kept
    total_pruned = result.n_active_pruned + result.n_resurrected_pruned
    
    lines = [
        f"Amnesty Result (r={result.resurrection_budget:.2%}):",
        f"  Active:      {result.n_active_kept:>6} kept, {result.n_active_pruned:>6} pruned",
        f"  Resurrected: {result.n_resurrected_kept:>6} kept, {result.n_resurrected_pruned:>6} pruned",
        f"  Total:       {total_kept:>6} kept, {total_pruned:>6} pruned",
    ]
    
    if total_kept > 0:
        lines.append(f"  Resurrection success rate: {result.n_resurrected_kept / total_kept:.1%}")
    
    return "\n".join(lines)
