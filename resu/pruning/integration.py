"""
Integration layer for Wanda++ and DSNoT.

Wraps the existing prune.py implementations to work with RESU.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Import from Huy's existing implementation
from prune import (
    PruneConfig,
    get_loaders,
    prepare_calibration_input,
    find_layers_qwen2,
    run_wanda_plus,
    run_dsnot,
)

from ..core.mask import SparseMask
from ..modules.linear import RESULinear


@dataclass 
class WandaDSNoTConfig:
    """Config bridging RESU and prune.py"""
    nsamples: int = 128
    seed: int = 0
    dataset: str = "c4"
    sparsity: float = 0.5
    wanda_alpha: float = 100.0
    dsnot_gamma: float = 1.0
    dsnot_cycles: int = 50
    dsnot_update_eps: float = 0.01
    ro_iters: int = 5
    ro_lr: float = 3e-7
    ro_subset: int = 32
    
    def to_prune_config(self) -> PruneConfig:
        return PruneConfig(
            nsamples=self.nsamples,
            seed=self.seed,
            dataset=self.dataset,
            sparsity=self.sparsity,
            wanda_alpha=self.wanda_alpha,
            dsnot_gamma=self.dsnot_gamma,
            dsnot_cycles=self.dsnot_cycles,
            dsnot_update_eps=self.dsnot_update_eps,
            ro_iters=self.ro_iters,
            ro_lr=self.ro_lr,
            ro_subset=self.ro_subset,
        )


class WandaPlusPruner:
    """RESU-compatible wrapper for Wanda++.
    
    Bridges run_wanda_plus() to RESU's pruning interface.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[WandaDSNoTConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or WandaDSNoTConfig()
        self.device = device or next(model.parameters()).device
        
        self._wanda_stats: Optional[Dict] = None
        self._calibrated = False
    
    @property
    def stats(self) -> Optional[Dict]:
        """Raw Wanda++ stats from last calibration."""
        return self._wanda_stats
    
    def calibrate(self, sparsity: Optional[float] = None):
        """Run Wanda++ calibration.
        
        Args:
            sparsity: Override config sparsity
        """
        cfg = self.config.to_prune_config()
        if sparsity is not None:
            cfg.sparsity = sparsity
        
        self._wanda_stats = run_wanda_plus(
            self.model,
            self.tokenizer,
            cfg,
            self.device,
        )
        self._calibrated = True
    
    def get_masks(self, sparsity: Optional[float] = None) -> Dict[int, Dict[str, SparseMask]]:
        """Get SparseMask objects from Wanda++ scores.
        
        Args:
            sparsity: Target sparsity (uses config if None)
            
        Returns:
            Dict[layer_idx, Dict[name, SparseMask]]
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate first")
        
        sparsity = sparsity or self.config.sparsity
        masks = {}
        
        for layer_idx, layer_stats in self._wanda_stats.items():
            masks[layer_idx] = {}
            for name, stats in layer_stats.items():
                score = stats["score"]
                mask = self._score_to_mask(score, sparsity)
                masks[layer_idx][name] = mask
        
        return masks
    
    def _score_to_mask(self, score: torch.Tensor, sparsity: float) -> SparseMask:
        """Convert scores to SparseMask (higher score = keep)."""
        flat = score.view(-1)
        k = int(sparsity * flat.numel())
        
        if k > 0:
            threshold = torch.kthvalue(flat, k).values
            mask_tensor = (score > threshold).float()
        else:
            mask_tensor = torch.ones_like(score)
        
        return SparseMask(mask_tensor)
    
    def apply_masks(self, masks: Dict[int, Dict[str, SparseMask]]):
        """Apply masks to model weights.
        
        Works with both nn.Linear and RESULinear.
        """
        layers = self.model.model.layers
        
        for layer_idx, layer_masks in masks.items():
            block = layers[layer_idx]
            subset = find_layers_qwen2(block)
            
            for name, sparse_mask in layer_masks.items():
                if name in subset:
                    mod = subset[name]
                    mask = sparse_mask.mask.to(mod.weight.device)
                    
                    if isinstance(mod, RESULinear):
                        mod.set_mask(sparse_mask)
                        mod.apply_mask()
                    else:
                        with torch.no_grad():
                            mod.weight.data *= mask
    
    def prune(self, sparsity: Optional[float] = None):
        """One-shot: calibrate + apply.
        
        Args:
            sparsity: Target sparsity
        """
        self.calibrate(sparsity)
        masks = self.get_masks(sparsity)
        self.apply_masks(masks)


class DSNoTStabilizer:
    """RESU-compatible wrapper for DSNoT.
    
    Bridges run_dsnot() to RESU's stabilization interface.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        wanda_stats: Dict,
        config: Optional[WandaDSNoTConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.wanda_stats = wanda_stats
        self.config = config or WandaDSNoTConfig()
        self.device = device or next(model.parameters()).device
        
        self._final_masks: Optional[Dict] = None
    
    @property
    def masks(self) -> Optional[Dict]:
        """Final masks after DSNoT refinement."""
        return self._final_masks
    
    def stabilize(self, sparsity: Optional[float] = None) -> Dict[int, Dict[str, SparseMask]]:
        """Run DSNoT stabilization.
        
        Args:
            sparsity: Target sparsity (uses config if None)
            
        Returns:
            Dict[layer_idx, Dict[name, SparseMask]]
        """
        cfg = self.config.to_prune_config()
        if sparsity is not None:
            cfg.sparsity = sparsity
        
        # Prepare calibration data
        dl, _ = get_loaders(
            cfg.dataset,
            cfg.nsamples,
            cfg.seed,
            self.model.seqlen,
            self.tokenizer,
        )
        
        with torch.no_grad():
            calib_inps, calib_outs, am, pos = prepare_calibration_input(
                self.model, dl, self.device
            )
        
        # Run DSNoT
        raw_masks = run_dsnot(
            self.model,
            self.wanda_stats,
            calib_inps,
            calib_outs,
            am,
            pos,
            cfg,
            self.device,
        )
        
        # Convert to SparseMask
        self._final_masks = {}
        for layer_idx, layer_masks in raw_masks.items():
            self._final_masks[layer_idx] = {}
            for name, mask_tensor in layer_masks.items():
                self._final_masks[layer_idx][name] = SparseMask(mask_tensor.float())
        
        return self._final_masks


def create_pruner_and_stabilizer(
    model: nn.Module,
    tokenizer,
    config: Optional[WandaDSNoTConfig] = None,
    device: Optional[torch.device] = None,
):
    """Factory to create both pruner and stabilizer.
    
    Returns:
        (pruner, stabilizer_factory)
        
    stabilizer_factory is a callable that creates DSNoTStabilizer
    after pruner.calibrate() has been called.
    """
    config = config or WandaDSNoTConfig()
    device = device or next(model.parameters()).device
    
    pruner = WandaPlusPruner(model, tokenizer, config, device)
    
    def make_stabilizer():
        if not pruner._calibrated:
            raise RuntimeError("Must call pruner.calibrate() first")
        return DSNoTStabilizer(
            model, tokenizer, pruner.stats, config, device
        )
    
    return pruner, make_stabilizer
