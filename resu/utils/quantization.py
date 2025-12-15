"""
Quantization utilities for QRESU.

Provides per-tensor and per-channel quantization/dequantization functions.
"""

import torch
from typing import Tuple, Literal


def quantize_per_tensor(
    W: torch.Tensor,
    bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize tensor with single scale/zero_point.

    Args:
        W: Weight tensor to quantize
        bits: Quantization bit-width (4 or 8)

    Returns:
        W_q: Quantized weights (uint8)
        scale: Scalar scale factor
        zero_point: Scalar zero point
    """
    qmin, qmax = 0, 2**bits - 1

    W_min = W.min()
    W_max = W.max()

    scale = (W_max - W_min) / (qmax - qmin)
    zero_point = qmin - W_min / scale

    W_q = ((W / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)

    return W_q, scale, zero_point


def dequantize_per_tensor(
    W_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-tensor quantized weights.

    Args:
        W_q: Quantized weights (uint8)
        scale: Scalar scale factor
        zero_point: Scalar zero point

    Returns:
        W: Dequantized weights (FP32)
    """
    return (W_q.float() - zero_point) * scale


def quantize_per_channel(
    W: torch.Tensor,
    bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize tensor per output channel (better quality).

    Args:
        W: Weight tensor (out_features, in_features)
        bits: Quantization bit-width (4 or 8)

    Returns:
        W_q: Quantized weights (uint8)
        scale: Per-channel scale factors (out_features,)
        zero_point: Per-channel zero points (out_features,)
    """
    qmin, qmax = 0, 2**bits - 1

    # Per-channel (along output dimension)
    W_min = W.min(dim=1, keepdim=True)[0]
    W_max = W.max(dim=1, keepdim=True)[0]

    scale = (W_max - W_min) / (qmax - qmin)
    zero_point = qmin - W_min / scale

    W_q = ((W / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)

    return W_q, scale.squeeze(1), zero_point.squeeze(1)


def dequantize_per_channel(
    W_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-channel quantized weights.

    Args:
        W_q: Quantized weights (uint8), shape (out_features, in_features)
        scale: Per-channel scale factors (out_features,)
        zero_point: Per-channel zero points (out_features,)

    Returns:
        W: Dequantized weights (FP32)
    """
    return (W_q.float() - zero_point.unsqueeze(1)) * scale.unsqueeze(1)


def quantize(
    W: torch.Tensor,
    bits: int = 4,
    scheme: Literal["per_tensor", "per_channel"] = "per_channel",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Quantize weights with specified scheme.

    Args:
        W: Weight tensor
        bits: Quantization bit-width (4 or 8)
        scheme: 'per_tensor' or 'per_channel'

    Returns:
        W_q: Quantized weights
        qparams: (scale, zero_point) tuple
    """
    if scheme == "per_tensor":
        W_q, scale, zero_point = quantize_per_tensor(W, bits)
    elif scheme == "per_channel":
        W_q, scale, zero_point = quantize_per_channel(W, bits)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return W_q, (scale, zero_point)


def dequantize(
    W_q: torch.Tensor,
    qparams: Tuple[torch.Tensor, ...],
    scheme: Literal["per_tensor", "per_channel"] = "per_channel",
) -> torch.Tensor:
    """Dequantize weights with specified scheme.

    Args:
        W_q: Quantized weights
        qparams: (scale, zero_point) tuple
        scheme: 'per_tensor' or 'per_channel'

    Returns:
        W: Dequantized weights (FP32)
    """
    scale, zero_point = qparams

    if scheme == "per_tensor":
        return dequantize_per_tensor(W_q, scale, zero_point)
    elif scheme == "per_channel":
        return dequantize_per_channel(W_q, scale, zero_point)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
