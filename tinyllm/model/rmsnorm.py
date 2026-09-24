"""Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize the last dimension by its root mean square, then scale.

    ``y = x / sqrt(mean(x^2, dim=-1) + eps) * weight``

    Input and output are ``[..., dim]`` (e.g. ``[B, T, D]``); each vector along
    ``D`` is normalized independently. Unlike LayerNorm there is no mean
    subtraction and no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable per-channel scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        # Normalize in float32 so bf16/fp16 inputs don't lose precision in x^2.
        x32 = x.float()
        inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * inv_rms).to(x.dtype) * self.weight.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
