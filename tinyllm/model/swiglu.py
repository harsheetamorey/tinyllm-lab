"""SwiGLU feed-forward network (Shazeer, 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyllm.config import ModelConfig


class SwiGLU(nn.Module):
    """Gated feed-forward layer: ``down( silu(gate(x)) * up(x) )``.

    Shape flow (``F = ffn_hidden_size``)::

        x                     [B, T, D]
        gate_proj(x)          [B, T, F]
        up_proj(x)            [B, T, F]   (the "value" branch)
        silu(gate) * up       [B, T, F]
        down_proj             [B, T, D]
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or ffn_hidden_size <= 0:
            raise ValueError(
                f"sizes must be positive, got hidden_size={hidden_size}, ffn_hidden_size={ffn_hidden_size}"
            )
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "SwiGLU":
        return cls(cfg.hidden_size, cfg.ffn_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected last dimension {self.hidden_size}, got {x.shape[-1]}")
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, ffn_hidden_size={self.ffn_hidden_size}"
