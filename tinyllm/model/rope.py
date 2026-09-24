"""Rotary positional embeddings (RoPE, Su et al., 2021)."""

from __future__ import annotations

import torch
from torch import nn

from tinyllm.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotate pairs of features of Q or K by an angle that depends on token position.

    Feature ``i`` is paired with feature ``i + head_dim/2``. The pair at position
    ``p`` is rotated by ``p * theta_i`` with ``theta_i = base^(-2i / head_dim)``.
    Apply it to queries and keys only, never values.

    The cos/sin tables for positions ``0 .. max_seq_len-1`` are computed once at
    construction and stored as non-persistent buffers (not saved in checkpoints,
    since they are a pure function of the arguments).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"head_dim must be a positive even number for RoPE, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # [Dh/2]
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)  # [T_max, Dh/2]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig, base: float = 10000.0) -> "RotaryEmbedding":
        """Build for a model: head size and max sequence length come from the config."""
        return cls(cfg.head_dim, cfg.max_sequence_length, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` of shape ``[B, H, T, head_dim]``; the shape is unchanged.

        Token ``t`` (0-based along T) is rotated as position ``t``.
        """
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError(f"expected [B, H, T, {self.head_dim}], got {tuple(x.shape)}")
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        cos = self.cos[:seq_len]  # [T, Dh/2], broadcasts over B and H
        sin = self.sin[:seq_len]
        x32 = x.float()  # rotate in float32 so bf16/fp16 keeps precision
        x1, x2 = x32[..., : self.head_dim // 2], x32[..., self.head_dim // 2 :]
        rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.to(x.dtype)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, base={self.base}"
