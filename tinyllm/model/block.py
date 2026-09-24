"""One pre-norm decoder Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from tinyllm.config import ModelConfig
from tinyllm.model.attention import CausalSelfAttention
from tinyllm.model.rmsnorm import RMSNorm
from tinyllm.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm decoder block built from the existing components::

        x = x + attention(attn_norm(x))
        x = x + ffn(ffn_norm(x))

    Input and output are both ``[B, T, D]``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, norm_eps)
        self.attn = CausalSelfAttention(hidden_size, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(hidden_size, norm_eps)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "TransformerBlock":
        return cls(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.ffn_hidden_size,
            cfg.max_sequence_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))  # [B, T, D] + [B, T, D]
        x = x + self.ffn(self.ffn_norm(x))  # [B, T, D] + [B, T, D]
        return x
