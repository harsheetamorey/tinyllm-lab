"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyllm.config import ModelConfig
from tinyllm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where token ``t`` only sees tokens ``0..t``.

    Shape flow (``Dh = D / H``)::

        x                 [B, T, D]
        q, k, v (proj)    [B, T, D]   -> split heads -> [B, H, T, Dh]
        RoPE on q, k      [B, H, T, Dh]              (v is not rotated)
        scores  q @ k^T   [B, H, T, T]  / sqrt(Dh)
        causal mask + softmax           [B, H, T, T]
        probs @ v         [B, H, T, Dh]
        merge heads       [B, T, D]
        output projection [B, T, D]
    """

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # True where attention is allowed: position i may look at j <= i.
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "CausalSelfAttention":
        return cls(cfg.hidden_size, cfg.num_attention_heads, cfg.max_sequence_length)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``x`` [B, T, D] to Q, K, V, each split into heads: [B, H, T, Dh]."""
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected [B, T, {self.hidden_size}], got {tuple(x.shape)}")
        B, T, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        return split_heads(self.q_proj(x)), split_heads(self.k_proj(x)), split_heads(self.v_proj(x))

    def attention_probs(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Causal attention weights [B, H, T, T] from (already rotated) q, k [B, H, T, Dh]."""
        T = q.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, H, T, T]
        allowed = self.causal_mask[:T, :T]  # [T, T], broadcasts over B and H
        scores = scores.masked_fill(~allowed, float("-inf"))  # future -> -inf -> weight 0
        return F.softmax(scores.float(), dim=-1).to(q.dtype)  # softmax in fp32 for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project_qkv(x)  # also validates x's shape
        B, T, D = x.shape
        q, k = self.rope(q), self.rope(k)  # position enters through Q and K only
        probs = self.attention_probs(q, k)  # [B, H, T, T]
        out = probs @ v  # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(B, T, D)  # merge heads -> [B, T, D]
        return self.o_proj(out)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"
