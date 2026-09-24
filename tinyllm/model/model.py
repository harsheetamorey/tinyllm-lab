"""The complete decoder-only language model."""

from __future__ import annotations

import torch
from torch import nn

from tinyllm.config import ModelConfig
from tinyllm.model.block import TransformerBlock
from tinyllm.model.rmsnorm import RMSNorm


class TinyLLM(nn.Module):
    """Token embedding -> N pre-norm Transformer blocks -> final RMSNorm -> LM head.

    Shape flow (``D`` = hidden_size, ``V`` = vocab_size)::

        input_ids   [B, T]        long token ids
        embedding   [B, T, D]
        blocks x N  [B, T, D]
        final norm  [B, T, D]
        lm_head     [B, T, V]     logits (no softmax, no loss here)

    Position information comes from RoPE inside attention, so there is no
    positional embedding. The LM head has its own weights (not tied to the
    embedding).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock.from_config(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"expected input_ids of shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.cfg.max_sequence_length:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds "
                f"max_sequence_length {self.cfg.max_sequence_length}"
            )
        x = self.embed(input_ids)  # [B, T, D]
        for block in self.blocks:
            x = block(x)  # [B, T, D]
        return self.lm_head(self.final_norm(x))  # [B, T, V]

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Exact number of parameters (trainable ones by default)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
