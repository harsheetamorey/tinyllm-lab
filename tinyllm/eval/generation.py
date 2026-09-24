"""Minimal text generation (greedy decoding, no KV cache)."""

from __future__ import annotations

import torch

from tinyllm.model.model import TinyLLM


@torch.no_grad()
def greedy_generate(model: TinyLLM, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Extend each prompt by ``max_new_tokens`` most-likely tokens.

    ``prompt_ids`` is ``[B, T0]``; returns ``[B, T0 + max_new_tokens]`` (prompt
    included). Recomputes the whole context every step, and only the last
    ``max_sequence_length`` tokens are fed once the context grows past it.
    Restores the model's train/eval mode afterwards.
    """
    if prompt_ids.ndim != 2:
        raise ValueError(f"expected prompt_ids [B, T0], got {tuple(prompt_ids.shape)}")
    was_training = model.training
    model.eval()
    max_len = model.cfg.max_sequence_length
    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_len:])  # [B, T, V]
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B, 1]
        ids = torch.cat([ids, next_id], dim=1)
    model.train(was_training)
    return ids
