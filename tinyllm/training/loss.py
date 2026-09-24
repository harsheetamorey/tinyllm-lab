"""Next-token language modeling loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # F.cross_entropy's default; used when there is no pad token


def shift_logits_and_labels(
    logits: torch.Tensor, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align predictions with their targets for next-token prediction.

    ``logits[:, t]`` is the model's prediction for token ``t + 1``, so:

        logits [B, T, V] -> logits[:, :-1]  [B, T-1, V]   (drop the last position)
        tokens [B, T]    -> tokens[:, 1:]   [B, T-1]      (drop the first token)
    """
    if logits.ndim != 3 or tokens.ndim != 2 or logits.shape[:2] != tokens.shape:
        raise ValueError(
            f"expected logits [B, T, V] and tokens [B, T], got "
            f"{tuple(logits.shape)} and {tuple(tokens.shape)}"
        )
    if tokens.shape[1] < 2:
        raise ValueError("need at least 2 tokens per sequence to predict a next token")
    return logits[:, :-1, :], tokens[:, 1:]


def next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, pad_id: int | None = None
) -> torch.Tensor:
    """Mean cross-entropy of predicting each token from the tokens before it.

    ``logits`` [B, T, V] come from running the model on ``tokens`` [B, T].
    Positions whose *target* equals ``pad_id`` are excluded from the mean; if
    every target is padding the loss is 0 (not NaN). Returns a scalar.
    """
    shifted_logits, targets = shift_logits_and_labels(logits, tokens)
    ignore_index = IGNORE_INDEX if pad_id is None else pad_id

    flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1]).float()  # [B*(T-1), V]
    flat_targets = targets.reshape(-1)  # [B*(T-1)]
    total = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum")
    num_valid = (flat_targets != ignore_index).sum().clamp(min=1)
    return total / num_valid
