"""AdamW construction with weight decay applied to weight matrices only."""

from __future__ import annotations

import torch
from torch import nn

from tinyllm.config import TrainConfig


def split_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (decay, no_decay) trainable parameters.

    Anything with fewer than 2 dimensions (RMSNorm gains, biases) is not decayed.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with two param groups (decayed / not decayed)."""
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )
