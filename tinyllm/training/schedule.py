"""Learning-rate scheduling: linear warmup followed by cosine decay."""

from __future__ import annotations

import math
from typing import Any, Protocol

import torch

from tinyllm.config import TrainConfig


class LRScheduler(Protocol):
    """Sets the optimizer's learning rate once per optimizer step."""

    @property
    def last_lr(self) -> float: ...

    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def warmup_cosine_lr(
    step: int, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int
) -> float:
    """LR to use for the optimizer update number ``step`` (0-based).

    Rises linearly to ``peak_lr`` over ``warmup_steps`` (reaching it on the last
    warmup step), then follows a half cosine down to ``min_lr`` at
    ``total_steps`` and stays there.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class WarmupCosineScheduler:
    """Applies :func:`warmup_cosine_lr` to every param group of an optimizer.

    The LR for the first update is set on construction; call :meth:`step`
    after each optimizer step to advance. Its only state is the step count, so
    resuming is exact.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
        self._optimizer = optimizer
        self._cfg = cfg
        self._step = 0
        self._apply()

    @property
    def last_lr(self) -> float:
        """LR currently set on the optimizer (used by the next update)."""
        return self._optimizer.param_groups[0]["lr"]

    def step(self) -> None:
        self._step += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = int(state["step"])
        self._apply()

    def _apply(self) -> None:
        lr = warmup_cosine_lr(
            self._step,
            self._cfg.learning_rate,
            self._cfg.min_learning_rate,
            self._cfg.warmup_steps,
            self._cfg.max_steps,
        )
        for group in self._optimizer.param_groups:
            group["lr"] = lr
