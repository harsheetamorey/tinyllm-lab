"""Validation: mean next-token loss and perplexity over a fixed set of batches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from tinyllm.model.model import TinyLLM
from tinyllm.training.loss import next_token_loss
from tinyllm.training.precision import MixedPrecision


@dataclass(frozen=True)
class EvalResult:
    loss: float
    perplexity: float
    num_tokens: int


class Evaluator(Protocol):
    """Scores a model on held-out data."""

    def evaluate(self, model: TinyLLM) -> EvalResult: ...


def perplexity(loss: float) -> float:
    """exp(loss), returning ``inf`` instead of overflowing."""
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


class LossEvaluator:
    """Evaluates on the same fixed batches every time, so results are comparable across steps."""

    def __init__(
        self, batches: Sequence[torch.Tensor], device: torch.device, precision: MixedPrecision
    ) -> None:
        if not batches:
            raise ValueError("need at least one validation batch")
        self._batches = batches
        self._device = device
        self._precision = precision

    @torch.no_grad()
    def evaluate(self, model: TinyLLM) -> EvalResult:
        """Token-weighted mean loss; the model's train/eval mode is restored afterwards."""
        was_training = model.training
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self._batches:
            batch = batch.to(self._device)
            with self._precision.autocast():
                logits = model(batch)
            num_targets = batch.shape[0] * (batch.shape[1] - 1)
            total_loss += next_token_loss(logits, batch).item() * num_targets
            total_tokens += num_targets
        model.train(was_training)
        mean_loss = total_loss / total_tokens
        return EvalResult(mean_loss, perplexity(mean_loss), total_tokens)
