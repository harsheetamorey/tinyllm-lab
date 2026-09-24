"""Checkpoint container and on-disk manager."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tinyllm.utils.repro import RNGState

_PATTERN = "step_{step:08d}.pt"


@dataclass
class Checkpoint:
    """Everything needed to continue a run exactly where it stopped."""

    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    step: int
    tokens_seen: int
    config: dict[str, Any]
    rng: RNGState
    data: dict[str, Any]  # position of the training batch source


class CheckpointManager:
    """Saves ``step_XXXXXXXX.pt`` files in one directory and finds the newest."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def save(self, checkpoint: Checkpoint) -> Path:
        """Write atomically (temp file + rename) so a crash never leaves a torn checkpoint."""
        path = self._dir / _PATTERN.format(step=checkpoint.step)
        tmp = path.with_suffix(".tmp")
        torch.save(vars(checkpoint), tmp)
        os.replace(tmp, path)
        return path

    def latest(self) -> Path | None:
        found = sorted(self._dir.glob("step_*.pt"))
        return found[-1] if found else None

    @staticmethod
    def load(path: Path) -> Checkpoint:
        # Our own files, and the RNG state holds Python/NumPy objects, so full unpickling is needed.
        return Checkpoint(**torch.load(path, map_location="cpu", weights_only=False))
