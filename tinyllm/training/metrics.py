"""Structured metric logging: JSON lines on disk, human-readable lines on the console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

MetricRecord = dict[str, Any]


class MetricLogger(Protocol):
    """Sink for one flat record of metrics per call."""

    def log(self, record: MetricRecord) -> None: ...

    def close(self) -> None: ...


class JsonlMetricLogger:
    """Appends each record as one JSON line (appending lets a resumed run continue the file)."""

    def __init__(self, path: Path) -> None:
        self._file = open(path, "a", buffering=1)

    def log(self, record: MetricRecord) -> None:
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._file.close()


class ConsoleMetricLogger:
    """Prints a compact one-line summary of each record."""

    def __init__(self, emit: Callable[[str], None] = print) -> None:
        self._emit = emit

    def log(self, record: MetricRecord) -> None:
        parts = [f"step {record['step']:>6}", f"tokens {record['tokens_seen']:>10,}"]
        parts.append(f"loss {record['train_loss']:.4f}")
        if record.get("val_loss") is not None:
            parts.append(f"val {record['val_loss']:.4f} (ppl {record['val_ppl']:.2f})")
        parts.append(f"lr {record['lr']:.2e}")
        parts.append(f"gnorm {record['grad_norm']:.3f}")
        parts.append(f"{record['tokens_per_sec']:,.0f} tok/s")
        self._emit(" | ".join(parts))

    def close(self) -> None:
        pass


class MultiMetricLogger:
    """Fans each record out to several loggers."""

    def __init__(self, *loggers: MetricLogger) -> None:
        self._loggers = loggers

    def log(self, record: MetricRecord) -> None:
        for logger in self._loggers:
            logger.log(record)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()


def device_memory_mb(device: torch.device) -> float | None:
    """Memory currently allocated on an accelerator in MiB; ``None`` on CPU."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None
