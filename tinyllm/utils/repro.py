"""Reproducibility helpers: seeding, RNG state snapshots, device and AMP resolution."""

from __future__ import annotations

import os
import random
from typing import Any, TypedDict

import numpy as np
import torch

_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_AMP_CHOICES = ("auto", "bf16", "fp16", "none")


class RNGState(TypedDict):
    """Snapshot of every RNG a training run touches; safe to ``torch.save``."""

    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch: torch.Tensor
    cuda: list[torch.Tensor]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices).

    With ``deterministic=True`` also force deterministic kernels. This is slower
    and raises on ops that have no deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> RNGState:
    """Return the current state of all RNGs, for storing in a checkpoint."""
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(legacy=False),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore RNGs from a snapshot produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def detect_device(preference: str = "auto") -> torch.device:
    """Resolve a device name to a ``torch.device``.

    ``auto`` picks CUDA, then MPS, then CPU. An explicit choice that is not
    available raises instead of silently falling back.
    """
    if preference not in _DEVICE_CHOICES:
        raise ValueError(f"device must be one of {_DEVICE_CHOICES}, got {preference!r}")

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    if preference == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not cuda_ok:
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if preference == "mps" and not mps_ok:
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(preference)


def resolve_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    """Pick the autocast dtype for ``device``; ``None`` means run in fp32.

    ``auto`` uses bf16 where supported (CUDA with bf16, MPS), fp16 on older
    CUDA GPUs, and fp32 on CPU.
    """
    if requested not in _AMP_CHOICES:
        raise ValueError(f"amp_dtype must be one of {_AMP_CHOICES}, got {requested!r}")

    if requested == "none":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return None
