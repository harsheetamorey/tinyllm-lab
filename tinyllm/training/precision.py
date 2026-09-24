"""Mixed-precision policy: autocast plus loss scaling where fp16 needs it."""

from __future__ import annotations

import contextlib
from typing import Any, ContextManager

import torch


class MixedPrecision:
    """Autocast context and (for fp16 on CUDA) gradient scaling in one place.

    ``dtype=None`` means plain fp32. bf16 needs no loss scaling. fp16 gets a
    ``GradScaler`` on CUDA; on CPU/MPS, where scaling is unavailable, fp16
    falls back to fp32 (see :attr:`effective_dtype`).
    """

    def __init__(self, device: torch.device, dtype: torch.dtype | None) -> None:
        if dtype is torch.float16 and device.type != "cuda":
            dtype = None
        self._device_type = device.type
        self._dtype = dtype
        self._scaler = torch.amp.GradScaler("cuda") if dtype is torch.float16 else None

    @property
    def effective_dtype(self) -> torch.dtype | None:
        """Dtype autocast actually uses; ``None`` means fp32."""
        return self._dtype

    def autocast(self) -> ContextManager[Any]:
        if self._dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device_type, dtype=self._dtype)

    def backward(self, loss: torch.Tensor) -> None:
        (self._scaler.scale(loss) if self._scaler else loss).backward()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Bring grads back to true scale so they can be clipped/inspected."""
        if self._scaler:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer update (skipped by the scaler if grads overflowed)."""
        if self._scaler:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return self._scaler.state_dict() if self._scaler else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._scaler and state:
            self._scaler.load_state_dict(state)
