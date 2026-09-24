"""Shared utilities: reproducibility, environment probing, run metadata."""

from tinyllm.utils.env import environment_info
from tinyllm.utils.repro import (
    capture_rng_state,
    detect_device,
    resolve_amp_dtype,
    restore_rng_state,
    set_seed,
)
from tinyllm.utils.run_context import RunContext

__all__ = [
    "environment_info",
    "capture_rng_state",
    "detect_device",
    "resolve_amp_dtype",
    "restore_rng_state",
    "set_seed",
    "RunContext",
]
