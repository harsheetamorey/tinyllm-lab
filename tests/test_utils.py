"""Tests for Phase 0 utilities: config, reproducibility and run context."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
import yaml

from tinyllm.config import Config
from tinyllm.utils import (
    RunContext,
    capture_rng_state,
    detect_device,
    resolve_amp_dtype,
    restore_rng_state,
    set_seed,
)


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), torch.rand(1).item()


def test_set_seed_is_repeatable():
    set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first


def test_rng_state_roundtrip():
    set_seed(0)
    state = capture_rng_state()
    expected = _draw()
    _draw()  # advance further
    restore_rng_state(state)
    assert _draw() == expected


def test_detect_device():
    assert detect_device("cpu") == torch.device("cpu")
    assert detect_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        detect_device("tpu")


def test_resolve_amp_dtype():
    cpu = torch.device("cpu")
    assert resolve_amp_dtype("auto", cpu) is None
    assert resolve_amp_dtype("none", cpu) is None
    assert resolve_amp_dtype("bf16", cpu) is torch.bfloat16
    assert resolve_amp_dtype("fp16", cpu) is torch.float16
    with pytest.raises(ValueError):
        resolve_amp_dtype("fp8", cpu)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 7\nbogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        Config.from_yaml(path)


def test_run_context_writes_metadata(tmp_path):
    config = Config(experiment_id="t", seed=5, device="cpu", amp_dtype="none")
    ctx = RunContext.create(config, results_root=tmp_path)
    ctx.save_metadata()

    assert ctx.run_dir == tmp_path / "t"
    assert yaml.safe_load((ctx.run_dir / "config.yaml").read_text()) == config.to_dict()
    run = json.loads((ctx.run_dir / "run.json").read_text())
    assert run["device"] == "cpu" and run["amp_dtype"] is None
    assert "torch_version" in json.loads((ctx.run_dir / "environment.json").read_text())
    assert ctx.checkpoint_dir.is_dir()


def test_run_context_refuses_to_overwrite(tmp_path):
    config = Config(experiment_id="t", device="cpu")
    RunContext.create(config, results_root=tmp_path)
    with pytest.raises(FileExistsError):
        RunContext.create(config, results_root=tmp_path)
    RunContext.create(config, results_root=tmp_path, exist_ok=True)
