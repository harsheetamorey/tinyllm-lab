"""Warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import pytest
import torch

from tinyllm.config import TrainConfig
from tinyllm.training.schedule import WarmupCosineScheduler, warmup_cosine_lr

PEAK, MIN, WARMUP, TOTAL = 1e-3, 1e-4, 10, 110


def lr(step: int) -> float:
    return warmup_cosine_lr(step, PEAK, MIN, WARMUP, TOTAL)


def test_warmup_is_linear_and_reaches_peak():
    assert lr(0) == pytest.approx(PEAK / WARMUP)
    assert lr(4) == pytest.approx(PEAK * 5 / WARMUP)
    assert lr(WARMUP - 1) == pytest.approx(PEAK)
    warm = [lr(s) for s in range(WARMUP)]
    assert warm == sorted(warm)


def test_cosine_decays_monotonically_from_peak_to_min():
    assert lr(WARMUP) == pytest.approx(PEAK)
    decay = [lr(s) for s in range(WARMUP, TOTAL + 1)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert lr(TOTAL) == pytest.approx(MIN)
    midpoint = WARMUP + (TOTAL - WARMUP) // 2
    assert lr(midpoint) == pytest.approx((PEAK + MIN) / 2)
    assert lr(TOTAL + 50) == pytest.approx(MIN)  # stays at the floor


def test_no_warmup_starts_at_peak():
    assert warmup_cosine_lr(0, PEAK, MIN, 0, TOTAL) == pytest.approx(PEAK)


def _scheduler() -> tuple[torch.optim.Optimizer, WarmupCosineScheduler]:
    cfg = TrainConfig(max_steps=TOTAL, warmup_steps=WARMUP, learning_rate=PEAK, min_learning_rate=MIN)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=123.0)
    return opt, WarmupCosineScheduler(opt, cfg)


def test_scheduler_sets_optimizer_lr_each_step():
    opt, sched = _scheduler()
    assert opt.param_groups[0]["lr"] == pytest.approx(lr(0))
    for step in range(1, 30):
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(lr(step))
        assert sched.last_lr == pytest.approx(lr(step))


def test_scheduler_state_roundtrip_continues_from_saved_step():
    _, sched = _scheduler()
    for _ in range(37):
        sched.step()
    state = sched.state_dict()

    opt2, fresh = _scheduler()
    fresh.load_state_dict(state)
    assert fresh.last_lr == pytest.approx(lr(37))
    fresh.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr(38))
    assert not math.isnan(fresh.last_lr)


def test_train_config_rejects_bad_schedule():
    with pytest.raises(ValueError):
        TrainConfig(max_steps=10, warmup_steps=10)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=1e-4, min_learning_rate=1e-3)
