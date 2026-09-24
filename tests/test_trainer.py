"""Trainer: accumulation, clipping, checkpoints, resume, validation, logging (tiny CPU model)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tinyllm.config import Config, ModelConfig, TokenizerConfig, TrainConfig
from tinyllm.data.packed_dataset import PackedBlocks, ShuffledBatchSource, fixed_batches
from tinyllm.training.checkpoint import CheckpointManager
from tinyllm.training.evaluator import LossEvaluator, perplexity
from tinyllm.training.factory import build_trainer
from tinyllm.training.metrics import JsonlMetricLogger, MetricLogger
from tinyllm.training.precision import MixedPrecision
from tinyllm.training.trainer import Trainer
from tinyllm.utils import RunContext
from tinyllm.utils.repro import capture_rng_state, restore_rng_state

VOCAB = 50
MODEL = ModelConfig(vocab_size=VOCAB, hidden_size=32, num_layers=2, num_attention_heads=4,
                    ffn_hidden_size=96, max_sequence_length=16)


def make_config(name: str = "t", **train) -> Config:
    defaults = dict(max_steps=6, micro_batch_size=4, grad_accum_steps=2, learning_rate=3e-3,
                    min_learning_rate=3e-4, warmup_steps=2, log_every=2, eval_every=3,
                    eval_batches=2, checkpoint_every=3)
    defaults.update(train)
    return Config(experiment_id=name, device="cpu", amp_dtype="none", seed=7,
                  model=MODEL, tokenizer=TokenizerConfig(vocab_size=VOCAB),
                  train=TrainConfig(**defaults))


def token_stream(n_blocks: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (n_blocks * MODEL.max_sequence_length,), generator=gen)


class ListLogger:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        self.records.append(record)

    def close(self) -> None:
        pass


def make_trainer(tmp_path: Path, name: str = "t", logger: MetricLogger | None = None, **train) -> Trainer:
    ctx = RunContext.create(make_config(name, **train), results_root=tmp_path)
    return build_trainer(ctx, token_stream(64, 1), token_stream(16, 2), logger or ListLogger())


def params(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


# ---- gradient accumulation ------------------------------------------------

class CountingBatches:
    def __init__(self, inner) -> None:
        self._inner, self.drawn = inner, 0

    def next_batch(self):
        self.drawn += 1
        return self._inner.next_batch()

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state):
        self._inner.load_state_dict(state)


def test_optimizer_steps_only_every_accum_interval(tmp_path):
    trainer = make_trainer(tmp_path, grad_accum_steps=3)
    steps_taken = []
    real_step = trainer._optimizer.step
    trainer._optimizer.step = lambda *a, **k: (steps_taken.append(1), real_step(*a, **k))[1]
    batch = token_stream(4, 3).view(4, -1)

    stepped = [trainer.micro_step(batch) for _ in range(7)]

    assert stepped == [False, False, True, False, False, True, False]
    assert len(steps_taken) == 2 and trainer.step == 2
    assert trainer.tokens_seen == 7 * batch.numel()


def test_train_step_draws_accum_batches_per_optimizer_step(tmp_path):
    trainer = make_trainer(tmp_path, grad_accum_steps=4)
    trainer._batches = CountingBatches(trainer._batches)
    trainer.train_step()
    trainer.train_step()
    assert trainer._batches.drawn == 8 and trainer.step == 2


def test_accumulated_gradient_matches_one_big_batch(tmp_path):
    big = make_trainer(tmp_path, "big", grad_accum_steps=1, micro_batch_size=8, grad_clip=0)
    small = make_trainer(tmp_path, "small", grad_accum_steps=2, micro_batch_size=4, grad_clip=0)
    small.model.load_state_dict(big.model.state_dict())
    batch = token_stream(8, 9).view(8, -1)

    big.micro_step(batch)
    small.micro_step(batch[:4])
    small.micro_step(batch[4:])

    for name, p in big.model.state_dict().items():
        torch.testing.assert_close(p, small.model.state_dict()[name], atol=1e-6, rtol=1e-4)


# ---- gradient clipping ----------------------------------------------------

def test_gradients_are_clipped_and_pre_clip_norm_is_reported(tmp_path):
    logger = ListLogger()
    trainer = make_trainer(tmp_path, logger=logger, grad_accum_steps=1, grad_clip=1e-3,
                           max_steps=2, warmup_steps=1, log_every=1, eval_every=2, checkpoint_every=2)
    seen = []
    real_step = trainer._optimizer.step

    def spy(*args, **kwargs):
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        seen.append(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])).item())
        return real_step(*args, **kwargs)

    trainer._optimizer.step = spy
    trainer.fit()

    assert all(norm <= 1e-3 * 1.001 for norm in seen)
    assert all(r["grad_norm"] > 1e-3 for r in logger.records)  # reported norm is before clipping


# ---- checkpoint / resume --------------------------------------------------

def test_checkpoint_restores_model_optimizer_scheduler_and_counters(tmp_path):
    a = make_trainer(tmp_path, "a")
    for _ in range(3):
        a.train_step()
    path = a.save_checkpoint()

    b = make_trainer(tmp_path, "b")
    assert any(not torch.equal(x, y) for x, y in zip(params(a).values(), params(b).values()))
    b.resume(path)

    for name, value in params(a).items():
        assert torch.equal(value, b.model.state_dict()[name])
    assert (b.step, b.tokens_seen) == (a.step, a.tokens_seen)
    assert b._scheduler.state_dict() == a._scheduler.state_dict()
    assert b._scheduler.last_lr == a._scheduler.last_lr
    state_a, state_b = a._optimizer.state_dict(), b._optimizer.state_dict()
    assert state_a["param_groups"] == state_b["param_groups"]
    for key, slot in state_a["state"].items():
        for field, value in slot.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(state_b["state"][key][field]))
    assert a._batches.state_dict() == b._batches.state_dict()


def test_checkpoint_contains_required_fields(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train_step()
    ckpt = CheckpointManager.load(trainer.save_checkpoint())
    assert ckpt.step == 1 and ckpt.tokens_seen == trainer.tokens_seen
    assert ckpt.config == make_config().to_dict()
    assert set(ckpt.rng) == {"python", "numpy", "torch", "cuda"}
    assert ckpt.model and ckpt.optimizer and ckpt.scheduler is not None


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    straight = make_trainer(tmp_path, "straight")
    straight.fit()

    first = make_trainer(tmp_path, "first", max_steps=6)
    for _ in range(3):
        first.train_step()
    path = first.save_checkpoint()

    resumed = make_trainer(tmp_path, "resumed")
    resumed.resume(path)
    resumed.fit()

    assert resumed.step == straight.step == 6 and resumed.tokens_seen == straight.tokens_seen
    for name, value in params(straight).items():
        assert torch.equal(value, resumed.model.state_dict()[name])


def test_resume_rejects_different_model_config(tmp_path):
    trainer = make_trainer(tmp_path, "a")
    trainer.train_step()
    path = trainer.save_checkpoint()
    other_model = ModelConfig(vocab_size=VOCAB, hidden_size=64, num_layers=2, num_attention_heads=4,
                              ffn_hidden_size=96, max_sequence_length=16)
    config = make_config("b")
    config.model = other_model
    ctx = RunContext.create(config, results_root=tmp_path)
    other = build_trainer(ctx, token_stream(64, 1), token_stream(16, 2), ListLogger())
    with pytest.raises(ValueError, match="model config"):
        other.resume(path)


def test_latest_checkpoint_is_highest_step(tmp_path):
    trainer = make_trainer(tmp_path, max_steps=6, checkpoint_every=3)
    trainer.fit()
    manager = CheckpointManager(tmp_path / "t" / "checkpoints")
    assert manager.latest().name == "step_00000006.pt"
    (tmp_path / "empty").mkdir()
    assert CheckpointManager(tmp_path / "empty").latest() is None


def test_rng_restore_reproduces_subsequent_random_values():
    import random
    import numpy as np
    state = capture_rng_state()
    first = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0] and first[1] == second[1] and torch.equal(first[2], second[2])


def test_resume_restores_rng_state(tmp_path):
    trainer = make_trainer(tmp_path, "a")
    trainer.train_step()
    path = trainer.save_checkpoint()
    expected = torch.rand(4)  # what the run would draw next
    torch.rand(100)  # perturb the global RNG

    other = make_trainer(tmp_path, "b")
    other.resume(path)
    assert torch.equal(torch.rand(4), expected)


# ---- data source ----------------------------------------------------------

def test_batch_source_position_restores_exactly():
    blocks = PackedBlocks(token_stream(10, 0), MODEL.max_sequence_length)
    a = ShuffledBatchSource(blocks, 4, seed=1)
    for _ in range(5):  # crosses an epoch boundary (10 blocks // 4 = 2 batches per epoch)
        a.next_batch()
    b = ShuffledBatchSource(blocks, 4, seed=1)
    b.load_state_dict(a.state_dict())
    for _ in range(4):
        assert torch.equal(a.next_batch(), b.next_batch())


# ---- validation -----------------------------------------------------------

def test_validation_loss_and_perplexity_are_finite(tmp_path):
    trainer = make_trainer(tmp_path)
    blocks = PackedBlocks(token_stream(16, 2), MODEL.max_sequence_length)
    evaluator = LossEvaluator(fixed_batches(blocks, 4, 3), torch.device("cpu"),
                              MixedPrecision(torch.device("cpu"), None))
    result = evaluator.evaluate(trainer.model)
    assert math.isfinite(result.loss) and math.isfinite(result.perplexity)
    assert result.perplexity == pytest.approx(math.exp(result.loss))
    assert result.num_tokens == 3 * 4 * (MODEL.max_sequence_length - 1)
    trainer.model.train()
    evaluator.evaluate(trainer.model)
    assert trainer.model.training  # training mode is restored after evaluation
    assert evaluator.evaluate(trainer.model).loss == result.loss  # fixed data -> repeatable


def test_perplexity_overflow_is_inf():
    assert perplexity(1e6) == math.inf


# ---- end-to-end logging ---------------------------------------------------

def test_fit_logs_required_metrics_and_writes_jsonl(tmp_path):
    ctx = RunContext.create(make_config("logged"), results_root=tmp_path)
    logger = JsonlMetricLogger(ctx.run_dir / "metrics.jsonl")
    trainer = build_trainer(ctx, token_stream(64, 1), token_stream(16, 2), logger)
    trainer.fit()

    rows = [json.loads(line) for line in (ctx.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 3, 4, 6]  # log_every=2 plus eval steps 3 and 6
    required = {"step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr", "grad_norm",
                "tokens_per_sec", "step_time_s", "memory_mb"}
    assert all(required <= set(r) for r in rows)
    assert [r["val_loss"] is not None for r in rows] == [False, True, False, True]
    assert all(math.isfinite(r["train_loss"]) and math.isfinite(r["grad_norm"]) for r in rows)
    assert rows[-1]["tokens_seen"] == 6 * 2 * 4 * MODEL.max_sequence_length
    assert sorted(p.name for p in ctx.checkpoint_dir.glob("*.pt")) == ["step_00000003.pt", "step_00000006.pt"]


def test_non_finite_loss_stops_training(tmp_path):
    trainer = make_trainer(tmp_path, learning_rate=1e30, min_learning_rate=1e30, warmup_steps=0,
                           grad_clip=0, log_every=1)
    with pytest.raises(FloatingPointError):
        trainer.fit()
