"""Tests for the overfit sanity-check machinery (fast, on a tiny CPU model)."""

from __future__ import annotations

import math

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.eval.generation import greedy_generate
from tinyllm.model.model import TinyLLM
from tinyllm.training.overfit import (
    OverfitConfig,
    build_tiny_dataset,
    evaluate_checks,
    evaluate_loss,
    fraction_of_params_changed,
    lr_at,
    memorization_rate,
    run_overfit,
    snapshot_params,
)

CPU = torch.device("cpu")
CFG = ModelConfig(
    vocab_size=40, hidden_size=32, num_layers=2, num_attention_heads=4,
    ffn_hidden_size=96, max_sequence_length=32,
)
RUN = OverfitConfig(num_sequences=8, seq_len=24, batch_size=8, steps=300, lr=3e-3,
                    warmup_steps=10, log_every=10, prompt_len=6, gen_tokens=12)


def _data() -> torch.Tensor:
    return torch.randint(1, CFG.vocab_size, (RUN.num_sequences, RUN.seq_len),
                         generator=torch.Generator().manual_seed(0))


def _model() -> TinyLLM:
    torch.manual_seed(0)
    return TinyLLM(CFG)


def _train(run: OverfitConfig = RUN):
    model, data = _model(), _data()
    before = snapshot_params(model)
    start = evaluate_loss(model, data, CPU)
    logs = run_overfit(model, data, run, CPU)
    final = evaluate_loss(model, data, CPU)
    match = memorization_rate(model, data, CPU, run.prompt_len, run.gen_tokens)
    return model, before, start, final, logs, match


def test_tiny_model_overfits_and_all_checks_pass():
    model, before, start, final, logs, match = _train()
    assert final < 0.1 * start
    checks = evaluate_checks(logs, start, final, fraction_of_params_changed(before, model), match)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


def test_checks_fail_when_nothing_is_learned():
    frozen = OverfitConfig(**{**RUN.__dict__, "lr": 0.0})
    model, before, start, final, logs, match = _train(frozen)
    checks = {c.name: c for c in evaluate_checks(
        logs, start, final, fraction_of_params_changed(before, model), match)}
    assert not checks["loss decreases substantially"].passed
    assert not checks["parameters change"].passed
    assert not checks["generation memorizes training text"].passed


def test_log_contents():
    _, _, _, _, logs, _ = _train()
    assert [l.step for l in logs][:2] == [1, 10] and logs[-1].step == RUN.steps
    assert all(math.isfinite(l.loss) and math.isfinite(l.grad_norm) and l.tokens_per_sec > 0 for l in logs)
    assert logs[0].lr == pytest.approx(RUN.lr / RUN.warmup_steps)
    assert logs[-1].lr == RUN.lr
    assert logs[-1].loss < logs[0].loss


def test_lr_schedule_warms_up_then_holds():
    assert lr_at(0, RUN) == pytest.approx(RUN.lr / RUN.warmup_steps)
    assert lr_at(RUN.warmup_steps - 1, RUN) == RUN.lr
    assert lr_at(10_000, RUN) == RUN.lr


def test_snapshots_called_at_requested_steps_in_eval_mode():
    seen = []
    model = _model()
    short = OverfitConfig(**{**RUN.__dict__, "steps": 6, "log_every": 3})
    run_overfit(model, _data(), short, CPU, snapshot_steps=[0, 3, 6],
                on_snapshot=lambda step, m: seen.append((step, m.training)))
    assert seen == [(0, False), (3, False), (6, False)]
    assert model.training  # restored after training


def test_run_is_reproducible():
    a = _train()[2:4]
    b = _train()[2:4]
    assert a == b


def test_greedy_generate_shape_prompt_and_determinism():
    model = _model().eval()
    prompt = _data()[:3, :5]
    out = greedy_generate(model, prompt, 7)
    assert out.shape == (3, 12) and torch.equal(out[:, :5], prompt)
    assert torch.equal(out, greedy_generate(model, prompt, 7))
    # Each new token is the argmax of the logits for the text before it.
    with torch.no_grad():
        assert torch.equal(out[:, 5], model(prompt)[:, -1].argmax(-1))


def test_greedy_generate_restores_training_mode_and_slides_past_max_length():
    model = _model().train()
    prompt = torch.randint(1, CFG.vocab_size, (2, 30))
    out = greedy_generate(model, prompt, 6)  # context grows past max_sequence_length=32
    assert out.shape == (2, 36) and model.training


def test_build_tiny_dataset_skips_short_stories_and_adds_bos():
    tok = BPETokenizer.train(["hello world, this is a story about a cat"] * 20, vocab_size=300)
    texts = ["hi", "hello world, this is a story about a cat " * 3, "yo", "hello world, this is a story about a cat " * 4]
    data = build_tiny_dataset(texts, tok, num_sequences=2, seq_len=10)
    assert data.shape == (2, 10) and (data[:, 0] == tok.bos_id).all()
    with pytest.raises(ValueError, match="only found"):
        build_tiny_dataset(texts, tok, num_sequences=3, seq_len=10)
