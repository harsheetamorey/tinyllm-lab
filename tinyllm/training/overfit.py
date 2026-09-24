"""Tiny-dataset overfit experiment: a sanity check for the model and training path.

If the model cannot memorize a few hundred sequences, something is broken
(label shift, causal mask, learning rate, data, gradients or architecture),
and full pretraining should not start. This is intentionally not the
production training loop: constant-ish LR, no validation, no checkpoints.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch

from tinyllm.data.tokenizer import Tokenizer
from tinyllm.eval.generation import greedy_generate
from tinyllm.model.model import TinyLLM
from tinyllm.training.loss import next_token_loss

# Pass criteria, fixed up front (not tuned after seeing results).
MAX_FINAL_TO_START_LOSS = 0.1  # final loss must be under 10% of the starting loss
MIN_TOKEN_MATCH_RATE = 0.8  # greedy generation must reproduce >= 80% of the true continuation
EVAL_BATCH_SIZE = 25  # small enough to fit alongside other apps in shared (MPS) memory


@dataclass(frozen=True)
class OverfitConfig:
    num_sequences: int = 200
    seq_len: int = 128
    batch_size: int = 20
    steps: int = 1500
    lr: float = 1e-3
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10
    prompt_len: int = 16
    gen_tokens: int = 48
    seed: int = 42


@dataclass(frozen=True)
class StepLog:
    step: int
    loss: float
    lr: float
    grad_norm: float
    tokens_per_sec: float


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def build_tiny_dataset(
    texts: Iterable[str], tokenizer: Tokenizer, num_sequences: int, seq_len: int
) -> torch.Tensor:
    """Tokenize texts in order into ``[num_sequences, seq_len]`` (BOS + story start).

    Stories with fewer than ``seq_len`` tokens are skipped, so no padding is needed.
    """
    rows: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True)
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
            if len(rows) == num_sequences:
                return torch.tensor(rows, dtype=torch.long)
    raise ValueError(f"only found {len(rows)} stories with >= {seq_len} tokens, need {num_sequences}")


def lr_at(step: int, cfg: OverfitConfig) -> float:
    """Linear warmup over ``warmup_steps`` (step is 0-based), then constant."""
    return cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))


def _batch_indices(n: int, batch_size: int, gen: torch.Generator) -> Iterator[torch.Tensor]:
    while True:  # reshuffle every epoch; the last incomplete batch is dropped
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - batch_size + 1, batch_size):
            yield perm[i : i + batch_size]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_loss(model: TinyLLM, data: torch.Tensor, device: torch.device, batch_size: int = EVAL_BATCH_SIZE) -> float:
    """Mean next-token loss over all of ``data`` (eval mode)."""
    was_training = model.training
    model.eval()
    losses = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)
        losses.append(next_token_loss(model(batch), batch).item() * len(batch))
    model.train(was_training)
    return sum(losses) / len(data)


@torch.no_grad()
def memorization_rate(
    model: TinyLLM, data: torch.Tensor, device: torch.device, prompt_len: int, gen_tokens: int
) -> dict[str, float]:
    """Give the model each sequence's first ``prompt_len`` tokens and greedily continue.

    ``token_match`` is the fraction of generated tokens equal to the real
    continuation; ``exact_match`` the fraction of sequences reproduced perfectly.
    """
    matches = []
    for i in range(0, len(data), EVAL_BATCH_SIZE):
        batch = data[i : i + EVAL_BATCH_SIZE].to(device)
        generated = greedy_generate(model, batch[:, :prompt_len], gen_tokens)
        truth = batch[:, prompt_len : prompt_len + gen_tokens]
        matches.append((generated[:, prompt_len:] == truth).cpu())
    match = torch.cat(matches)  # [N, gen_tokens]
    return {
        "token_match": match.float().mean().item(),
        "exact_match": match.all(dim=1).float().mean().item(),
    }


def snapshot_params(model: TinyLLM) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def fraction_of_params_changed(before: dict[str, torch.Tensor], model: TinyLLM) -> float:
    """Fraction of parameter tensors whose values differ from ``before``."""
    changed = [not torch.equal(before[n], p.detach().cpu()) for n, p in model.named_parameters()]
    return sum(changed) / len(changed)


def run_overfit(
    model: TinyLLM,
    data: torch.Tensor,
    cfg: OverfitConfig,
    device: torch.device,
    snapshot_steps: Iterable[int] = (),
    on_snapshot: Callable[[int, TinyLLM], None] | None = None,
) -> list[StepLog]:
    """Train ``model`` on ``data`` [N, T] with AdamW; log every ``cfg.log_every`` steps.

    ``on_snapshot(step, model)`` is called (model in eval mode) after each step in
    ``snapshot_steps``; step 0 means before any update.
    """
    snapshots = set(snapshot_steps)

    def snapshot(step: int) -> None:
        if step in snapshots and on_snapshot is not None:
            was_training = model.training
            model.eval()
            on_snapshot(step, model)
            model.train(was_training)

    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    batches = _batch_indices(len(data), cfg.batch_size, gen)
    tokens_per_step = cfg.batch_size * data.shape[1]

    logs: list[StepLog] = []
    snapshot(0)
    _sync(device)
    window_start, window_steps = time.perf_counter(), 0
    for step in range(1, cfg.steps + 1):
        lr = lr_at(step - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = data[next(batches)].to(device)
        loss = next_token_loss(model(batch), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # pre-clip norm
        optimizer.step()
        window_steps += 1

        if step == 1 or step % cfg.log_every == 0:
            loss_value, grad_value = loss.item(), grad_norm.item()  # syncs the device
            elapsed = time.perf_counter() - window_start
            logs.append(StepLog(step, loss_value, lr, grad_value, window_steps * tokens_per_step / elapsed))
            window_start, window_steps = time.perf_counter(), 0
        snapshot(step)
    return logs


def evaluate_checks(
    logs: list[StepLog],
    start_loss: float,
    final_loss: float,
    params_changed: float,
    match: dict[str, float],
) -> list[Check]:
    """Turn the run's numbers into pass/fail checks."""
    ratio = final_loss / start_loss
    return [
        Check("loss decreases substantially", ratio <= MAX_FINAL_TO_START_LOSS,
              f"final/start = {final_loss:.4f}/{start_loss:.4f} = {ratio:.4f} (need <= {MAX_FINAL_TO_START_LOSS})"),
        Check("loss stays finite", all(math.isfinite(l.loss) for l in logs) and math.isfinite(final_loss),
              f"{sum(math.isfinite(l.loss) for l in logs)}/{len(logs)} logged losses finite"),
        Check("gradients are finite", all(math.isfinite(l.grad_norm) for l in logs),
              f"max grad norm {max(l.grad_norm for l in logs):.3f}"),
        Check("parameters change", params_changed == 1.0,
              f"{params_changed:.0%} of parameter tensors changed"),
        Check("generation memorizes training text", match["token_match"] >= MIN_TOKEN_MATCH_RATE,
              f"token match {match['token_match']:.1%} (need >= {MIN_TOKEN_MATCH_RATE:.0%}), "
              f"exact sequences {match['exact_match']:.1%}"),
    ]
