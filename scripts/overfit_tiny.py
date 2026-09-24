"""Overfit the debug model on ~200 TinyStories to sanity-check the training path.

Usage: python -m scripts.overfit_tiny

Writes to results/: raw/overfit_debug/{train_log.csv,checks.json,...},
generations/overfit_debug.json and plots/overfit_loss.png. Exits non-zero if
any check fails; if it does, do not start full pretraining.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinyllm.config import Config
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.tokenizer_metadata import load_verified_tokenizer
from tinyllm.eval.generation import greedy_generate
from tinyllm.model.model import TinyLLM
from tinyllm.training.overfit import (
    OverfitConfig,
    StepLog,
    build_tiny_dataset,
    evaluate_checks,
    evaluate_loss,
    fraction_of_params_changed,
    memorization_rate,
    run_overfit,
    snapshot_params,
)
from tinyllm.utils import RunContext

EXPERIMENT_ID = "overfit_debug"
SAMPLE_INDICES = (0, 1, 2)


def ascii_curve(logs: list[StepLog], width: int = 64, height: int = 14) -> str:
    """Log-scale loss curve for the terminal."""
    steps = [l.step for l in logs]
    logs_y = [math.log10(l.loss) for l in logs]
    lo, hi = min(logs_y), max(logs_y)
    grid = [[" "] * width for _ in range(height)]
    for step, y in zip(steps, logs_y):
        col = min(width - 1, int((step - steps[0]) / max(steps[-1] - steps[0], 1) * (width - 1)))
        row = height - 1 - int((y - lo) / max(hi - lo, 1e-9) * (height - 1))
        grid[row][col] = "*"
    lines = []
    for r, row in enumerate(grid):
        value = 10 ** (hi - (hi - lo) * r / (height - 1))
        lines.append(f"{value:9.3f} |{''.join(row)}")
    lines.append(" " * 10 + "+" + "-" * width)
    lines.append(f"{'':10} step {steps[0]}{'':>{width - 14}}{steps[-1]}")
    return "\n".join(lines)


def save_plot(logs: list[StepLog], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([l.step for l in logs], [l.loss for l in logs])
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (log scale)")
    ax.set_title("Overfit test: debug model on 200 TinyStories")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    run = OverfitConfig()
    cfg = Config.from_yaml("configs/debug.yaml")
    cfg.experiment_id, cfg.amp_dtype = EXPERIMENT_ID, "none"  # plain fp32, keep it simple
    ctx = RunContext.create(cfg, exist_ok=True)  # re-runs overwrite this experiment only
    ctx.save_metadata()
    device = ctx.device

    tokenizer = load_verified_tokenizer(cfg.tokenizer.dir)
    train = load_pretrain_splits(cfg.data)["train"]  # training split only
    data = build_tiny_dataset(
        (row[cfg.data.text_field] for row in train), tokenizer, run.num_sequences, run.seq_len
    )
    print(f"device={device}  data={tuple(data.shape)} ({data.numel():,} tokens)  "
          f"model params={TinyLLM(cfg.model).num_parameters():,}")

    model = TinyLLM(cfg.model).to(device)
    params_before = snapshot_params(model)

    samples: dict[str, dict] = {}

    def on_snapshot(step: int, m: TinyLLM) -> None:
        batch = data.to(device)
        prompts = batch[list(SAMPLE_INDICES), : run.prompt_len]
        out = greedy_generate(m, prompts, run.gen_tokens).cpu()
        samples[str(step)] = {
            "match": memorization_rate(m, data, device, run.prompt_len, run.gen_tokens),
            "generated": [tokenizer.decode(row[run.prompt_len:].tolist()) for row in out],
        }

    start_loss = evaluate_loss(model, data, device)
    steps = (0, run.steps // 2, run.steps)
    logs = run_overfit(model, data, run, device, snapshot_steps=steps, on_snapshot=on_snapshot)
    final_loss = evaluate_loss(model, data, device)
    match = memorization_rate(model, data, device, run.prompt_len, run.gen_tokens)
    checks = evaluate_checks(logs, start_loss, final_loss, fraction_of_params_changed(params_before, model), match)

    # ---- save ----
    with open(ctx.run_dir / "train_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "lr", "grad_norm", "tokens_per_sec"])
        w.writerows([l.step, l.loss, l.lr, l.grad_norm, round(l.tokens_per_sec)] for l in logs)
    (ctx.run_dir / "checks.json").write_text(json.dumps([c.__dict__ for c in checks], indent=2) + "\n")
    truth = [tokenizer.decode(data[i, run.prompt_len : run.prompt_len + run.gen_tokens].tolist())
             for i in SAMPLE_INDICES]
    prompts = [tokenizer.decode(data[i, : run.prompt_len].tolist()) for i in SAMPLE_INDICES]
    gen_path = Path("results/generations") / f"{EXPERIMENT_ID}.json"
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps({"prompts": prompts, "true_continuations": truth,
                                    "snapshots": samples}, indent=2, ensure_ascii=False) + "\n")
    save_plot(logs, Path("results/plots/overfit_loss.png"))

    # ---- report ----
    print(f"\nstart loss (all {run.num_sequences} sequences): {start_loss:.4f}   final: {final_loss:.4f}")
    print(f"logged batch loss: first {logs[0].loss:.4f} -> last {logs[-1].loss:.4f}\n")
    print(f"{'step':>5} {'loss':>9} {'lr':>9} {'grad_norm':>10} {'tok/s':>9}")
    for l in [x for x in logs if x.step in (1, 10, 50, 100, 200, 400, 750, 1000, 1500)]:
        print(f"{l.step:>5} {l.loss:>9.4f} {l.lr:>9.2e} {l.grad_norm:>10.3f} {l.tokens_per_sec:>9,.0f}")
    print("\nloss curve (log scale):\n" + ascii_curve(logs))
    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"\n=== sample {idx} ===\nPROMPT : {prompts[i]!r}\nTRUE   : {truth[i]!r}")
        for step, label in zip(steps, ("BEFORE", "MIDDLE", "END   ")):
            snap = samples[str(step)]
            print(f"{label} (step {step}, match {snap['match']['token_match']:.0%}): {snap['generated'][i]!r}")
    print("\nchecks:")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    passed = all(c.passed for c in checks)
    print(f"\nOVERFIT TEST {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
