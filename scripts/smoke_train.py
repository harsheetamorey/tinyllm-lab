"""Smoke-test the pretraining pipeline: short run, then an exact-resume check.

Usage: python -m scripts.smoke_train [--overwrite]

1. Trains ``configs/smoke.yaml`` (~40 optimizer steps) and checks the loss falls,
   nothing is NaN/Inf, metrics are written and checkpoints exist.
2. Starts a second run (``<id>_resume``) from the first run's mid-way
   checkpoint, trains to the same final step and checks it lands on the same
   parameters as the uninterrupted run.

Exits non-zero if any check fails. Existing run directories need ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from scripts.pretrain import METRICS_FILE, load_token_streams, start_trainer
from tinyllm.config import Config

CONFIG_PATH = Path("configs/smoke.yaml")
RESUME_PARAM_TOLERANCE = 1e-5  # bf16 autocast on MPS/CUDA need not be bit-exact across runs


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def max_param_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max((a[k].cpu() - b[k].cpu()).abs().max().item() for k in a)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    overwrite = parser.parse_args().overwrite

    config = Config.from_yaml(CONFIG_PATH)
    streams = load_token_streams(config)

    print("== run 1: uninterrupted ==")
    trainer, ctx = start_trainer(config, streams, overwrite=overwrite, resume=False)
    trainer.fit()
    final_params = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    rows = read_metrics(ctx.run_dir / METRICS_FILE)
    checkpoints = sorted((ctx.checkpoint_dir).glob("step_*.pt"))

    print("\n== run 2: resume from the mid-way checkpoint ==")
    resumed_config = Config.from_yaml(CONFIG_PATH)
    resumed_config.experiment_id = f"{config.experiment_id}_resume"
    resumed, resumed_ctx = start_trainer(resumed_config, streams, overwrite=overwrite, resume=False)
    mid = checkpoints[0]
    resumed.resume(mid)
    print(f"resumed from {mid.name}: step={resumed.step} tokens_seen={resumed.tokens_seen:,}")
    resumed.fit()
    diff = max_param_diff(final_params, resumed.model.state_dict())

    checks = {
        "loss decreases": rows[-1]["train_loss"] < rows[0]["train_loss"],
        "no NaN/Inf in metrics": all(
            math.isfinite(v) for r in rows for v in r.values() if isinstance(v, float)
        ),
        "metrics written": len(rows) > 0,
        "checkpoints created": len(checkpoints) >= 2,
        "resume continues at right step": resumed.step == trainer.step == config.train.max_steps,
        "resume tokens_seen matches": resumed.tokens_seen == trainer.tokens_seen,
        f"resumed params match uninterrupted (max diff {diff:.2e})": diff <= RESUME_PARAM_TOLERANCE,
    }
    print(f"\nfirst logged train loss {rows[0]['train_loss']:.4f} -> final {rows[-1]['train_loss']:.4f}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
