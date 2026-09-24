"""Pretrain TinyLLM from a config file.

Usage:
    python -m scripts.pretrain --config configs/smoke.yaml --experiment-id my_run
    python -m scripts.pretrain --config configs/smoke.yaml --experiment-id my_run --resume

Outputs go to results/raw/<experiment_id>/ (metrics.jsonl, checkpoints/, metadata).
An existing experiment directory is never touched silently: pick a new
``--experiment-id``, pass ``--overwrite`` to delete and restart it, or
``--resume`` to continue from its latest checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tinyllm.config import Config
from tinyllm.data.packed_dataset import tokenize_stories
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.tokenizer_metadata import load_verified_tokenizer
from tinyllm.training.checkpoint import CheckpointManager
from tinyllm.training.factory import build_trainer
from tinyllm.training.metrics import ConsoleMetricLogger, JsonlMetricLogger, MetricLogger, MultiMetricLogger
from tinyllm.training.trainer import Trainer
from tinyllm.utils import RunContext

METRICS_FILE = "metrics.jsonl"


def load_token_streams(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize the (optionally capped) train and validation splits."""
    tokenizer = load_verified_tokenizer(config.tokenizer.dir)
    splits = load_pretrain_splits(config.data)
    field, data = config.data.text_field, config.data

    def tokens(split: str, cap: int | None) -> torch.Tensor:
        return tokenize_stories((row[field] for row in splits[split]), tokenizer, cap)

    return tokens("train", data.max_train_stories), tokens("validation", data.max_val_stories)


def make_logger(ctx: RunContext) -> MetricLogger:
    return MultiMetricLogger(ConsoleMetricLogger(), JsonlMetricLogger(ctx.run_dir / METRICS_FILE))


def start_trainer(
    config: Config, streams: tuple[torch.Tensor, torch.Tensor], overwrite: bool, resume: bool
) -> tuple[Trainer, RunContext]:
    """Create the run directory (fresh, overwritten or resumed) and build its trainer."""
    ctx = RunContext.create(config, exist_ok=resume, overwrite=overwrite)
    if resume:
        latest = CheckpointManager(ctx.checkpoint_dir).latest()
        if latest is None:
            raise FileNotFoundError(f"no checkpoint to resume from in {ctx.checkpoint_dir}")
    else:
        ctx.save_metadata()
    trainer = build_trainer(ctx, *streams, logger=make_logger(ctx))
    if resume:
        trainer.resume(latest)
        print(f"resumed from {latest} at step {trainer.step}")
    return trainer, ctx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", help="run name; defaults to the config's experiment_id")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing run directory and start over")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint of an existing run")
    args = parser.parse_args(argv)
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")

    config = Config.from_yaml(args.config)
    if args.experiment_id:
        config.experiment_id = args.experiment_id
    try:
        trainer, ctx = start_trainer(config, load_token_streams(config), args.overwrite, args.resume)
    except FileExistsError:
        parser.error(
            f"results/raw/{config.experiment_id} already exists: use a new --experiment-id, "
            "--overwrite to replace it, or --resume to continue it"
        )
    print(f"device={ctx.device} amp_dtype={ctx.amp_dtype} experiment={config.experiment_id}")
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
