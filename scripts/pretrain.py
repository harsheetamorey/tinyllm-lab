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

from tinyllm.config import Config
from tinyllm.data.pretrain_data import PretrainBlocks, load_pretrain_blocks
from tinyllm.training.checkpoint import CheckpointManager
from tinyllm.training.factory import build_trainer
from tinyllm.training.metrics import ConsoleMetricLogger, JsonlMetricLogger, MetricLogger, MultiMetricLogger
from tinyllm.training.trainer import Trainer
from tinyllm.utils import RunContext

METRICS_FILE = "metrics.jsonl"


def make_logger(ctx: RunContext) -> MetricLogger:
    return MultiMetricLogger(ConsoleMetricLogger(), JsonlMetricLogger(ctx.run_dir / METRICS_FILE))


def start_trainer(
    config: Config, data: PretrainBlocks, overwrite: bool, resume: bool
) -> tuple[Trainer, RunContext]:
    """Create the run directory (fresh, overwritten or resumed) and build its trainer."""
    ctx = RunContext.create(config, exist_ok=resume, overwrite=overwrite)
    if resume:
        latest = CheckpointManager(ctx.checkpoint_dir).latest()
        if latest is None:
            raise FileNotFoundError(f"no checkpoint to resume from in {ctx.checkpoint_dir}")
    else:
        ctx.save_metadata()
    trainer = build_trainer(ctx, data.train, data.validation, logger=make_logger(ctx))
    if resume:
        trainer.resume(latest)
        print(f"resumed from {latest} at step {trainer.step}")
    return trainer, ctx


def print_run_summary(trainer: Trainer, ctx: RunContext) -> None:
    """Exact parameter count, batch geometry, device and precision, before training starts."""
    cfg = ctx.config
    micro, accum, seq_len = cfg.train.micro_batch_size, cfg.train.grad_accum_steps, cfg.model.max_sequence_length
    print(f"experiment={cfg.experiment_id} device={ctx.device} amp_dtype={ctx.amp_dtype}")
    print(f"trainable parameters: {trainer.model.num_parameters():,}")
    print(f"batch: micro={micro} x accum={accum} = {micro * accum} sequences of {seq_len} "
          f"= {micro * accum * seq_len:,} tokens per optimizer step; {cfg.train.max_steps} steps "
          f"= {cfg.train.max_steps * micro * accum * seq_len:,} tokens")


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
    data = load_pretrain_blocks(config)
    print(data.description)
    try:
        trainer, ctx = start_trainer(config, data, args.overwrite, args.resume)
    except FileExistsError:
        parser.error(
            f"results/raw/{config.experiment_id} already exists: use a new --experiment-id, "
            "--overwrite to replace it, or --resume to continue it"
        )
    print_run_summary(trainer, ctx)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
