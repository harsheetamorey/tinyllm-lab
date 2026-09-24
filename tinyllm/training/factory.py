"""Wires the training components together from a run context and token streams."""

from __future__ import annotations

import torch

from tinyllm.data.packed_dataset import PackedBlocks, ShuffledBatchSource, fixed_batches
from tinyllm.model.model import TinyLLM
from tinyllm.training.checkpoint import CheckpointManager
from tinyllm.training.evaluator import LossEvaluator
from tinyllm.training.metrics import MetricLogger
from tinyllm.training.optimizer import build_optimizer
from tinyllm.training.precision import MixedPrecision
from tinyllm.training.schedule import WarmupCosineScheduler
from tinyllm.training.trainer import Trainer
from tinyllm.utils.run_context import RunContext


def build_trainer(
    ctx: RunContext,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    logger: MetricLogger,
) -> Trainer:
    """Build a ready-to-run :class:`Trainer` for ``ctx.config``.

    ``train_tokens`` / ``val_tokens`` are flat token streams (see
    ``tinyllm.data.packed_dataset.tokenize_stories``). The model is created
    after the context's seeding, so a given seed always gives the same init.
    """
    config, device = ctx.config, ctx.device
    seq_len = config.model.max_sequence_length
    precision = MixedPrecision(device, ctx.amp_dtype)

    model = TinyLLM(config.model).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = WarmupCosineScheduler(optimizer, config.train)
    batches = ShuffledBatchSource(
        PackedBlocks(train_tokens, seq_len), config.train.micro_batch_size, config.seed
    )
    val_batches = fixed_batches(
        PackedBlocks(val_tokens, seq_len), config.train.micro_batch_size, config.train.eval_batches
    )
    return Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
        batches=batches,
        evaluator=LossEvaluator(val_batches, device, precision),
        logger=logger,
        checkpoints=CheckpointManager(ctx.checkpoint_dir),
        device=device,
    )
