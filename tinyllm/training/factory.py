"""Wires the training components together from a run context and packed token blocks."""

from __future__ import annotations

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
    train_blocks: PackedBlocks,
    val_blocks: PackedBlocks,
    logger: MetricLogger,
) -> Trainer:
    """Build a ready-to-run :class:`Trainer` for ``ctx.config``.

    The blocks must be ``model.max_sequence_length`` long (see
    ``tinyllm.data.pretrain_data.load_pretrain_blocks``). The model is created
    after the context's seeding, so a given seed always gives the same init.
    """
    config, device = ctx.config, ctx.device
    for name, blocks in (("train", train_blocks), ("validation", val_blocks)):
        if blocks.seq_len != config.model.max_sequence_length:
            raise ValueError(
                f"{name} blocks are {blocks.seq_len} tokens but model.max_sequence_length "
                f"is {config.model.max_sequence_length}"
            )
    precision = MixedPrecision(device, ctx.amp_dtype)

    model = TinyLLM(config.model).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = WarmupCosineScheduler(optimizer, config.train)
    batches = ShuffledBatchSource(train_blocks, config.train.micro_batch_size, config.seed)
    val_batches = fixed_batches(val_blocks, config.train.micro_batch_size, config.train.eval_batches)
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
