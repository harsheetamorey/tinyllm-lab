"""Pretraining loop: gradient accumulation, clipping, mixed precision, eval, logging, checkpoints.

All collaborators (optimizer, scheduler, precision, batch source, evaluator,
logger, checkpoint manager) are injected, so the loop only orchestrates.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from tinyllm.config import Config
from tinyllm.data.packed_dataset import BatchSource
from tinyllm.model.model import TinyLLM
from tinyllm.training.checkpoint import Checkpoint, CheckpointManager
from tinyllm.training.evaluator import Evaluator
from tinyllm.training.loss import next_token_loss
from tinyllm.training.metrics import MetricLogger, MetricRecord, device_memory_mb
from tinyllm.training.precision import MixedPrecision
from tinyllm.training.schedule import LRScheduler
from tinyllm.utils.repro import capture_rng_state, restore_rng_state


def _sync(device: torch.device) -> None:
    """Wait for queued device work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Trainer:
    """Trains ``model`` for ``config.train.max_steps`` optimizer steps.

    One *optimizer step* consumes ``grad_accum_steps`` micro-batches; ``step``
    and every ``*_every`` setting count optimizer steps.
    """

    def __init__(
        self,
        config: Config,
        model: TinyLLM,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        precision: MixedPrecision,
        batches: BatchSource,
        evaluator: Evaluator,
        logger: MetricLogger,
        checkpoints: CheckpointManager,
        device: torch.device,
    ) -> None:
        self._config = config
        self._cfg = config.train
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._precision = precision
        self._batches = batches
        self._evaluator = evaluator
        self._logger = logger
        self._checkpoints = checkpoints
        self._device = device

        self._step = 0
        self._tokens_seen = 0
        self._micro_in_window = 0
        # Device-side accumulators; read with .item() only when logging.
        self._accum_loss = torch.zeros((), device=device)
        self._last_grad_norm = torch.zeros((), device=device)
        self._last_lr = scheduler.last_lr
        self._reset_log_window()

    @property
    def model(self) -> TinyLLM:
        return self._model

    @property
    def step(self) -> int:
        """Optimizer steps completed."""
        return self._step

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    # ---- one step -------------------------------------------------------

    def micro_step(self, batch: torch.Tensor) -> bool:
        """Forward/backward one micro-batch; returns True if it completed an optimizer step."""
        batch = batch.to(self._device)
        with self._precision.autocast():
            logits = self._model(batch)
        loss = next_token_loss(logits, batch)
        self._precision.backward(loss / self._cfg.grad_accum_steps)

        self._accum_loss += loss.detach() / self._cfg.grad_accum_steps
        self._tokens_seen += batch.numel()
        self._micro_in_window += 1
        if self._micro_in_window < self._cfg.grad_accum_steps:
            return False
        self._finish_optimizer_step()
        return True

    def train_step(self) -> None:
        """Run micro-batches until one optimizer step has been taken."""
        while not self.micro_step(self._batches.next_batch()):
            pass

    def _finish_optimizer_step(self) -> None:
        self._precision.unscale(self._optimizer)
        max_norm = self._cfg.grad_clip if self._cfg.grad_clip > 0 else math.inf
        self._last_grad_norm = torch.nn.utils.clip_grad_norm_(  # returns the pre-clip norm
            self._model.parameters(), max_norm
        )
        self._last_lr = self._scheduler.last_lr
        self._precision.step(self._optimizer)
        self._optimizer.zero_grad(set_to_none=True)
        self._scheduler.step()
        self._step += 1
        self._micro_in_window = 0

        self._window_loss += self._accum_loss
        self._window_steps += 1
        self._accum_loss = torch.zeros((), device=self._device)

    # ---- full run -------------------------------------------------------

    def fit(self) -> None:
        """Train until ``max_steps``, logging/evaluating/checkpointing on schedule."""
        cfg = self._cfg
        self._model.train()
        self._reset_log_window()
        while self._step < cfg.max_steps:
            self.train_step()
            last = self._step == cfg.max_steps
            do_eval = last or self._step % cfg.eval_every == 0
            if do_eval or self._step % cfg.log_every == 0:
                self._log(evaluate=do_eval)
            if last or self._step % cfg.checkpoint_every == 0:
                self.save_checkpoint()
                self._reset_log_window()  # keep checkpoint I/O out of the throughput numbers
        self._logger.close()

    def _log(self, evaluate: bool) -> None:
        _sync(self._device)
        elapsed = time.perf_counter() - self._window_start  # taken before eval, which is not training time
        val = self._evaluator.evaluate(self._model) if evaluate else None
        train_loss = (self._window_loss / self._window_steps).item()
        grad_norm = self._last_grad_norm.item()
        self._require_finite(train_loss=train_loss, grad_norm=grad_norm)
        if val is not None:
            self._require_finite(val_loss=val.loss)
        record: MetricRecord = {
            "step": self._step,
            "tokens_seen": self._tokens_seen,
            "train_loss": train_loss,
            "val_loss": val.loss if val else None,
            "val_ppl": val.perplexity if val else None,
            "lr": self._last_lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": (self._tokens_seen - self._window_start_tokens) / elapsed,
            "step_time_s": elapsed / self._window_steps,
            "memory_mb": device_memory_mb(self._device),
        }
        self._logger.log(record)
        self._reset_log_window()  # also keeps the eval pass out of the throughput numbers

    def _reset_log_window(self) -> None:
        _sync(self._device)
        self._window_start = time.perf_counter()
        self._window_start_tokens = self._tokens_seen
        self._window_steps = 0
        self._window_loss = torch.zeros((), device=self._device)

    @staticmethod
    def _require_finite(**values: float) -> None:
        for name, value in values.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"{name} is not finite: {value}")

    # ---- checkpoints ----------------------------------------------------

    def save_checkpoint(self) -> Path:
        """Persist model, optimizer, scheduler, counters, config, RNG and data position."""
        self._require_finite(grad_norm=self._last_grad_norm.item())  # never save a poisoned model
        return self._checkpoints.save(
            Checkpoint(
                model=self._model.state_dict(),
                optimizer=self._optimizer.state_dict(),
                scheduler=self._scheduler.state_dict(),
                scaler=self._precision.state_dict(),
                step=self._step,
                tokens_seen=self._tokens_seen,
                config=self._config.to_dict(),
                rng=capture_rng_state(),
                data=self._batches.state_dict(),
            )
        )

    def resume(self, path: Path) -> None:
        """Restore everything from a checkpoint; must be called before :meth:`fit`."""
        if self._micro_in_window:
            raise RuntimeError("cannot resume in the middle of a gradient-accumulation window")
        ckpt = self._checkpoints.load(path)
        if ckpt.config["model"] != self._config.to_dict()["model"]:
            raise ValueError("checkpoint was trained with a different model config")
        self._model.load_state_dict(ckpt.model)
        self._optimizer.load_state_dict(ckpt.optimizer)
        self._scheduler.load_state_dict(ckpt.scheduler)
        self._precision.load_state_dict(ckpt.scaler)
        self._batches.load_state_dict(ckpt.data)
        self._step, self._tokens_seen = ckpt.step, ckpt.tokens_seen
        self._last_lr = self._scheduler.last_lr
        restore_rng_state(ckpt.rng)  # last, so nothing above consumes random numbers after it
        self._reset_log_window()
