"""Tokenize stories into one stream, cut it into fixed-length blocks, and batch them.

Each story becomes ``<bos> ... <eos>`` and stories are concatenated, so every
block is exactly ``seq_len`` tokens (no padding). A block may start or end
mid-story; the next-token loss handles that like any other position.
"""

from __future__ import annotations

from array import array
from typing import Any, Iterable, Protocol

import numpy as np
import torch

from tinyllm.data.tokenizer import Tokenizer


def tokenize_stories(
    texts: Iterable[str], tokenizer: Tokenizer, max_stories: int | None = None
) -> torch.Tensor:
    """Encode up to ``max_stories`` texts into one flat int32 token stream."""
    stream = array("i")
    for count, text in enumerate(texts):
        if max_stories is not None and count >= max_stories:
            break
        stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.from_numpy(np.frombuffer(stream, dtype=np.int32).copy())


class PackedBlocks:
    """``[N, seq_len]`` blocks of token ids, held either in memory or as a memory-mapped cache array."""

    def __init__(self, sequences: torch.Tensor | np.ndarray) -> None:
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError(f"sequences must be [N, seq_len>=2], got shape {tuple(sequences.shape)}")
        self._blocks = sequences
        self.seq_len = sequences.shape[1]

    @classmethod
    def from_stream(cls, tokens: torch.Tensor, seq_len: int) -> "PackedBlocks":
        """Cut a flat token stream into non-overlapping blocks, dropping the ragged tail."""
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        return cls(tokens[: (len(tokens) // seq_len) * seq_len].view(-1, seq_len))

    def __len__(self) -> int:
        return len(self._blocks)

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        """Blocks at ``indices`` as a ``[len(indices), seq_len]`` long tensor."""
        if isinstance(self._blocks, torch.Tensor):
            return self._blocks[indices].long()
        return torch.from_numpy(self._blocks[indices.numpy()].astype(np.int64))


class BatchSource(Protocol):
    """Endless supply of training batches whose position can be checkpointed."""

    def next_batch(self) -> torch.Tensor: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class ShuffledBatchSource:
    """Epoch-wise shuffled batches of blocks; the last partial batch of an epoch is dropped.

    The order of epoch ``e`` is a pure function of ``(seed, e)``, so the whole
    position is just ``(epoch, cursor)`` and resuming replays the exact same
    batches.
    """

    def __init__(self, blocks: PackedBlocks, batch_size: int, seed: int) -> None:
        if len(blocks) < batch_size:
            raise ValueError(f"only {len(blocks)} blocks, fewer than batch_size={batch_size}")
        self._blocks = blocks
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self._cursor = 0
        self._order = self._epoch_order()

    @property
    def epoch(self) -> int:
        return self._epoch

    def next_batch(self) -> torch.Tensor:
        if self._cursor + self._batch_size > len(self._order):
            self._epoch += 1
            self._cursor = 0
            self._order = self._epoch_order()
        indices = self._order[self._cursor : self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return self._blocks.get(indices)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "cursor": self._cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch, self._cursor = int(state["epoch"]), int(state["cursor"])
        self._order = self._epoch_order()

    def _epoch_order(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self._seed + self._epoch)
        return torch.randperm(len(self._blocks), generator=gen)


def fixed_batches(blocks: PackedBlocks, batch_size: int, max_batches: int) -> list[torch.Tensor]:
    """The first ``max_batches`` full batches in stream order (a fixed validation set)."""
    batches = []
    for start in range(0, len(blocks) - batch_size + 1, batch_size):
        if len(batches) == max_batches:
            break
        batches.append(blocks.get(torch.arange(start, start + batch_size)))
    if not batches:
        raise ValueError(f"only {len(blocks)} blocks, fewer than batch_size={batch_size}")
    return batches
