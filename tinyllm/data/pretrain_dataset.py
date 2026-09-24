"""TinyStories pretraining data: loading and a fixed train/validation split.

The split is a pure function of (dataset length, val_fraction, split_seed), so
every experiment that uses the same ``DataConfig`` gets identical splits.
Text is returned raw; tokenization happens in a later step.

Run ``python -m tinyllm.data.pretrain_dataset`` to print sizes and examples.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from tinyllm.config import DataConfig


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted (train, val) row indices for a dataset of ``n`` rows.

    Rows are assigned by a seeded permutation; indices are sorted so reads
    stay sequential on disk (training order is shuffled later by the loader).
    """
    n_val = round(n * val_fraction)
    if not 0 < n_val < n:
        raise ValueError(f"val_fraction={val_fraction} leaves an empty split for n={n}")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def split_train_val(dataset: Dataset, val_fraction: float, seed: int) -> DatasetDict:
    """Split ``dataset`` into fixed ``train`` / ``validation`` subsets."""
    train_idx, val_idx = split_indices(len(dataset), val_fraction, seed)
    return DatasetDict(
        train=dataset.select(train_idx),
        validation=dataset.select(val_idx),
    )


def load_pretrain_splits(cfg: DataConfig) -> DatasetDict:
    """Load the dataset's ``train`` split and carve our own train/validation from it.

    The Hub's own ``validation`` split is not used, so the split is fully
    controlled by ``cfg.val_fraction`` and ``cfg.split_seed``.
    """
    raw = load_dataset(cfg.dataset_name, split="train")
    return split_train_val(raw, cfg.val_fraction, cfg.split_seed)


def _preview(splits: DatasetDict, text_field: str, num_examples: int = 4) -> None:
    for name, ds in splits.items():
        print(f"{name}: {len(ds):,} examples")
    for i in range(num_examples):
        text = splits["train"][i][text_field]
        print(f"\n--- train example {i} ---\n{text}")


if __name__ == "__main__":
    cfg = DataConfig()
    _preview(load_pretrain_splits(cfg), cfg.text_field)
