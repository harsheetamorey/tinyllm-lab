"""Tests for the fixed pretraining train/validation split."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from tinyllm.data.pretrain_dataset import split_indices, split_train_val


def _toy(n: int = 1000) -> Dataset:
    return Dataset.from_dict({"text": [f"story {i}" for i in range(n)]})


def test_split_is_reproducible():
    a = split_train_val(_toy(), val_fraction=0.02, seed=42)
    b = split_train_val(_toy(), val_fraction=0.02, seed=42)
    assert a["train"]["text"] == b["train"]["text"]
    assert a["validation"]["text"] == b["validation"]["text"]


def test_different_seed_gives_different_split():
    _, a_val = split_indices(1000, 0.02, seed=42)
    _, b_val = split_indices(1000, 0.02, seed=7)
    assert not np.array_equal(a_val, b_val)


def test_split_sizes_and_disjoint_cover():
    train, val = split_indices(1000, 0.02, seed=42)
    assert len(val) == 20 and len(train) == 980
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(1000))


def test_split_indices_are_pinned():
    # Guards against silent changes to the split algorithm across versions.
    _, val = split_indices(1000, 0.02, seed=42)
    assert val[:5].tolist() == [55, 61, 127, 147, 260]


@pytest.mark.parametrize("fraction", [0.0, 1.0, 0.0001])
def test_degenerate_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        split_indices(100, fraction, seed=42)

