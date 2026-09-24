"""Train the BPE tokenizer on the TinyStories *training* split and save it.

Refuses to run if a tokenizer already exists (it is the fixed project
tokenizer); pass --force to replace it, then regenerate its metadata with
``python -m scripts.write_tokenizer_metadata``.

Usage: python -m scripts.train_tokenizer [config.yaml] [--force]
"""

from __future__ import annotations

import argparse
from typing import Iterator

from datasets import Dataset

from tinyllm.config import Config
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.data.tokenizer_metadata import ensure_safe_to_train

BATCH_SIZE = 1000


def iter_texts(dataset: Dataset, text_field: str) -> Iterator[str]:
    """Yield raw texts in batches, without loading the split into memory."""
    for start in range(0, len(dataset), BATCH_SIZE):
        yield from dataset[start : start + BATCH_SIZE][text_field]


def main(config_path: str | None = None, force: bool = False) -> None:
    cfg = Config.from_yaml(config_path) if config_path else Config()
    ensure_safe_to_train(cfg.tokenizer.dir, force)
    train = load_pretrain_splits(cfg.data)["train"]  # validation is never seen
    print(f"training BPE (vocab_size={cfg.tokenizer.vocab_size}) on {len(train):,} train stories")

    tokenizer = BPETokenizer.train(iter_texts(train, cfg.data.text_field), cfg.tokenizer.vocab_size)
    path = tokenizer.save(cfg.tokenizer.dir)

    print(f"saved {path} (vocab_size={tokenizer.vocab_size}, "
          f"bos_id={tokenizer.bos_id}, eos_id={tokenizer.eos_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?")
    parser.add_argument("--force", action="store_true", help="replace an existing tokenizer")
    args = parser.parse_args()
    main(args.config, args.force)
