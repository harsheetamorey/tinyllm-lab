"""Tokenize and pack the TinyStories train/validation splits once, for all later training runs.

Usage: python -m scripts.build_token_cache [--config configs/debug.yaml] [--seq-len 512] [--overwrite]

Uses the frozen tokenizer (verified against its metadata; never retrained) and
writes ``<data.token_cache_dir>/seq<L>/{train,validation}.{npy,json}``. An
existing cache is never replaced unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tinyllm.config import Config
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.token_cache import CacheIdentity, build_token_cache, cache_paths
from tinyllm.data.tokenizer_metadata import load_verified_tokenizer

DEFAULT_SEQ_LEN = 512
CHUNK_DOCS = 20_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--overwrite", action="store_true", help="rebuild caches that already exist")
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    tokenizer = load_verified_tokenizer(config.tokenizer.dir)
    splits = load_pretrain_splits(config.data)
    root = Path(config.data.token_cache_dir)

    for split in ("train", "validation"):
        identity = CacheIdentity.create(tokenizer, config.tokenizer.dir, config.data, split, args.seq_len)
        chunks = (batch[config.data.text_field] for batch in splits[split].iter(batch_size=CHUNK_DOCS))
        start = time.perf_counter()
        try:
            stats = build_token_cache(chunks, tokenizer, identity, root, overwrite=args.overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
        elapsed = time.perf_counter() - start
        array_path, meta_path = cache_paths(root, split, args.seq_len)
        size_mb = (array_path.stat().st_size + meta_path.stat().st_size) / 2**20
        print(f"{split:>10}: {stats.num_documents:,} docs, {stats.total_tokens:,} tokens, "
              f"{stats.num_sequences:,} x {args.seq_len} sequences "
              f"({stats.dropped_tokens} tail tokens dropped), {size_mb:,.1f} MiB, {elapsed:,.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
