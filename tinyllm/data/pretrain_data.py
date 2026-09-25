"""Choose where pretraining blocks come from: the token cache (normal) or raw text (debug)."""

from __future__ import annotations

from dataclasses import dataclass

from tinyllm.config import Config
from tinyllm.data.packed_dataset import PackedBlocks, tokenize_stories
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.token_cache import CacheIdentity, CacheStats, load_token_cache
from tinyllm.data.tokenizer_metadata import load_verified_tokenizer


@dataclass(frozen=True)
class PretrainBlocks:
    """Train and validation blocks plus a one-line description of where they came from."""

    train: PackedBlocks
    validation: PackedBlocks
    description: str


def load_pretrain_blocks(config: Config) -> PretrainBlocks:
    """Blocks for ``config.data.source``; the tokenizer is always the verified frozen one."""
    tokenizer = load_verified_tokenizer(config.tokenizer.dir)
    if config.data.source == "raw":
        return _load_raw(config, tokenizer)
    return _load_cached(config, tokenizer)


def _load_cached(config: Config, tokenizer) -> PretrainBlocks:
    """Open the cache after checking it matches this tokenizer, split and sequence length.

    Never builds or rebuilds a cache: a missing or incompatible one is an error
    (build it with ``python -m scripts.build_token_cache``). The content hash is
    verified too, so corruption is caught before training starts.
    """
    seq_len = config.model.max_sequence_length
    loaded, stats = {}, {}
    for split in ("train", "validation"):
        expected = CacheIdentity.create(tokenizer, config.tokenizer.dir, config.data, split, seq_len)
        cache = load_token_cache(config.data.token_cache_dir, expected, verify_hash=True)
        loaded[split], stats[split] = PackedBlocks(cache.sequences), cache.stats
    return PretrainBlocks(loaded["train"], loaded["validation"], _describe_cache(stats, seq_len))


def _describe_cache(stats: dict[str, CacheStats], seq_len: int) -> str:
    return "token cache (verified): " + "; ".join(
        f"{split} {s.num_documents:,} docs / {s.num_sequences:,} x {seq_len} sequences / "
        f"{s.total_tokens:,} tokens" for split, s in stats.items()
    )


def _load_raw(config: Config, tokenizer) -> PretrainBlocks:
    """Debug path: tokenize (optionally capped) raw text on the spot."""
    data, seq_len = config.data, config.model.max_sequence_length
    splits = load_pretrain_splits(data)

    def blocks(split: str, cap: int | None) -> PackedBlocks:
        stream = tokenize_stories((row[data.text_field] for row in splits[split]), tokenizer, cap)
        return PackedBlocks.from_stream(stream, seq_len)

    train, validation = blocks("train", data.max_train_stories), blocks("validation", data.max_val_stories)
    return PretrainBlocks(train, validation, f"RAW TEXT tokenized on the fly (debug): "
                          f"{len(train):,} train / {len(validation):,} validation sequences")
