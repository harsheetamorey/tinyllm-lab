"""Reusable on-disk cache of the tokenized, packed pretraining splits.

Each split is stored as one ``[num_sequences, seq_len]`` uint16 ``.npy`` array
plus a JSON sidecar describing exactly how it was made. Stories are encoded as
``<bos> story <eos>``, concatenated in split order (nothing is shuffled),
and cut into ``seq_len`` blocks; the leftover tail shorter than one block is
dropped and counted in the metadata.

The sidecar's *identity* (tokenizer hash, special ids, dataset split, sequence
length, format version) must equal what the caller expects, otherwise
:func:`load_token_cache` raises :class:`IncompatibleCacheError` instead of
silently feeding training data made with something else.
"""

from __future__ import annotations

import hashlib
import json
import os
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tinyllm.config import DataConfig
from tinyllm.data.tokenizer import Tokenizer
from tinyllm.data.tokenizer_metadata import METADATA_FILE

FORMAT_VERSION = 1
TOKEN_DTYPE = np.uint16  # vocab ids must fit; checked at build time
SPLITS = ("train", "validation")


class IncompatibleCacheError(RuntimeError):
    """The cache was built with a different tokenizer, split, sequence length or format."""


@dataclass(frozen=True)
class CacheIdentity:
    """Everything that determines a cache's contents; two caches with equal identity are interchangeable."""

    format_version: int
    tokenizer_sha256: str
    vocab_size: int
    bos_id: int
    eos_id: int
    dataset_name: str
    split: str
    val_fraction: float
    split_seed: int
    seq_len: int

    @classmethod
    def create(
        cls,
        tokenizer: Tokenizer,
        tokenizer_dir: str | Path,
        data_cfg: DataConfig,
        split: str,
        seq_len: int,
    ) -> "CacheIdentity":
        """Identity for ``split`` under ``tokenizer``, which must be the verified one loaded from ``tokenizer_dir``."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        meta = json.loads((Path(tokenizer_dir) / METADATA_FILE).read_text())
        return cls(
            format_version=FORMAT_VERSION,
            tokenizer_sha256=meta["artifact_sha256"],
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            dataset_name=data_cfg.dataset_name,
            split=split,
            val_fraction=data_cfg.val_fraction,
            split_seed=data_cfg.split_seed,
            seq_len=seq_len,
        )

    def differences(self, other: "CacheIdentity") -> dict[str, tuple[Any, Any]]:
        """Fields where ``other`` differs, as ``{field: (mine, theirs)}``."""
        mine, theirs = asdict(self), asdict(other)
        return {k: (mine[k], theirs[k]) for k in mine if mine[k] != theirs[k]}


@dataclass(frozen=True)
class CacheStats:
    """Size of a cached split."""

    num_documents: int
    num_sequences: int
    total_tokens: int    # every token produced, including BOS/EOS and the dropped tail
    packed_tokens: int   # num_sequences * seq_len
    dropped_tokens: int  # tail too short to fill a sequence


@dataclass(frozen=True)
class TokenCache:
    """A loaded split: ``sequences`` is a read-only memory-mapped ``[N, seq_len]`` array."""

    sequences: np.ndarray
    identity: CacheIdentity
    stats: CacheStats


def cache_paths(root: str | Path, split: str, seq_len: int) -> tuple[Path, Path]:
    """(array path, metadata path) for a split; caches of different lengths live side by side."""
    base = Path(root) / f"seq{seq_len}" / split
    return base.with_suffix(".npy"), base.with_suffix(".json")


def pack_tokens(stream: np.ndarray, seq_len: int) -> np.ndarray:
    """Cut a flat stream into ``[N, seq_len]`` blocks, dropping the ragged tail."""
    usable = (len(stream) // seq_len) * seq_len
    return stream[:usable].reshape(-1, seq_len)


def tokenize_documents(
    text_batches: Iterable[Sequence[str]], tokenizer: Tokenizer
) -> tuple[np.ndarray, int]:
    """Encode every document as ``<bos> text <eos>``; returns (flat uint16 stream, document count)."""
    if tokenizer.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in {TOKEN_DTYPE.__name__}")
    stream = array("H")
    num_documents = 0
    for texts in text_batches:
        for ids in tokenizer.encode_batch(texts, add_bos=True, add_eos=True):
            stream.extend(ids)
        num_documents += len(texts)
    return np.frombuffer(stream, dtype=TOKEN_DTYPE), num_documents


def build_token_cache(
    text_batches: Iterable[Sequence[str]],
    tokenizer: Tokenizer,
    identity: CacheIdentity,
    root: str | Path,
    overwrite: bool = False,
) -> CacheStats:
    """Tokenize and pack one split and write it under ``root``.

    Deterministic: the same texts, tokenizer and identity always give
    byte-identical files. Refuses to replace an existing cache unless
    ``overwrite``. The metadata file is written last, so a crash mid-build
    leaves no loadable (half-written) cache.
    """
    array_path, meta_path = cache_paths(root, identity.split, identity.seq_len)
    if (array_path.exists() or meta_path.exists()) and not overwrite:
        raise FileExistsError(f"{array_path} already exists; pass overwrite=True to rebuild it")

    stream, num_documents = tokenize_documents(text_batches, tokenizer)
    sequences = pack_tokens(stream, identity.seq_len)
    if len(sequences) == 0:
        raise ValueError(f"only {len(stream)} tokens, fewer than one sequence of {identity.seq_len}")
    stats = CacheStats(
        num_documents=num_documents,
        num_sequences=len(sequences),
        total_tokens=len(stream),
        packed_tokens=sequences.size,
        dropped_tokens=len(stream) - sequences.size,
    )

    array_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # an overwritten cache is invalid until fully rewritten
    tmp = array_path.with_suffix(".tmp.npy")
    np.save(tmp, sequences)
    os.replace(tmp, array_path)
    meta = {
        "identity": asdict(identity),
        "stats": asdict(stats),
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "shape": list(sequences.shape),
        "tokens_sha256": _sha256_array(sequences),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return stats


def load_token_cache(root: str | Path, expected: CacheIdentity, verify_hash: bool = False) -> TokenCache:
    """Open a cached split, raising :class:`IncompatibleCacheError` unless it matches ``expected``.

    ``verify_hash`` additionally re-hashes the whole array to detect corruption
    (reads the entire file, so it is off by default).
    """
    array_path, meta_path = cache_paths(root, expected.split, expected.seq_len)
    if not meta_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"no complete token cache at {array_path} (build it first)")
    meta = json.loads(meta_path.read_text())

    try:
        stored = CacheIdentity(**meta["identity"])
    except TypeError as exc:
        raise IncompatibleCacheError(f"{meta_path} has an unrecognized identity layout: {exc}") from exc
    diff = expected.differences(stored)
    if diff:
        details = ", ".join(f"{k}: expected {want!r}, cache has {got!r}" for k, (want, got) in diff.items())
        raise IncompatibleCacheError(f"{array_path} is incompatible ({details}); rebuild it")

    sequences = np.load(array_path, mmap_mode="r")
    stats = CacheStats(**meta["stats"])
    if (
        sequences.dtype != np.dtype(meta["dtype"])
        or list(sequences.shape) != meta["shape"]
        or sequences.shape != (stats.num_sequences, expected.seq_len)
    ):
        raise IncompatibleCacheError(f"{array_path} does not match its metadata (truncated or replaced?)")
    if verify_hash and _sha256_array(sequences) != meta["tokens_sha256"]:
        raise IncompatibleCacheError(f"{array_path} content hash differs from its metadata (corrupted?)")
    return TokenCache(sequences=sequences, identity=stored, stats=stats)


def _sha256_array(array_: np.ndarray, chunk_rows: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(array_), chunk_rows):
        digest.update(np.ascontiguousarray(array_[start : start + chunk_rows]).tobytes())
    return digest.hexdigest()
