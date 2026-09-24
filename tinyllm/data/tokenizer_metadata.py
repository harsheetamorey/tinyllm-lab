"""Metadata describing a trained tokenizer artifact, saved next to it.

Lets anyone confirm which tokenizer they have (algorithm, ids, training data,
file hash) without retraining.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import tokenizers

from tinyllm.config import DataConfig
from tinyllm.data.tokenizer import BOS_TOKEN, EOS_TOKEN, TOKENIZER_FILE, BPETokenizer, Tokenizer

METADATA_FILE = "metadata.json"


class TokenizerIntegrityError(RuntimeError):
    """The tokenizer artifact is missing, changed, or does not match its metadata."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tokenizer_metadata(
    tokenizer: Tokenizer,
    directory: str | Path,
    data_cfg: DataConfig,
    num_train_docs: int,
) -> dict[str, Any]:
    """Describe the tokenizer saved in ``directory``.

    ``num_train_docs`` is the size of the training split it was trained on.
    """
    directory = Path(directory)
    artifact = directory / TOKENIZER_FILE
    return {
        "algorithm": "byte-level BPE",
        "library": f"tokenizers {tokenizers.__version__}",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {
            "bos": {"token": BOS_TOKEN, "id": tokenizer.bos_id},
            "eos": {"token": EOS_TOKEN, "id": tokenizer.eos_id},
        },
        "training_data": {
            "dataset": data_cfg.dataset_name,
            "split": "train",
            "val_fraction": data_cfg.val_fraction,
            "split_seed": data_cfg.split_seed,
            "num_documents": num_train_docs,
        },
        "artifact_path": artifact.as_posix(),
        "artifact_sha256": _sha256(artifact),
    }


def write_tokenizer_metadata(directory: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(directory) / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load_verified_tokenizer(directory: str | Path) -> BPETokenizer:
    """Load the fixed tokenizer artifact after checking it against its metadata.

    Training, SFT and evaluation code must load the tokenizer through this
    function. It never trains: a missing artifact, a file that differs from the
    recorded hash, or different vocab size / BOS / EOS ids all raise
    :class:`TokenizerIntegrityError`.
    """
    directory = Path(directory)
    artifact = directory / TOKENIZER_FILE
    meta_path = directory / METADATA_FILE
    if not artifact.is_file():
        raise TokenizerIntegrityError(f"tokenizer artifact not found: {artifact}")
    if not meta_path.is_file():
        raise TokenizerIntegrityError(f"tokenizer metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    if _sha256(artifact) != meta["artifact_sha256"]:
        raise TokenizerIntegrityError(
            f"{artifact} does not match the hash in {meta_path}; the tokenizer was "
            "modified or retrained. Restore it (git checkout) or, if intended, "
            "regenerate the metadata with scripts.write_tokenizer_metadata."
        )

    tokenizer = BPETokenizer.load(directory)
    actual = {
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
    }
    expected = {
        "vocab_size": meta["vocab_size"],
        "bos_id": meta["special_tokens"]["bos"]["id"],
        "eos_id": meta["special_tokens"]["eos"]["id"],
    }
    if actual != expected:
        raise TokenizerIntegrityError(f"tokenizer {actual} does not match metadata {expected}")
    return tokenizer


def ensure_safe_to_train(directory: str | Path, force: bool = False) -> None:
    """Refuse to train over an existing tokenizer unless ``force`` is set."""
    artifact = Path(directory) / TOKENIZER_FILE
    if artifact.exists() and not force:
        raise TokenizerIntegrityError(
            f"{artifact} already exists and is the fixed tokenizer for this project; "
            "refusing to retrain. Pass --force only if you intend to replace it."
        )
