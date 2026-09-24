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
from tinyllm.data.tokenizer import BOS_TOKEN, EOS_TOKEN, TOKENIZER_FILE, Tokenizer

METADATA_FILE = "metadata.json"


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
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def write_tokenizer_metadata(directory: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(directory) / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path
