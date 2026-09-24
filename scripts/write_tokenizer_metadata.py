"""Write metadata.json for the already-trained tokenizer (no retraining).

Usage: python -m scripts.write_tokenizer_metadata [config.yaml]
"""

from __future__ import annotations

import sys

from tinyllm.config import Config
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.data.tokenizer_metadata import build_tokenizer_metadata, write_tokenizer_metadata


def main(config_path: str | None = None) -> None:
    cfg = Config.from_yaml(config_path) if config_path else Config()
    tokenizer = BPETokenizer.load(cfg.tokenizer.dir)
    num_train_docs = len(load_pretrain_splits(cfg.data)["train"])
    metadata = build_tokenizer_metadata(tokenizer, cfg.tokenizer.dir, cfg.data, num_train_docs)
    print(f"wrote {write_tokenizer_metadata(cfg.tokenizer.dir, metadata)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
