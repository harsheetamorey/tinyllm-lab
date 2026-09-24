"""Evaluate the saved tokenizer on the validation split and save the statistics.

The validation split is text the tokenizer was never trained on.

Usage: python -m scripts.evaluate_tokenizer [config.yaml]
"""

from __future__ import annotations

import sys

from tinyllm.config import Config
from tinyllm.data.pretrain_dataset import load_pretrain_splits
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.eval.tokenizer_stats import compute_stats, encode_examples, save_report

REPORT_PATH = "results/tables/tokenizer_stats.json"


def main(config_path: str | None = None) -> None:
    cfg = Config.from_yaml(config_path) if config_path else Config()
    tokenizer = BPETokenizer.load(cfg.tokenizer.dir)
    validation = load_pretrain_splits(cfg.data)["validation"]

    stats = compute_stats(tokenizer, validation[cfg.data.text_field])
    examples = encode_examples(tokenizer)
    corpus = f"{cfg.data.dataset_name} validation split ({stats.num_documents:,} documents)"
    path = save_report(REPORT_PATH, stats, examples, corpus)

    print(f"corpus: {corpus}")
    print(f"final vocabulary size:        {stats.vocab_size:,}")
    print(f"average tokens per document:  {stats.avg_tokens_per_document:.2f}")
    print(f"average characters per token: {stats.avg_characters_per_token:.3f}")
    print(f"unknown-token frequency:      n/a ({stats.unknown_token_note})")
    for ex in examples:
        print(f"\ntext:    {ex.text!r}\nids:     {ex.ids}\ntokens:  {ex.tokens}\ndecoded: {ex.decoded!r}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
