"""Tests for tokenizer statistics and metadata."""

from __future__ import annotations

import json

import pytest

from tinyllm.config import DataConfig
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.data.tokenizer_metadata import (
    build_tokenizer_metadata,
    write_tokenizer_metadata,
)
from tinyllm.eval.tokenizer_stats import compute_stats, encode_examples, save_report

CORPUS = [
    "One day, a little girl named Lily found a needle in her room.",
    "Once upon a time, there was a little car named Beep.",
] * 50


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=320)


def test_stats_match_manual_counts(tok):
    texts = ["Once upon a time", "Lily"]
    stats = compute_stats(tok, texts)
    tokens = sum(len(tok.encode(t)) for t in texts)
    chars = sum(len(t) for t in texts)
    assert stats.num_documents == 2
    assert stats.total_tokens == tokens and stats.total_characters == chars
    assert stats.avg_tokens_per_document == tokens / 2
    assert stats.avg_characters_per_token == chars / tokens
    assert stats.vocab_size == 320
    assert stats.unknown_token_frequency is None


def test_empty_corpus_rejected(tok):
    with pytest.raises(ValueError):
        compute_stats(tok, [])


def test_examples_roundtrip_and_report(tok, tmp_path):
    examples = encode_examples(tok)
    assert [e.text for e in examples] == ["Once upon a time...", "Machine learning is...", "Hello, how are you?"]
    assert all(e.decoded == e.text for e in examples)
    assert all("".join(e.tokens) == e.text for e in examples)

    path = save_report(tmp_path / "out" / "stats.json", compute_stats(tok, CORPUS), examples, "toy")
    report = json.loads(path.read_text())
    assert report["corpus"] == "toy" and report["stats"]["vocab_size"] == 320
    assert len(report["examples"]) == 3


def test_metadata_contents(tok, tmp_path):
    tok.save(tmp_path)
    meta = build_tokenizer_metadata(tok, tmp_path, DataConfig(), num_train_docs=123)
    assert meta["algorithm"] == "byte-level BPE"
    assert meta["vocab_size"] == 320
    assert meta["special_tokens"]["bos"] == {"token": "<bos>", "id": 0}
    assert meta["special_tokens"]["eos"] == {"token": "<eos>", "id": 1}
    assert meta["training_data"]["split"] == "train"
    assert meta["training_data"]["num_documents"] == 123
    assert meta["artifact_path"].endswith("tokenizer.json")
    assert len(meta["artifact_sha256"]) == 64
    assert json.loads(write_tokenizer_metadata(tmp_path, meta).read_text()) == meta
