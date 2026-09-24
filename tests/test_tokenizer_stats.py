"""Tests for tokenizer statistics and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.config import DataConfig
from tinyllm.data.tokenizer import BPETokenizer
from tinyllm.data.tokenizer_metadata import (
    TokenizerIntegrityError,
    build_tokenizer_metadata,
    ensure_safe_to_train,
    load_verified_tokenizer,
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


# --- fixed tokenizer artifact: integrity checks -------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _saved(tok, directory):
    tok.save(directory)
    meta = build_tokenizer_metadata(tok, directory, DataConfig(), num_train_docs=1)
    write_tokenizer_metadata(directory, meta)
    return directory


def test_verified_load_accepts_matching_artifact(tok, tmp_path):
    loaded = load_verified_tokenizer(_saved(tok, tmp_path))
    assert loaded.encode("Lily") == tok.encode("Lily")


def test_verified_load_never_trains_when_missing(tmp_path):
    with pytest.raises(TokenizerIntegrityError, match="not found"):
        load_verified_tokenizer(tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing was created


def test_verified_load_requires_metadata(tok, tmp_path):
    tok.save(tmp_path)
    with pytest.raises(TokenizerIntegrityError, match="metadata"):
        load_verified_tokenizer(tmp_path)


def test_verified_load_detects_replaced_tokenizer(tok, tmp_path):
    _saved(tok, tmp_path)
    other = BPETokenizer.train(CORPUS[:2], vocab_size=300)
    other.save(tmp_path)  # a retrained tokenizer silently swapped in
    with pytest.raises(TokenizerIntegrityError, match="hash"):
        load_verified_tokenizer(tmp_path)


def test_train_refuses_to_overwrite_existing(tok, tmp_path):
    ensure_safe_to_train(tmp_path)  # empty dir: fine
    _saved(tok, tmp_path)
    with pytest.raises(TokenizerIntegrityError, match="refusing to retrain"):
        ensure_safe_to_train(tmp_path)
    ensure_safe_to_train(tmp_path, force=True)


def test_committed_tokenizer_artifact_is_valid():
    tok = load_verified_tokenizer(REPO_ROOT / "artifacts" / "tokenizer")
    assert (tok.vocab_size, tok.bos_id, tok.eos_id) == (16000, 0, 1)
    meta = json.loads((REPO_ROOT / "artifacts" / "tokenizer" / "metadata.json").read_text())
    assert meta["algorithm"] == "byte-level BPE"
    assert meta["training_data"]["dataset"] == "roneneldan/TinyStories"
    assert meta["training_data"]["split"] == "train"
    assert meta["training_data"]["split_seed"] == 42
