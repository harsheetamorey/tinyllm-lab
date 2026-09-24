"""Token cache: equivalence with direct tokenization, boundaries, split separation, compatibility."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from tinyllm.config import DataConfig
from tinyllm.data.token_cache import (
    CacheIdentity,
    IncompatibleCacheError,
    build_token_cache,
    cache_paths,
    load_token_cache,
    pack_tokens,
)
from tinyllm.data.tokenizer import Tokenizer

BOS, EOS, SEQ_LEN = 0, 1, 8


class FakeTokenizer(Tokenizer):
    """Word-length tokenizer; ids 2.. so BOS/EOS are recognizable."""

    vocab_size = 100
    bos_id, eos_id = BOS, EOS

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [2 + len(w) for w in text.split()]
        return [*([BOS] if add_bos else []), *ids, *([EOS] if add_eos else [])]

    def decode(self, ids, skip_special_tokens=True):
        raise NotImplementedError

    def save(self, directory):
        raise NotImplementedError


TRAIN = [f"{'w' * (i % 5 + 1)} {'x' * (i % 3 + 1)} tail{i}" for i in range(40)]
VALID = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota", "kappa lambda mu nu xi omicron"] * 3


@pytest.fixture
def tokenizer_dir(tmp_path):
    directory = tmp_path / "tok"
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps({"artifact_sha256": "abc123"}))
    return directory


def identity(tokenizer_dir, split="train", seq_len=SEQ_LEN, **data) -> CacheIdentity:
    return CacheIdentity.create(FakeTokenizer(), tokenizer_dir, DataConfig(**data), split, seq_len)


def chunks(texts, size=7):
    return [texts[i : i + size] for i in range(0, len(texts), size)]


def direct_stream(texts) -> np.ndarray:
    tok = FakeTokenizer()
    return np.array([i for t in texts for i in tok.encode(t, add_bos=True, add_eos=True)], dtype=np.uint16)


def build(root, tokenizer_dir, texts, split="train", **kwargs):
    return build_token_cache(chunks(texts), FakeTokenizer(), identity(tokenizer_dir, split), root, **kwargs)


def test_cached_data_matches_directly_tokenized_data(tmp_path, tokenizer_dir):
    build(tmp_path, tokenizer_dir, TRAIN)
    cache = load_token_cache(tmp_path, identity(tokenizer_dir))
    expected = pack_tokens(direct_stream(TRAIN), SEQ_LEN)
    assert np.array_equal(cache.sequences, expected)


def test_documents_are_wrapped_in_bos_and_eos(tmp_path, tokenizer_dir):
    stats = build(tmp_path, tokenizer_dir, TRAIN)
    flat = np.asarray(load_token_cache(tmp_path, identity(tokenizer_dir)).sequences).reshape(-1)
    full = direct_stream(TRAIN)
    assert (full == BOS).sum() == (full == EOS).sum() == len(TRAIN) == stats.num_documents
    # In the cached stream every EOS is immediately followed by a BOS (or the stream ends).
    for pos in np.flatnonzero(flat[:-1] == EOS):
        assert flat[pos + 1] == BOS
    assert flat[0] == BOS
    # A literal BOS/EOS never appears inside a document body.
    assert set(np.unique(flat)) <= set(np.unique(full))


def test_sequences_have_expected_shape_and_counts(tmp_path, tokenizer_dir):
    stats = build(tmp_path, tokenizer_dir, TRAIN)
    cache = load_token_cache(tmp_path, identity(tokenizer_dir))
    full = direct_stream(TRAIN)
    assert cache.sequences.shape == (len(full) // SEQ_LEN, SEQ_LEN)
    assert cache.sequences.dtype == np.uint16
    assert (stats.total_tokens, stats.num_documents) == (len(full), len(TRAIN))
    assert stats.packed_tokens + stats.dropped_tokens == stats.total_tokens
    assert 0 <= stats.dropped_tokens < SEQ_LEN and cache.stats == stats


def test_train_and_validation_are_separate_and_unshuffled(tmp_path, tokenizer_dir):
    build(tmp_path, tokenizer_dir, TRAIN, split="train")
    build(tmp_path, tokenizer_dir, VALID, split="validation")
    train = load_token_cache(tmp_path, identity(tokenizer_dir, "train"))
    valid = load_token_cache(tmp_path, identity(tokenizer_dir, "validation"))

    assert cache_paths(tmp_path, "train", SEQ_LEN) != cache_paths(tmp_path, "validation", SEQ_LEN)
    assert valid.stats.num_documents == len(VALID) and train.stats.num_documents == len(TRAIN)
    assert np.array_equal(valid.sequences, pack_tokens(direct_stream(VALID), SEQ_LEN))  # original order
    assert not np.array_equal(train.sequences[: len(valid.sequences)], valid.sequences[: len(train.sequences)])
    train_array, train_meta = cache_paths(tmp_path, "train", SEQ_LEN)  # train files misplaced as validation
    valid_array, valid_meta = cache_paths(tmp_path, "validation", SEQ_LEN)
    valid_array.write_bytes(train_array.read_bytes())
    valid_meta.write_bytes(train_meta.read_bytes())
    with pytest.raises(IncompatibleCacheError, match="split"):
        load_token_cache(tmp_path, identity(tokenizer_dir, "validation"))


def test_build_is_deterministic_and_reload_is_identical(tmp_path, tokenizer_dir):
    build(tmp_path / "a", tokenizer_dir, TRAIN)
    build(tmp_path / "b", tokenizer_dir, TRAIN)  # different chunking is irrelevant too
    build_token_cache(chunks(TRAIN, 3), FakeTokenizer(), identity(tokenizer_dir), tmp_path / "c")
    files = {r: cache_paths(tmp_path / r, "train", SEQ_LEN) for r in "abc"}
    for kind in (0, 1):
        assert len({files[r][kind].read_bytes() for r in "abc"}) == 1

    first = np.array(load_token_cache(tmp_path / "a", identity(tokenizer_dir), verify_hash=True).sequences)
    again = np.array(load_token_cache(tmp_path / "a", identity(tokenizer_dir)).sequences)
    assert np.array_equal(first, again)


@pytest.mark.parametrize(
    "change",
    [
        {"tokenizer_sha256": "different"},
        {"vocab_size": 101},
        {"bos_id": 5},
        {"seq_len": 16},
        {"split_seed": 7},
        {"val_fraction": 0.5},
        {"dataset_name": "other/dataset"},
        {"format_version": 99},
    ],
)
def test_incompatible_metadata_is_rejected(tmp_path, tokenizer_dir, change):
    build(tmp_path, tokenizer_dir, TRAIN)
    wanted = dataclasses.replace(identity(tokenizer_dir), **change)
    if "seq_len" in change:  # a different length looks in a different folder; point at the real file
        array_path, meta_path = cache_paths(tmp_path, "train", SEQ_LEN)
        other_array, other_meta = cache_paths(tmp_path, "train", change["seq_len"])
        other_array.parent.mkdir()
        other_array.write_bytes(array_path.read_bytes())
        other_meta.write_bytes(meta_path.read_bytes())
    with pytest.raises(IncompatibleCacheError, match=next(iter(change))):
        load_token_cache(tmp_path, wanted)


def test_different_tokenizer_artifact_is_rejected(tmp_path, tokenizer_dir):
    build(tmp_path, tokenizer_dir, TRAIN)
    (tokenizer_dir / "metadata.json").write_text(json.dumps({"artifact_sha256": "retrained"}))
    with pytest.raises(IncompatibleCacheError, match="tokenizer_sha256"):
        load_token_cache(tmp_path, identity(tokenizer_dir))


def test_corrupted_or_truncated_cache_is_detected(tmp_path, tokenizer_dir):
    build(tmp_path, tokenizer_dir, TRAIN)
    array_path, _ = cache_paths(tmp_path, "train", SEQ_LEN)
    good = np.load(array_path)
    tampered = good.copy()
    tampered[0, 0] += 1
    np.save(array_path, tampered)
    load_token_cache(tmp_path, identity(tokenizer_dir))  # cheap check passes...
    with pytest.raises(IncompatibleCacheError, match="hash"):  # ...the hash check does not
        load_token_cache(tmp_path, identity(tokenizer_dir), verify_hash=True)
    np.save(array_path, good[:-1])
    with pytest.raises(IncompatibleCacheError, match="metadata"):
        load_token_cache(tmp_path, identity(tokenizer_dir))


def test_existing_cache_is_not_overwritten_silently(tmp_path, tokenizer_dir):
    build(tmp_path, tokenizer_dir, TRAIN)
    with pytest.raises(FileExistsError):
        build(tmp_path, tokenizer_dir, TRAIN)
    build(tmp_path, tokenizer_dir, TRAIN[:20], overwrite=True)
    assert load_token_cache(tmp_path, identity(tokenizer_dir)).stats.num_documents == 20


def test_missing_cache_and_too_little_data(tmp_path, tokenizer_dir):
    with pytest.raises(FileNotFoundError):
        load_token_cache(tmp_path, identity(tokenizer_dir))
    with pytest.raises(ValueError, match="fewer than one sequence"):
        build(tmp_path, tokenizer_dir, ["a"])
