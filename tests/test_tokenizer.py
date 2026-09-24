"""Tests for the BPE tokenizer: training, persistence and encode/decode."""

from __future__ import annotations

import pytest

from tinyllm.config import Config, ModelConfig, TokenizerConfig
from tinyllm.data.tokenizer import BOS_TOKEN, EOS_TOKEN, BPETokenizer, Tokenizer

CORPUS = [
    "One day, a little girl named Lily found a needle in her room.",
    "Once upon a time, there was a little car named Beep.",
    "The cherry tree was happy because it had many friends now.",
] * 50
VOCAB = 320


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=VOCAB)


def test_is_a_tokenizer(tok):
    assert isinstance(tok, Tokenizer)


def test_vocab_size_and_special_ids(tok):
    assert tok.vocab_size == VOCAB
    assert (tok.bos_id, tok.eos_id) == (0, 1)


def test_encode_decode_roundtrip(tok):
    text = "Once upon a time, Lily found a needle."
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unseen_and_unicode(tok):
    # Byte-level BPE has no unknown token: any string must round-trip.
    for text in ["zebra quokka!", "héllo wörld 你好 🙂", "  spaces\n\nand\nnewlines\t"]:
        assert tok.decode(tok.encode(text)) == text


def test_bos_eos_added_only_on_request(tok):
    plain = tok.encode("Lily")
    assert tok.bos_id not in plain and tok.eos_id not in plain
    assert tok.encode("Lily", add_bos=True) == [tok.bos_id, *plain]
    assert tok.encode("Lily", add_eos=True) == [*plain, tok.eos_id]
    assert tok.encode("Lily", add_bos=True, add_eos=True) == [tok.bos_id, *plain, tok.eos_id]


def test_decode_special_tokens(tok):
    ids = tok.encode("Lily", add_bos=True, add_eos=True)
    assert tok.decode(ids) == "Lily"
    assert tok.decode(ids, skip_special_tokens=False) == f"{BOS_TOKEN}Lily{EOS_TOKEN}"


def test_special_token_text_is_not_parsed_from_input(tok):
    # Story text containing "<eos>" must not inject a real EOS id.
    text = f"a {BOS_TOKEN} b {EOS_TOKEN}"
    ids = tok.encode(text)
    assert tok.bos_id not in ids and tok.eos_id not in ids
    assert tok.decode(ids) == text


def test_save_load_gives_identical_tokenizer(tok, tmp_path):
    path = tok.save(tmp_path / "tok")
    assert path.is_file()
    loaded = BPETokenizer.load(tmp_path / "tok")
    assert loaded.vocab_size == tok.vocab_size
    assert (loaded.bos_id, loaded.eos_id) == (tok.bos_id, tok.eos_id)
    for text in CORPUS[:3] + ["héllo 🙂"]:
        assert loaded.encode(text, add_bos=True, add_eos=True) == tok.encode(
            text, add_bos=True, add_eos=True
        )


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=VOCAB)
    b = BPETokenizer.train(CORPUS, vocab_size=VOCAB)
    assert a.encode(CORPUS[0]) == b.encode(CORPUS[0])


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "nope")


def test_config_tokenizer_defaults_and_vocab_consistency():
    cfg = Config()
    assert cfg.tokenizer.vocab_size == cfg.model.vocab_size == 16000
    with pytest.raises(ValueError, match="must equal"):
        Config(model=ModelConfig(vocab_size=8000), tokenizer=TokenizerConfig(vocab_size=16000))


def test_encode_batch_matches_encode(tmp_path):
    tok = BPETokenizer.train(["hello world story", "another tiny story here"] * 20, vocab_size=300)
    texts = ["hello world", "", "a <bos> literal <eos> inside", "another story"]
    for add_bos, add_eos in [(False, False), (True, True), (True, False)]:
        assert tok.encode_batch(texts, add_bos=add_bos, add_eos=add_eos) == [
            tok.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts
        ]
