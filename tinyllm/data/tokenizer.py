"""Byte-level BPE tokenizer with explicit BOS/EOS tokens.

One trained tokenizer (``tokenizer.json`` in a directory) is shared by
pretraining, SFT and evaluation: always obtain it via :meth:`BPETokenizer.load`.

BOS/EOS are never added implicitly. Callers choose per call, so pretraining
can wrap each story and SFT can build its own sequence layout.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (BOS_TOKEN, EOS_TOKEN)
TOKENIZER_FILE = "tokenizer.json"
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


class Tokenizer(ABC):
    """Contract every tokenizer used by the data, training and eval code follows."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total number of token ids, special tokens included."""

    @property
    @abstractmethod
    def bos_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Turn ``text`` into token ids, optionally wrapped in BOS / EOS."""

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """Encode many texts; same result as calling :meth:`encode` on each (subclasses may be faster)."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    @abstractmethod
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Turn token ids back into text."""

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Persist the tokenizer under ``directory``."""


class BPETokenizer(Tokenizer):
    """Byte-level BPE (GPT-2 style): any string round-trips, there is no unknown token."""

    def __init__(self, tokenizer: _HFTokenizer) -> None:
        self._tok = tokenizer
        self._bos_id = self._require_id(BOS_TOKEN)
        self._eos_id = self._require_id(EOS_TOKEN)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPETokenizer":
        """Train a new BPE tokenizer on ``texts``.

        Pass only training-split text. ``vocab_size`` includes the special
        tokens, which take ids 0 (BOS) and 1 (EOS).
        """
        tok = _HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save`."""
        path = Path(directory) / TOKENIZER_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no tokenizer at {path}; train it first")
        return cls(_HFTokenizer.from_file(str(path)))

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TOKENIZER_FILE
        self._tok.save(str(path))
        return path

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._encode_text(text)
        if add_bos:
            ids = [self._bos_id, *ids]
        if add_eos:
            ids = [*ids, self._eos_id]
        return ids

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """Parallel batch encode; texts containing literal "<bos>"/"<eos>" take the safe per-text path."""
        ids: list[list[int] | None] = [None] * len(texts)
        plain = [i for i, text in enumerate(texts) if not _SPECIAL_RE.search(text)]
        encodings = self._tok.encode_batch([texts[i] for i in plain], add_special_tokens=False)
        for i, encoding in zip(plain, encodings):
            ids[i] = encoding.ids
        for i, text in enumerate(texts):
            if ids[i] is None:
                ids[i] = self._encode_text(text)
        return [
            [*([self._bos_id] if add_bos else []), *doc, *([self._eos_id] if add_eos else [])]
            for doc in ids
        ]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def _encode_text(self, text: str) -> list[int]:
        """Encode ``text`` so literal "<bos>" / "<eos>" in it stay ordinary text.

        The underlying tokenizer would turn them into the real special ids,
        letting data silently inject BOS/EOS; only ``encode(add_*)`` may do that.
        """
        if not _SPECIAL_RE.search(text):
            return self._tok.encode(text, add_special_tokens=False).ids
        ids: list[int] = []
        for part in _SPECIAL_RE.split(text):
            if part in SPECIAL_TOKENS:
                # Split so the pieces no longer match the special token.
                ids += self._plain_ids(part[:1]) + self._plain_ids(part[1:])
            elif part:
                ids += self._plain_ids(part)
        return ids

    def _plain_ids(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def _require_id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return token_id
