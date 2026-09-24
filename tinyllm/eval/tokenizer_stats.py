"""Tokenizer evaluation: size, compression statistics and encode/decode examples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tinyllm.data.tokenizer import Tokenizer

EXAMPLE_TEXTS = (
    "Once upon a time...",
    "Machine learning is...",
    "Hello, how are you?",
)

UNK_NOTE = "byte-level BPE has no unknown token: every string is representable"


@dataclass(frozen=True)
class TokenizerStats:
    """Corpus statistics; token counts exclude BOS/EOS."""

    vocab_size: int
    num_documents: int
    total_characters: int
    total_tokens: int
    avg_tokens_per_document: float
    avg_characters_per_token: float
    unknown_token_frequency: float | None  # None: tokenizer has no unknown token
    unknown_token_note: str


@dataclass(frozen=True)
class EncodeExample:
    text: str
    ids: list[int]
    tokens: list[str]
    decoded: str


def compute_stats(tokenizer: Tokenizer, texts: Iterable[str]) -> TokenizerStats:
    """Encode every text (without BOS/EOS) and aggregate the statistics."""
    num_docs = total_chars = total_tokens = 0
    for text in texts:
        num_docs += 1
        total_chars += len(text)
        total_tokens += len(tokenizer.encode(text))
    if num_docs == 0 or total_tokens == 0:
        raise ValueError("cannot compute statistics on an empty corpus")
    return TokenizerStats(
        vocab_size=tokenizer.vocab_size,
        num_documents=num_docs,
        total_characters=total_chars,
        total_tokens=total_tokens,
        avg_tokens_per_document=total_tokens / num_docs,
        avg_characters_per_token=total_chars / total_tokens,
        unknown_token_frequency=None,
        unknown_token_note=UNK_NOTE,
    )


def encode_examples(tokenizer: Tokenizer, texts: Iterable[str] = EXAMPLE_TEXTS) -> list[EncodeExample]:
    """Encode then decode each text, keeping the per-token strings for display."""
    examples = []
    for text in texts:
        ids = tokenizer.encode(text)
        examples.append(
            EncodeExample(
                text=text,
                ids=ids,
                tokens=[tokenizer.decode([i], skip_special_tokens=False) for i in ids],
                decoded=tokenizer.decode(ids),
            )
        )
    return examples


def save_report(
    path: str | Path,
    stats: TokenizerStats,
    examples: list[EncodeExample],
    corpus: str,
) -> Path:
    """Write the statistics and examples as JSON; ``corpus`` says what was measured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus": corpus,
        "stats": asdict(stats),
        "examples": [asdict(e) for e in examples],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path
