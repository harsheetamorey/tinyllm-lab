"""Central config for a TinyLLM run.

``Config`` holds run-level knobs; ``ModelConfig`` holds the architecture and
``DataConfig`` the pretraining data source and split, ``TokenizerConfig``
the BPE tokenizer and ``TrainConfig`` the pretraining loop.
We add fields (optimizer, data, etc.) as later phases require them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the decoder-only transformer.

    Defaults are the main TinyLLM model.
    """

    vocab_size: int = 16000          # number of tokenizer tokens (embedding rows)
    hidden_size: int = 512           # width of the residual stream / embeddings
    num_layers: int = 8              # number of transformer blocks
    num_attention_heads: int = 8     # heads per attention layer
    ffn_hidden_size: int = 1536      # inner width of the SwiGLU feed-forward
    max_sequence_length: int = 512   # longest context the model is trained on

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"model.{f.name} must be a positive int, got {value!r}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Size of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        _reject_unknown(cls, data, prefix="model.")
        return cls(**data)


@dataclass(frozen=True)
class DataConfig:
    """Pretraining data source and its fixed train/validation split.

    ``split_seed`` is separate from the run seed so every experiment sees the
    same split even when the training seed changes.
    """

    dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"
    val_fraction: float = 0.02
    split_seed: int = 42
    max_train_stories: int | None = None  # cap for quick runs; None = the whole split
    max_val_stories: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"data.val_fraction must be in (0, 1), got {self.val_fraction!r}")
        for name in ("max_train_stories", "max_val_stories"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"data.{name} must be a positive int or null, got {value!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        _reject_unknown(cls, data, prefix="data.")
        return cls(**data)


@dataclass(frozen=True)
class TokenizerConfig:
    """BPE tokenizer settings.

    ``vocab_size`` counts the special tokens and must equal
    ``model.vocab_size``. The trained tokenizer is saved to ``dir`` and that
    same directory is loaded for pretraining, SFT and evaluation.
    """

    vocab_size: int = 16000
    dir: str = "artifacts/tokenizer"

    def __post_init__(self) -> None:
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError(f"tokenizer.vocab_size must be a positive int, got {self.vocab_size!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerConfig":
        _reject_unknown(cls, data, prefix="tokenizer.")
        return cls(**data)


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, schedule, batching and checkpoint/logging cadence.

    Batch size in tokens = ``micro_batch_size * grad_accum_steps *
    model.max_sequence_length``. All ``*_steps`` and ``*_every`` counts are
    optimizer steps, not micro-batches.
    """

    max_steps: int = 1000
    micro_batch_size: int = 32       # sequences per forward/backward pass
    grad_accum_steps: int = 1        # micro-batches per optimizer step
    learning_rate: float = 3e-4      # peak LR, reached at the end of warmup
    min_learning_rate: float = 3e-5  # LR the cosine decays to
    warmup_steps: int = 100
    weight_decay: float = 0.1        # applied to matrices only, not norms/biases
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0           # max global grad norm; 0 disables clipping
    log_every: int = 10
    eval_every: int = 100
    eval_batches: int = 20           # fixed validation batches per evaluation
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        for name in ("max_steps", "micro_batch_size", "grad_accum_steps", "log_every",
                     "eval_every", "eval_batches", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"train.{name} must be a positive int, got {value!r}")
        if not isinstance(self.warmup_steps, int) or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError(
                f"train.warmup_steps must be an int in [0, max_steps), got {self.warmup_steps!r}"
            )
        if not 0.0 <= self.min_learning_rate <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("train needs 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("train.weight_decay and train.grad_clip must be >= 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        _reject_unknown(cls, data, prefix="train.")
        return cls(**data)


@dataclass
class Config:
    experiment_id: str = "debug"
    seed: int = 42
    device: str = "auto"      # auto | cpu | cuda | mps
    amp_dtype: str = "auto"   # auto | bf16 | fp16 | none
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.tokenizer.vocab_size != self.model.vocab_size:
            raise ValueError(
                f"tokenizer.vocab_size ({self.tokenizer.vocab_size}) must equal "
                f"model.vocab_size ({self.model.vocab_size})"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        _reject_unknown(cls, data)
        data = dict(data)
        if "model" in data:
            data["model"] = ModelConfig.from_dict(data["model"] or {})
        if "data" in data:
            data["data"] = DataConfig.from_dict(data["data"] or {})
        if "tokenizer" in data:
            data["tokenizer"] = TokenizerConfig.from_dict(data["tokenizer"] or {})
        if "train" in data:
            data["train"] = TrainConfig.from_dict(data["train"] or {})
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def _reject_unknown(cls: type, data: dict[str, Any], prefix: str = "") -> None:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(prefix + k for k in unknown)}")
