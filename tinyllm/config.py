"""Central config for a TinyLLM run.

Minimal for now: just the knobs step 1 needs. We add fields (model size,
optimizer, etc.) as later phases require them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    experiment_id: str = "debug"
    seed: int = 42
    device: str = "auto"      # auto | cpu | cuda | mps
    amp_dtype: str = "auto"   # auto | bf16 | fp16 | none

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
