"""Tests for loading Config / ModelConfig from YAML."""

from __future__ import annotations

import pytest
import yaml

from tinyllm.config import Config, ModelConfig


def test_model_defaults_are_main_tinyllm():
    m = Config().model
    assert (m.vocab_size, m.hidden_size, m.num_layers) == (16000, 512, 8)
    assert (m.num_attention_heads, m.ffn_hidden_size, m.max_sequence_length) == (8, 1536, 512)
    assert m.head_dim == 64


def test_yaml_model_section_overrides_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("experiment_id: x\nmodel:\n  num_layers: 2\n  hidden_size: 128\n")
    cfg = Config.from_yaml(path)
    assert cfg.model.num_layers == 2
    assert cfg.model.hidden_size == 128
    assert cfg.model.vocab_size == 16000  # untouched fields keep defaults


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(Config().to_dict()))
    assert Config.from_yaml(path) == Config()


def test_unknown_model_key_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model:\n  hiden_size: 512\n")
    with pytest.raises(ValueError, match="model.hiden_size"):
        Config.from_yaml(path)


@pytest.mark.parametrize(
    "kwargs",
    [{"hidden_size": 500}, {"num_layers": 0}, {"vocab_size": -1}, {"ffn_hidden_size": 1.5}],
)
def test_invalid_model_values_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
