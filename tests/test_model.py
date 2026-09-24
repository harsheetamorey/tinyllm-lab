"""Tests for the complete TinyLLM decoder model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinyllm.config import Config, ModelConfig
from tinyllm.model.block import TransformerBlock
from tinyllm.model.model import TinyLLM

CFG = ModelConfig(
    vocab_size=100, hidden_size=32, num_layers=2, num_attention_heads=4,
    ffn_hidden_size=96, max_sequence_length=16,
)
B = 2
REPO_ROOT = Path(__file__).resolve().parent.parent


def _model(seed: int = 0, cfg: ModelConfig = CFG) -> TinyLLM:
    torch.manual_seed(seed)
    return TinyLLM(cfg).eval()


def _ids(t: int = 8, seed: int = 0, cfg: ModelConfig = CFG) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (B, t), generator=torch.Generator().manual_seed(seed))


def expected_param_count(cfg: ModelConfig) -> int:
    """Parameter count derived by hand from the config."""
    d, f, v = cfg.hidden_size, cfg.ffn_hidden_size, cfg.vocab_size
    per_block = 4 * d * d + 3 * d * f + 2 * d  # q,k,v,o + gate,up,down + two norms
    return v * d + cfg.num_layers * per_block + d + d * v  # embed + blocks + final norm + lm_head


def test_token_ids_produce_logits_of_vocab_size():
    logits = _model()(_ids())
    assert logits.shape == (B, 8, CFG.vocab_size)
    assert logits.dtype == torch.float32


def test_uses_configured_number_of_existing_blocks():
    model = _model()
    assert len(model.blocks) == CFG.num_layers
    assert all(isinstance(b, TransformerBlock) for b in model.blocks)
    assert model.blocks[0] is not model.blocks[1]  # independent weights per layer


@pytest.mark.parametrize("t", [1, 2, CFG.max_sequence_length])
def test_sequence_lengths_up_to_maximum(t):
    assert _model()(_ids(t)).shape == (B, t, CFG.vocab_size)


def test_sequence_longer_than_maximum_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        _model()(_ids(CFG.max_sequence_length + 1))
    with pytest.raises(ValueError, match="expected"):
        _model()(torch.zeros(8, dtype=torch.long))


def test_gradients_flow_everywhere():
    model = TinyLLM(CFG)
    model(_ids()).pow(2).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name
    groups = {name.split(".")[0] for name, _ in model.named_parameters()}
    assert groups == {"embed", "blocks", "final_norm", "lm_head"}
    assert {n.split(".")[1] for n, _ in model.named_parameters() if n.startswith("blocks")} == {"0", "1"}


def test_no_nans_or_infs():
    model = _model()
    for t in (1, 8, CFG.max_sequence_length):
        assert torch.isfinite(model(_ids(t))).all()


def test_deterministic_in_eval_mode():
    model, ids = _model(), _ids()
    assert not model.training
    assert torch.equal(model(ids), model(ids))
    assert torch.equal(_model(seed=1)(ids), _model(seed=1)(ids))


def test_model_is_causal():
    model, ids = _model(), _ids()
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % CFG.vocab_size
    assert torch.equal(model(ids)[:, :5], model(changed)[:, :5])


def test_parameter_count_matches_hand_calculation():
    model = _model()
    assert model.num_parameters() == expected_param_count(CFG)
    assert model.num_parameters() == sum(p.numel() for p in model.parameters())


def test_parameter_count_respects_requires_grad():
    model = _model()
    total = model.num_parameters()
    model.embed.weight.requires_grad_(False)
    assert model.num_parameters() == total - CFG.vocab_size * CFG.hidden_size
    assert model.num_parameters(trainable_only=False) == total


def test_main_config_parameter_count():
    cfg = ModelConfig()  # the planned main model
    assert TinyLLM(cfg).num_parameters() == expected_param_count(cfg) == 43_655_680


def test_debug_config_builds_and_runs():
    cfg = Config.from_yaml(REPO_ROOT / "configs" / "debug.yaml").model
    model = TinyLLM(cfg).eval()
    assert model.num_parameters() == expected_param_count(cfg) == 4_522_624
    assert model(_ids(cfg.max_sequence_length, cfg=cfg)).shape == (B, cfg.max_sequence_length, cfg.vocab_size)
