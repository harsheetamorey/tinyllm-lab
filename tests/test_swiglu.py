"""Tests for the SwiGLU feed-forward network."""

from __future__ import annotations

import math

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.model.swiglu import SwiGLU

B, T, D, FF = 2, 5, 16, 48


def _ffn(seed: int = 0) -> SwiGLU:
    torch.manual_seed(seed)
    return SwiGLU(D, FF)


def _x(seed: int = 0) -> torch.Tensor:
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


def test_output_shape():
    assert _ffn()(_x()).shape == (B, T, D)


def test_hidden_dimension_is_ffn_hidden_size():
    ffn = _ffn()
    assert ffn.gate_proj.weight.shape == (FF, D)
    assert ffn.up_proj.weight.shape == (FF, D)
    assert ffn.down_proj.weight.shape == (D, FF)
    # Intermediate activation really is [B, T, F].
    seen = []
    ffn.down_proj.register_forward_hook(lambda m, inp, out: seen.append(inp[0].shape))
    ffn(_x())
    assert seen == [(B, T, FF)]


def test_deterministic():
    x = _x()
    ffn = _ffn()
    assert torch.equal(ffn(x), ffn(x))
    assert torch.equal(_ffn(seed=1)(x), _ffn(seed=1)(x))


def test_no_nans_or_infs():
    ffn = _ffn()
    for x in [_x(), _x() * 100, _x() * 1e-4]:
        assert torch.isfinite(ffn(x)).all()


def test_gradients_flow_to_all_projections_and_input():
    ffn = _ffn()
    x = _x().requires_grad_(True)
    ffn(x).pow(2).sum().backward()
    for name in ("gate_proj", "up_proj", "down_proj"):
        grad = getattr(ffn, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_matches_manual_swiglu_on_tiny_tensor():
    ffn = SwiGLU(hidden_size=2, ffn_hidden_size=3)
    with torch.no_grad():
        ffn.gate_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
        ffn.up_proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
        ffn.down_proj.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]))
    x = torch.tensor([[[1.0, 2.0]]])  # [B=1, T=1, D=2]

    silu = lambda z: z / (1 + math.exp(-z))
    gate = [1.0, 2.0, -1.0]          # x @ gate_proj^T
    up = [2.0, 6.0, 3.0]             # x @ up_proj^T
    hidden = [silu(g) * u for g, u in zip(gate, up)]
    expected = torch.tensor([[[hidden[0] + hidden[2], hidden[1] - hidden[2]]]])

    torch.testing.assert_close(ffn(x), expected, rtol=1e-5, atol=1e-6)


def test_no_bias_and_three_weight_matrices():
    assert sorted(n for n, _ in _ffn().named_parameters()) == [
        "down_proj.weight", "gate_proj.weight", "up_proj.weight",
    ]


def test_wrong_input_dimension_rejected():
    with pytest.raises(ValueError, match="expected"):
        _ffn()(torch.randn(B, T, D + 1))
    with pytest.raises(ValueError):
        SwiGLU(0, FF)


def test_from_config_uses_ffn_hidden_size():
    cfg = ModelConfig()
    ffn = SwiGLU.from_config(cfg)
    assert (ffn.hidden_size, ffn.ffn_hidden_size) == (512, 1536)
    assert ffn(torch.randn(1, 4, 512)).shape == (1, 4, 512)
