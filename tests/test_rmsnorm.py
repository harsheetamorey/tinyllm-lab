"""Tests for RMSNorm."""

from __future__ import annotations

import pytest
import torch

from tinyllm.model.rmsnorm import RMSNorm

B, T, D = 2, 5, 16


def _x(seed: int = 0) -> torch.Tensor:
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


def test_shape_preserved():
    assert RMSNorm(D)(_x()).shape == (B, T, D)


def test_scale_parameter_is_learnable_and_starts_at_one():
    norm = RMSNorm(D)
    assert norm.weight.shape == (D,) and norm.weight.requires_grad
    assert torch.equal(norm.weight, torch.ones(D))
    assert [n for n, _ in norm.named_parameters()] == ["weight"]  # no bias


def test_forward_is_deterministic():
    norm = RMSNorm(D)
    x = _x()
    assert torch.equal(norm(x), norm(x))
    assert torch.equal(RMSNorm(D)(x), RMSNorm(D)(x))


def test_no_nans_on_normal_inputs():
    norm = RMSNorm(D)
    for x in [_x(), _x() * 1e4, _x() * 1e-4]:
        assert torch.isfinite(norm(x)).all()


def test_epsilon_prevents_nan_on_zero_input():
    out = RMSNorm(D)(torch.zeros(B, T, D))
    assert torch.isfinite(out).all() and (out == 0).all()


def test_gradients_flow_to_input_and_scale():
    norm = RMSNorm(D)
    x = _x().requires_grad_(True)
    norm(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0


def test_matches_manual_formula():
    eps = 1e-6
    norm = RMSNorm(D, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, D))
    x = _x(seed=1)

    rms = torch.sqrt((x**2).sum(dim=-1, keepdim=True) / D + eps)  # written out by hand
    expected = x / rms * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)


def test_output_has_unit_rms_when_scale_is_one():
    rms = RMSNorm(D)(_x()).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(B, T), rtol=1e-4, atol=1e-4)


def test_each_position_normalized_independently():
    norm = RMSNorm(D)
    x = _x()
    torch.testing.assert_close(norm(x)[:, 2], norm(x[:, 2:3])[:, 0])


def test_half_precision_keeps_input_dtype():
    out = RMSNorm(D)(_x().to(torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_wrong_last_dimension_rejected():
    with pytest.raises(ValueError):
        RMSNorm(D)(torch.randn(B, T, D + 1))
    with pytest.raises(ValueError):
        RMSNorm(0)
