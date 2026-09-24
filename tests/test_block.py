"""Tests for the pre-norm Transformer block."""

from __future__ import annotations

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.model.attention import CausalSelfAttention
from tinyllm.model.block import TransformerBlock
from tinyllm.model.rmsnorm import RMSNorm
from tinyllm.model.swiglu import SwiGLU

B, T, D, H, FF, MAX_T = 2, 6, 32, 4, 96, 16


def _block(seed: int = 0) -> TransformerBlock:
    torch.manual_seed(seed)
    return TransformerBlock(D, H, FF, MAX_T).eval()


def _x(seed: int = 0, t: int = T) -> torch.Tensor:
    return torch.randn(B, t, D, generator=torch.Generator().manual_seed(seed))


def test_shape_preserved():
    assert _block()(_x()).shape == (B, T, D)


def test_uses_existing_components():
    block = _block()
    assert isinstance(block.attn, CausalSelfAttention)
    assert isinstance(block.ffn, SwiGLU)
    assert isinstance(block.attn_norm, RMSNorm) and isinstance(block.ffn_norm, RMSNorm)
    assert block.attn_norm is not block.ffn_norm  # separate learnable scales


def test_forward_matches_pre_norm_formula():
    block, x = _block(), _x()
    h = x + block.attn(block.attn_norm(x))
    expected = h + block.ffn(block.ffn_norm(h))
    torch.testing.assert_close(block(x), expected)


def test_residual_paths_pass_input_through_when_sublayers_output_zero():
    block, x = _block(), _x()
    with torch.no_grad():
        block.attn.o_proj.weight.zero_()
        block.ffn.down_proj.weight.zero_()
    torch.testing.assert_close(block(x), x)


def test_backward_and_gradients_reach_all_submodules():
    block = TransformerBlock(D, H, FF, MAX_T)
    x = _x().requires_grad_(True)
    block(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name
    groups = {name.split(".")[0] for name, _ in block.named_parameters()}
    assert groups == {"attn_norm", "attn", "ffn_norm", "ffn"}


def test_no_nans_or_infs():
    block = _block()
    for x in [_x(), _x() * 100, _x(t=MAX_T)]:
        assert torch.isfinite(block(x)).all()


def test_deterministic_in_eval_mode():
    block, x = _block(), _x()
    assert not block.training
    assert torch.equal(block(x), block(x))
    assert torch.equal(_block(seed=1)(x), _block(seed=1)(x))


def test_block_stays_causal():
    block, x = _block(), _x()
    changed = x.clone()
    changed[:, 4:] = torch.randn(B, T - 4, D)
    assert torch.equal(block(x)[:, :4], block(changed)[:, :4])


def test_from_config_matches_model_config():
    cfg = ModelConfig()
    block = TransformerBlock.from_config(cfg)
    assert block.attn.num_heads == 8 and block.ffn.ffn_hidden_size == 1536
    assert block(torch.randn(1, 4, 512)).shape == (1, 4, 512)
