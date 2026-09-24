"""Tests for causal multi-head self-attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tinyllm.config import ModelConfig
from tinyllm.model.attention import CausalSelfAttention

B, T, D, H = 2, 6, 32, 4
DH = D // H
MAX_T = 16


def _attn(seed: int = 0) -> CausalSelfAttention:
    torch.manual_seed(seed)
    return CausalSelfAttention(D, H, MAX_T).eval()


def _x(seed: int = 0, t: int = T) -> torch.Tensor:
    return torch.randn(B, t, D, generator=torch.Generator().manual_seed(seed))


def test_output_shape():
    assert _attn()(_x()).shape == (B, T, D)


def test_qkv_shapes():
    q, k, v = _attn().project_qkv(_x())
    assert q.shape == k.shape == v.shape == (B, H, T, DH)


def test_attention_probs_shape_and_rows_sum_to_one():
    attn = _attn()
    q, k, _ = attn.project_qkv(_x())
    probs = attn.attention_probs(q, k)
    assert probs.shape == (B, H, T, T)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(B, H, T))


def test_causal_mask_zeroes_future_weights():
    attn = _attn()
    q, k, _ = attn.project_qkv(_x())
    probs = attn.attention_probs(q, k)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (probs[..., future] == 0).all()
    assert (probs[..., ~future] > 0).all()  # every past/current position gets some weight
    assert torch.equal(probs[..., 0, :], F.one_hot(torch.tensor(0), T).expand(B, H, T).float())


def test_changing_future_token_does_not_change_earlier_outputs():
    attn = _attn()
    x = _x()
    changed = x.clone()
    changed[:, 4:] = torch.randn(B, T - 4, D)  # tokens 4.. are the "future" of token 3
    out, out_changed = attn(x), attn(changed)
    assert torch.equal(out[:, :4], out_changed[:, :4])
    assert not torch.allclose(out[:, 4:], out_changed[:, 4:])


def test_changing_past_token_does_change_later_outputs():
    attn = _attn()
    x = _x()
    changed = x.clone()
    changed[:, 0] = torch.randn(B, D)
    assert not torch.allclose(attn(x)[:, 1:], attn(changed)[:, 1:])


def test_no_nans_or_infs():
    attn = _attn()
    for x in [_x(), _x() * 100, _x(t=MAX_T)]:
        assert torch.isfinite(attn(x)).all()


def test_gradients_flow_through_qkv_and_output_projection():
    attn = CausalSelfAttention(D, H, MAX_T)
    x = _x().requires_grad_(True)
    attn(x).pow(2).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        grad = getattr(attn, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_deterministic_in_eval_mode():
    attn = _attn()
    x = _x()
    assert torch.equal(attn(x), attn(x))
    assert torch.equal(_attn(seed=1)(x), _attn(seed=1)(x))


def test_matches_reference_implementation():
    # Same projections and RoPE, but attention done by PyTorch's fused causal SDPA.
    attn = _attn()
    x = _x()
    q, k, v = attn.project_qkv(x)
    q, k = attn.rope(q), attn.rope(k)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.o_proj(ref.transpose(1, 2).reshape(B, T, D))
    torch.testing.assert_close(attn(x), ref, rtol=1e-4, atol=1e-5)


def test_rope_is_applied_to_q_and_k_but_not_v():
    attn = _attn()
    x = _x()
    q, k, v = attn.project_qkv(x)
    assert not torch.allclose(attn.rope(q), q)
    assert not torch.allclose(attn.rope(k), k)
    # Rebuilding forward with only q and k rotated must reproduce it exactly.
    probs = attn.attention_probs(attn.rope(q), attn.rope(k))
    out = attn.o_proj((probs @ v).transpose(1, 2).reshape(B, T, D))
    torch.testing.assert_close(attn(x), out)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(30, 4, MAX_T)
    attn = _attn()
    with pytest.raises(ValueError, match="expected"):
        attn(torch.randn(B, T, D + 1))
    with pytest.raises(ValueError, match="expected"):
        attn(torch.randn(T, D))
    with pytest.raises(ValueError, match="exceeds"):
        attn(_x(t=MAX_T + 1))


def test_from_config_matches_model_config():
    cfg = ModelConfig()
    attn = CausalSelfAttention.from_config(cfg)
    assert (attn.hidden_size, attn.num_heads, attn.head_dim) == (512, 8, 64)
    assert attn(torch.randn(1, 4, 512)).shape == (1, 4, 512)


def test_no_bias_and_only_four_projection_weights():
    assert sorted(n for n, _ in _attn().named_parameters()) == [
        "k_proj.weight", "o_proj.weight", "q_proj.weight", "v_proj.weight",
    ]
