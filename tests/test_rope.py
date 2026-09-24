"""Tests for rotary positional embeddings."""

from __future__ import annotations

import math

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.model.rope import RotaryEmbedding

B, H, T, DH = 2, 4, 8, 16


def _x(seed: int = 0, t: int = T) -> torch.Tensor:
    return torch.randn(B, H, t, DH, generator=torch.Generator().manual_seed(seed))


def _same_vector_everywhere(t: int = T) -> torch.Tensor:
    """The same vector at every position, so any difference comes from position alone."""
    vec = torch.randn(DH, generator=torch.Generator().manual_seed(3))
    return vec.expand(B, H, t, DH).clone()


def test_shape_preserved():
    assert RotaryEmbedding(DH, max_seq_len=32)(_x()).shape == (B, H, T, DH)


def test_deterministic():
    rope = RotaryEmbedding(DH, max_seq_len=32)
    x = _x()
    assert torch.equal(rope(x), rope(x))
    assert torch.equal(RotaryEmbedding(DH, 32)(x), RotaryEmbedding(DH, 32)(x))


def test_same_vector_same_position_gives_same_output_across_batch_and_heads():
    out = RotaryEmbedding(DH, 32)(_same_vector_everywhere())
    assert torch.equal(out[0, 0], out[1, 3])  # same position t -> identical rotation


def test_different_positions_give_different_values():
    out = RotaryEmbedding(DH, 32)(_same_vector_everywhere())
    for t in range(1, T):
        assert not torch.allclose(out[0, 0, 0], out[0, 0, t])
    assert not torch.allclose(out[0, 0, 1], out[0, 0, 2])


def test_position_zero_is_unrotated():
    x = _x()
    torch.testing.assert_close(RotaryEmbedding(DH, 32)(x)[:, :, 0], x[:, :, 0])


def test_no_nans_or_infs():
    rope = RotaryEmbedding(DH, max_seq_len=512)
    for x in [_x(), _x() * 1e4, _x(t=512)]:
        assert torch.isfinite(rope(x)).all()


def test_rotation_preserves_vector_magnitude():
    x = _x(t=32)
    out = RotaryEmbedding(DH, 32)(x)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_matches_manual_rotation():
    base = 10000.0
    x = _x(seed=1)
    out = RotaryEmbedding(DH, 32, base=base)(x)
    half = DH // 2
    for pos in (0, 3, 7):
        for i in range(half):
            theta = pos * base ** (-2 * i / DH)
            a, b = x[0, 0, pos, i], x[0, 0, pos, i + half]
            torch.testing.assert_close(out[0, 0, pos, i], a * math.cos(theta) - b * math.sin(theta))
            torch.testing.assert_close(out[0, 0, pos, i + half], a * math.sin(theta) + b * math.cos(theta))


def test_attention_score_depends_only_on_relative_position():
    # The defining RoPE property: <rot(q, m), rot(k, n)> depends on m - n only.
    rope = RotaryEmbedding(DH, 32)
    q = _same_vector_everywhere(t=16)
    k = torch.randn(DH, generator=torch.Generator().manual_seed(4)).expand(B, H, 16, DH).clone()
    rq, rk = rope(q), rope(k)
    score = lambda m, n: (rq[0, 0, m] * rk[0, 0, n]).sum()
    torch.testing.assert_close(score(5, 2), score(9, 6), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(score(3, 3), score(12, 12), rtol=1e-4, atol=1e-4)


def test_gradient_flows_to_input():
    x = _x().requires_grad_(True)
    RotaryEmbedding(DH, 32)(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_odd_head_dim_fails_clearly():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(15, max_seq_len=32)


def test_invalid_arguments_and_inputs_rejected():
    rope = RotaryEmbedding(DH, max_seq_len=8)
    with pytest.raises(ValueError, match="exceeds"):
        rope(_x(t=9))
    with pytest.raises(ValueError, match="expected"):
        rope(torch.randn(B, T, DH))  # missing head dimension
    with pytest.raises(ValueError, match="expected"):
        rope(torch.randn(B, H, T, DH + 2))
    with pytest.raises(ValueError):
        RotaryEmbedding(DH, max_seq_len=0)


def test_from_config_uses_head_dim_and_sequence_length():
    cfg = ModelConfig()
    rope = RotaryEmbedding.from_config(cfg)
    assert (rope.head_dim, rope.max_seq_len) == (64, 512)
    assert rope(torch.randn(1, cfg.num_attention_heads, 512, 64)).shape == (1, 8, 512, 64)


def test_tables_are_not_saved_in_checkpoints():
    assert list(RotaryEmbedding(DH, 32).state_dict()) == []


def test_half_precision_keeps_input_dtype():
    out = RotaryEmbedding(DH, 32)(_x().to(torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
