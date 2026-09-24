"""Tests for the next-token loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tinyllm.config import ModelConfig
from tinyllm.model.model import TinyLLM
from tinyllm.training import loss as loss_module
from tinyllm.training.loss import next_token_loss, shift_logits_and_labels

CFG = ModelConfig(
    vocab_size=50, hidden_size=32, num_layers=2, num_attention_heads=4,
    ffn_hidden_size=96, max_sequence_length=16,
)


def _tokens(b: int = 2, t: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.randint(1, CFG.vocab_size, (b, t), generator=torch.Generator().manual_seed(seed))


def test_shift_aligns_logits_and_labels():
    tokens = torch.tensor([[10, 11, 12, 13, 14]])  # A B C D E
    logits = torch.arange(5 * 20, dtype=torch.float32).reshape(1, 5, 20)
    shifted_logits, labels = shift_logits_and_labels(logits, tokens)
    assert shifted_logits.shape == (1, 4, 20) and labels.shape == (1, 4)
    assert torch.equal(shifted_logits, logits[:, :4])  # positions A B C D
    assert labels.tolist() == [[11, 12, 13, 14]]  # targets B C D E


def test_flattened_shapes_passed_to_cross_entropy(monkeypatch):
    seen = {}
    real = F.cross_entropy

    def spy(inp, target, **kwargs):
        seen["input"], seen["target"] = tuple(inp.shape), tuple(target.shape)
        return real(inp, target, **kwargs)

    monkeypatch.setattr(loss_module.F, "cross_entropy", spy)
    B, T, V = 3, 7, 11
    next_token_loss(torch.randn(B, T, V), torch.randint(0, V, (B, T)))
    assert seen == {"input": (B * (T - 1), V), "target": (B * (T - 1),)}


def test_loss_is_finite_scalar():
    logits = torch.randn(2, 6, CFG.vocab_size)
    loss = next_token_loss(logits, _tokens())
    assert loss.shape == () and torch.isfinite(loss)


def test_gradients_flow_back_through_model():
    model = TinyLLM(CFG)
    tokens = _tokens()
    next_token_loss(model(tokens), tokens).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name


def test_manual_logits_give_expected_loss():
    tokens = torch.tensor([[0, 2, 1]])
    logits = torch.tensor([[
        [1.0, 2.0, 3.0],       # position 0 predicts tokens[1] = 2
        [0.0, 0.0, 0.0],       # position 1 predicts tokens[2] = 1 (uniform)
        [100.0, -100.0, 5.0],  # position 2 predicts nothing: must be ignored
    ]])
    loss_pos0 = math.log(math.e**1 + math.e**2 + math.e**3) - 3.0  # -log softmax([1,2,3])[2]
    loss_pos1 = math.log(3.0)
    expected = (loss_pos0 + loss_pos1) / 2
    torch.testing.assert_close(next_token_loss(logits, tokens), torch.tensor(expected), rtol=1e-6, atol=1e-6)


def test_sequence_length_two_uses_only_first_position():
    tokens = torch.tensor([[4, 1]])
    logits = torch.randn(1, 2, 5)
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[1]  # only position 0 -> token 1
    torch.testing.assert_close(next_token_loss(logits, tokens), expected)
    shifted, labels = shift_logits_and_labels(logits, tokens)
    assert shifted.shape == (1, 1, 5) and labels.shape == (1, 1)


def test_sequence_length_one_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        next_token_loss(torch.randn(1, 1, 5), torch.tensor([[3]]))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="expected"):
        next_token_loss(torch.randn(2, 5, 7), torch.zeros(2, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="expected"):
        next_token_loss(torch.randn(2, 5), torch.zeros(2, 5, dtype=torch.long))


def test_padding_targets_are_ignored():
    PAD = 0
    logits = torch.randn(2, 5, 8)
    tokens = torch.tensor([[3, 4, 5, PAD, PAD], [6, 7, 1, 2, 5]])
    with_pad = next_token_loss(logits, tokens, pad_id=PAD)

    per_target = -torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, tokens[:, 1:, None])[..., 0]
    keep = tokens[:, 1:] != PAD
    torch.testing.assert_close(with_pad, per_target[keep].mean())
    assert not torch.allclose(with_pad, next_token_loss(logits, tokens))  # padding really is excluded


def test_all_padding_gives_zero_not_nan():
    tokens = torch.zeros(2, 4, dtype=torch.long)
    loss = next_token_loss(torch.randn(2, 4, 8, requires_grad=True), tokens, pad_id=0)
    assert loss.item() == 0.0
    loss.backward()  # still differentiable


def test_feeding_all_tokens_equals_feeding_all_but_last():
    # Causality means dropping the last input token gives the same predictions,
    # so shifting logits is equivalent to the "input [:-1], targets [1:]" view.
    model = TinyLLM(CFG).eval()
    tokens = _tokens()
    with torch.no_grad():
        full = next_token_loss(model(tokens), tokens)
        short_logits = model(tokens[:, :-1])
        alt = F.cross_entropy(short_logits.reshape(-1, CFG.vocab_size), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(full, alt, rtol=1e-5, atol=1e-6)
