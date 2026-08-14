from __future__ import annotations

import inspect

import torch
from torch import nn

from methods.liveedit_med.cached_suffix import forward_suffix_hidden
from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook
from methods.liveedit_med.source_training_continuation import (
    SourceTrainingContinuationMode,
    forward_source_training_hidden,
)
from methods.liveedit_med.trainer import LiveEditMedicalConfig


class ToyLayer(nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.index = index
        self.last_kwargs = None

    def forward(self, hidden, **kwargs):
        self.last_kwargs = kwargs
        return (hidden + self.linear(hidden) * (self.index + 1) / 100.0,)


class ToyRotary(nn.Module):
    def forward(self, hidden, position_ids):
        shape = (*position_ids.shape, hidden.shape[-1])
        return torch.zeros(shape, device=hidden.device), torch.ones(shape, device=hidden.device)


class ToyCore(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(ToyLayer(dim, index) for index in range(32))
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = ToyRotary()
        self.config = type("Config", (), {"num_hidden_layers": 32})()

    def _update_causal_mask(self, attention_mask, hidden, cache_position, _past, _attn):
        return attention_mask[:, None, None, :].to(hidden.dtype)


class ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyCore()
        self.lm_head = nn.Linear(8, 11, bias=False)


def manual_layers(model, hidden, start):
    for index in range(start, 32):
        hidden = hidden + model.model.layers[index].linear(hidden) * (index + 1) / 100.0
    return model.model.norm(hidden)


def inputs():
    torch.manual_seed(11)
    return torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.long)


def test_strict_reapplies_layer21_then_adds_residual():
    model = ToyLanguageModel()
    clean, residual, attention = inputs()
    expected = model.model.layers[21](clean)[0] + residual
    expected = manual_layers(model, expected, 22)
    actual = forward_source_training_hidden(
        model, clean, attention, residual=residual,
        mode=SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21,
    )
    assert torch.equal(actual, expected)


def test_corrected_adds_residual_then_starts_layer22():
    model = ToyLanguageModel()
    clean, residual, attention = inputs()
    expected = manual_layers(model, clean + residual, 22)
    actual = forward_source_training_hidden(
        model, clean, attention, residual=residual,
        mode=SourceTrainingContinuationMode.CORRECTED_SEMANTICS_CONTINUE_LAYER22,
    )
    assert torch.equal(actual, expected)
    assert model.model.layers[21].last_kwargs is None


def test_legacy_corrected_suffix_wrapper_is_exact():
    model = ToyLanguageModel()
    edited, _residual, attention = inputs()
    assert torch.equal(forward_suffix_hidden(model, edited, attention), manual_layers(model, edited, 22))


def test_decoder_layer_receives_full_hf_continuation_kwargs():
    model = ToyLanguageModel()
    clean, residual, attention = inputs()
    forward_source_training_hidden(
        model, clean, attention, residual=residual,
        mode=SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21,
    )
    expected = {
        "attention_mask", "position_ids", "past_key_value", "output_attentions",
        "use_cache", "cache_position", "position_embeddings",
    }
    assert set(model.model.layers[21].last_kwargs) == expected
    assert model.model.layers[21].last_kwargs["use_cache"] is False
    assert model.model.layers[21].last_kwargs["output_attentions"] is False


def test_both_modes_preserve_residual_gradient():
    for mode in SourceTrainingContinuationMode:
        model = ToyLanguageModel()
        clean, residual, attention = inputs()
        residual.requires_grad_(True)
        output = forward_source_training_hidden(model, clean, attention, residual=residual, mode=mode)
        output.square().sum().backward()
        assert residual.grad is not None and torch.isfinite(residual.grad).all()


def test_default_config_preserves_archived_corrected_semantics():
    assert LiveEditMedicalConfig().source_training_continuation_mode is (
        SourceTrainingContinuationMode.CORRECTED_SEMANTICS_CONTINUE_LAYER22
    )


def test_training_mode_does_not_enter_inference_hook_api():
    signature = inspect.signature(Layer21ResidualHook)
    assert "source_training_continuation_mode" not in signature.parameters
    assert "mode" not in signature.parameters
