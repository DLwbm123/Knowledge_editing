import torch

from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram.engram_main import apply_engram_to_linear
from easyeditor.models.engram.solver import apply_update_to_module


def _toy_activations():
    target = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.9, -0.1]])
    reference = torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 0.9]])
    return target, reference


def test_engram_linear_erasure_reduces_target_more_than_reference():
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.data[:] = torch.tensor([[1.0, 1.0]])
    target, reference = _toy_activations()
    before_target = layer(target).norm()
    before_reference = layer(reference).norm()
    hparams = EngramMultimodalHparams(alpha=0.8, absorb_bias=False, token_scope="all", module_patterns=[r".*"])
    update = apply_engram_to_linear(layer, target, reference, hparams=hparams)
    after_target = layer(target).norm()
    after_reference = layer(reference).norm()
    assert after_target < before_target
    assert abs(after_reference - before_reference) < abs(after_target - before_target)
    apply_update_to_module(layer, update, direction=1)
    assert torch.allclose(layer.weight, torch.tensor([[1.0, 1.0]]), atol=1.0e-6)


def test_engram_bias_absorption_updates_and_rolls_back_bias():
    layer = torch.nn.Linear(2, 1, bias=True)
    layer.weight.data[:] = torch.tensor([[1.0, 1.0]])
    layer.bias.data[:] = torch.tensor([0.25])
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()
    target, reference = _toy_activations()
    hparams = EngramMultimodalHparams(alpha=0.5, absorb_bias=True, token_scope="all", module_patterns=[r".*"])
    update = apply_engram_to_linear(layer, target, reference, hparams=hparams)
    assert update.bias is not None
    assert not torch.allclose(layer.bias, original_bias)
    apply_update_to_module(layer, update, direction=1)
    assert torch.allclose(layer.weight, original_weight, atol=1.0e-6)
    assert torch.allclose(layer.bias, original_bias, atol=1.0e-6)


def test_engram_replacement_mode_uses_candidate_delta():
    base = torch.nn.Linear(2, 1, bias=False)
    candidate = torch.nn.Linear(2, 1, bias=False)
    base.weight.data[:] = torch.tensor([[1.0, 0.0]])
    candidate.weight.data[:] = torch.tensor([[0.0, 1.0]])
    target, reference = _toy_activations()
    before_candidate_distance = (base(target) - candidate(target)).norm()
    hparams = EngramMultimodalHparams(
        edit_mode="replacement",
        candidate_delta_source="state_dict_pair",
        alpha=0.2,
        beta=0.8,
        absorb_bias=False,
        token_scope="all",
        module_patterns=[r".*"],
    )
    apply_engram_to_linear(base, target, reference, hparams=hparams, candidate_module=candidate)
    after_candidate_distance = (base(target) - candidate(target)).norm()
    assert after_candidate_distance < before_candidate_distance

