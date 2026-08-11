from pathlib import Path

import pytest
import torch

from scripts.engram.v3_1_locality_corrected_utils import (
    FixedRightWeight,
    choose_directional_sign,
    copy_flat,
    ensure_new_output_dir,
    equality_kl_gradient_is_unusable,
    fixed_right_basis,
    fragile_positive_position,
    induced_effective_norm,
    locality_basis,
    normalize_effective_step,
    preservation_margin_loss,
    preservation_nll,
    project_gradient,
    select_modules,
    unsupported_specificity_terms,
)
from scripts.engram.natural_generation_recovery_utils import assert_no_target_leakage


def test_kl_at_equality_is_detected_as_unusable():
    assert equality_kl_gradient_is_unusable(torch.randn(4, 11))


def test_baseline_token_nll_gradient_is_nonzero():
    logits = torch.randn(5, 13, requires_grad=True)
    loss = preservation_nll(logits, logits.detach().argmax(dim=1))
    assert torch.autograd.grad(loss, logits)[0].norm() > 0


def test_fragile_margin_gradient_is_nonzero_and_position_is_fixed():
    logits = torch.tensor([[4.0, 2.0, 0.0], [0.0, 1.1, 1.0]], requires_grad=True)
    ids = torch.tensor([0, 1])
    position = fragile_positive_position(logits, ids)
    assert position == 1
    assert torch.autograd.grad(preservation_margin_loss(logits, ids, position), logits)[0].norm() > 0


def test_module_selection_is_deterministic_and_excludes_zero_locality():
    rows = [
        {"module_name": "z", "layer": 20, "target_size_normalized_norm": 3.0, "locality_size_normalized_norm": 1.0},
        {"module_name": "a", "layer": 19, "target_size_normalized_norm": 2.0, "locality_size_normalized_norm": 1.0},
        {"module_name": "b", "layer": 18, "target_size_normalized_norm": 4.0, "locality_size_normalized_norm": 2.0},
        {"module_name": "zero", "layer": 18, "target_size_normalized_norm": 100.0, "locality_size_normalized_norm": 0.0},
    ]
    assert [row["module_name"] for row in select_modules(rows)] == ["z", "b", "a"]


def test_exact_zero_initialization_and_s0_parity():
    base = torch.randn(7, 9)
    parameterization = FixedRightWeight(base, torch.eye(4, 9))
    assert torch.equal(parameterization.B, torch.zeros_like(parameterization.B))
    assert torch.equal(parameterization(base), base)


def test_fixed_right_rank4_mapping_and_energy():
    gradient = torch.randn(8, 9)
    result = fixed_right_basis(-gradient)
    a_fixed = result["A_fixed"]
    assert a_fixed.shape == (4, 9)
    assert torch.linalg.matrix_rank(a_fixed) == 4
    assert torch.allclose(a_fixed @ a_fixed.T, torch.eye(4), atol=1e-5)
    assert 0 < result["captured_energy_fraction"] <= 1


def test_effective_weight_step_normalization():
    parameters = [torch.zeros(6, 4), torch.zeros(5, 4)]
    bases = [torch.eye(4, 8), torch.eye(4, 7)]
    direction = torch.randn(sum(item.numel() for item in parameters))
    step = normalize_effective_step(direction, parameters, bases, 0.125)
    assert induced_effective_norm(step, parameters, bases) == pytest.approx(0.125, abs=1e-8)


def test_nonzero_locality_basis_rank():
    result = locality_basis([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 2.0, 0.0])])
    assert result["rank"] == 2
    assert result["nonzero_directions"] == 2


def test_projection_orthogonality_residual():
    basis = locality_basis([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])])["basis"]
    projected, residual = project_gradient(torch.tensor([3.0, 4.0, 5.0]), basis)
    assert residual <= 1e-6
    assert torch.allclose(projected, torch.tensor([0.0, 0.0, 5.0]), atol=1e-6)


def test_plus_minus_directional_gate():
    baseline = {"baseline_effect_loss": 2.0, "baseline_primary_margin": -2.0, "baseline_primary_sequence_score": -3.0}
    safe = {**baseline, "effect_loss": 1.0, "primary_margin": -1.0, "primary_sequence_score": -2.5, "maximum_locality_nll_drift": 0.001, "paired_first_top1_equal": True, "rollback_exact": True}
    unsafe = {**baseline, "effect_loss": 3.0, "primary_margin": -3.0, "primary_sequence_score": -4.0, "maximum_locality_nll_drift": 0.0, "paired_first_top1_equal": True, "rollback_exact": True}
    assert choose_directional_sign(safe, unsafe) == 1


def test_no_target_leakage():
    assert_no_target_leakage(["What abnormality is shown?"], "pneumothorax")
    with pytest.raises(AssertionError):
        assert_no_target_leakage(["Is this pneumothorax?"], "pneumothorax")


def test_exact_rollback_and_replay_in_factor_space():
    parameter = torch.zeros(2, 3)
    candidate = torch.arange(6.0)
    copy_flat([parameter], candidate)
    assert torch.equal(parameter.reshape(-1), candidate)
    copy_flat([parameter], torch.zeros(6))
    assert torch.equal(parameter, torch.zeros_like(parameter))
    copy_flat([parameter], candidate)
    assert torch.equal(parameter.reshape(-1), candidate)


def test_output_directory_non_overwrite(tmp_path: Path):
    path = tmp_path / "run"
    ensure_new_output_dir(path)
    with pytest.raises(FileExistsError):
        ensure_new_output_dir(path)


def test_unsupported_subtype_specificity_is_preregistered_failure():
    terms = unsupported_specificity_terms(
        "The image shows a 12-day-old embryonic quail.",
        "The image shows a 12-day-old embryo of a Japanese quail.",
        "12-day-old Japanese quail embryo",
    )
    assert terms == ["japanese"]
