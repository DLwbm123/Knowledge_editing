import pytest
import torch

from scripts.engram.natural_generation_recovery_utils import (
    assert_no_target_leakage,
    canonical_natural_response,
    exact_tensor_collection_equal,
    expanded_predictor_positions,
    load_v3_bank,
    materialize_fp32_shadow,
    orthonormal_locality_basis,
    project_delta_to_norm,
    project_deltas_to_relative_budget,
    project_effect_gradient,
    rank4_svd_initialization,
    relative_displacement,
    save_v3_bank,
    select_candidate_modules,
)
from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir


def test_fixed_canonical_natural_response_construction():
    assert canonical_natural_response(" completely ectocervical and fully visible. ") == "The answer is completely ectocervical and fully visible."


def test_no_target_leakage():
    assert_no_target_leakage(["Question: what is shown?", "Give only the answer"], "edited target")
    with pytest.raises(AssertionError):
        assert_no_target_leakage(["Question: edited target"], "edited target")


def test_shifted_labels_after_multimodal_expansion():
    assert expanded_predictor_positions(10, 3, 575) == [584, 585, 586]


def test_fp32_accumulates_across_low_precision_plateau():
    base = torch.tensor([1.0], dtype=torch.float32)
    delta = torch.zeros(1, dtype=torch.float32)
    observed = []
    for _ in range(8):
        delta += 1e-4
        observed.append(float(materialize_fp32_shadow(base, delta, torch.float16)[0]))
    assert delta.item() == pytest.approx(8e-4)
    assert len(set(observed[:4])) < 4 and observed[-1] > observed[0]


def test_zero_delta_s0_parity():
    base = torch.tensor([1.0, -2.0], dtype=torch.float16)
    effective = materialize_fp32_shadow(base, torch.zeros(2, dtype=torch.float32), torch.float16)
    assert torch.equal(base, effective)


def test_exact_b1_and_relative_displacement_projection():
    delta = project_delta_to_norm(torch.ones(100), 0.007530835302)
    assert delta.double().norm().item() == pytest.approx(0.007530835302, abs=5e-10)
    projected = project_deltas_to_relative_budget([torch.ones(10)], [torch.ones(10)], 0.003)
    assert relative_displacement(projected, [torch.ones(10)]) == pytest.approx(0.003, abs=1e-8)


def test_deterministic_fixed_candidate_module_selection():
    rows = [
        {"layer": 19, "module_name": "b", "target_size_normalized_norm": 2.0, "locality_size_normalized_norm": 1.0},
        {"layer": 18, "module_name": "z", "target_size_normalized_norm": 2.0, "locality_size_normalized_norm": 1.0},
        {"layer": 18, "module_name": "a", "target_size_normalized_norm": 2.0, "locality_size_normalized_norm": 1.0},
        {"layer": 20, "module_name": "a", "target_size_normalized_norm": 1.0, "locality_size_normalized_norm": 1.0},
    ]
    assert [row["module_name"] for row in select_candidate_modules(rows)] == ["a", "z", "b"]


def test_rank4_svd_initialization():
    gradient = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5]))
    factors = rank4_svd_initialization(gradient)
    reconstruction = factors["B"] @ factors["A"]
    assert reconstruction.shape == gradient.shape
    assert torch.linalg.matrix_rank(reconstruction).item() == 4
    assert torch.sum(reconstruction * gradient).item() < 0


def test_locality_gradient_projection_in_factor_space():
    probes = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]
    basis = orthonormal_locality_basis(probes)
    projected = project_effect_gradient(torch.tensor([2.0, 3.0, 4.0]), basis)
    assert projected.tolist() == pytest.approx([0.0, 0.0, 4.0], abs=1e-6)


def test_isolated_v3_bank_reload(tmp_path):
    root = tmp_path / "v3_bank"
    factors = {"layer": {"A": torch.ones(2, 3), "B": torch.ones(4, 2), "scale": torch.tensor(0.5)}}
    save_v3_bank(root, factors, {"rank": 2})
    loaded = load_v3_bank(root)
    assert torch.equal(loaded["factors"]["layer"]["A"], factors["layer"]["A"])


def test_exact_rollback_and_replay():
    base = [torch.tensor([1.0, 2.0])]
    candidate = [torch.tensor([1.1, 2.2])]
    assert exact_tensor_collection_equal(base, [value.clone() for value in base])
    assert exact_tensor_collection_equal(candidate, [value.clone() for value in candidate])
    assert not exact_tensor_collection_equal(base, candidate)


def test_output_directory_nonoverwrite(tmp_path):
    path = tmp_path / "recovery"
    create_new_output_dir(path)
    with pytest.raises(FileExistsError):
        create_new_output_dir(path)
