from pathlib import Path

import pytest
import torch

from scripts.engram.stage0abc_diagnostic_utils import (
    create_new_output_dir,
    first_prefix_parity,
    projected_adjoint_via_autograd,
    temporary_parameter_delta,
)
from scripts.engram.stage0_generation_audit_utils import CanonicalInputs, assert_no_gold_leakage


def test_first_64_token_cap_extension_parity():
    reference = list(range(64))
    assert first_prefix_parity(reference, reference + [64, 65])
    changed = reference.copy()
    changed[63] = -1
    assert not first_prefix_parity(reference, changed + [64])


def test_no_target_leakage_in_extended_generation():
    prompt = torch.tensor([[1, 2]])
    item = CanonicalInputs("p", "pa", prompt, torch.tensor([[1, 2, 3]]), torch.zeros(1), 2, torch.tensor([3]), "p", "f", "i")
    assert_no_gold_leakage(prompt, item)
    with pytest.raises(AssertionError):
        assert_no_gold_leakage(torch.tensor([[1, 2, 3]]), item)


def test_projector_adjoint_uses_transpose_for_nonsymmetric_projector():
    projector = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    output_gradient = torch.tensor([[3.0, 5.0]])
    effective = projected_adjoint_via_autograd(output_gradient, (1, 2), lambda value: value @ projector)
    assert torch.equal(effective, output_gradient @ projector.T)
    assert not torch.equal(effective, output_gradient @ projector)


def test_zero_step_finite_difference_is_identity():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = parameter.detach().clone()
    with temporary_parameter_delta(parameter, torch.zeros_like(parameter)) as ledger:
        assert torch.equal(parameter, before)
    assert ledger["rollback_exact"]
    assert ledger["before_hash"] == ledger["temporary_hash"] == ledger["after_hash"]


def test_temporary_delta_rolls_back_exactly():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = parameter.detach().clone()
    with temporary_parameter_delta(parameter, torch.tensor([0.25, -0.5])) as ledger:
        assert not torch.equal(parameter, before)
    assert torch.equal(parameter, before)
    assert ledger["rollback_exact"]


def test_cached_no_cache_hf_parity_comparator():
    tokens = list(range(80))
    assert first_prefix_parity(tokens, tokens)
    assert first_prefix_parity(tokens, list(tokens))


def test_existing_output_directory_is_never_reused(tmp_path: Path):
    output = tmp_path / "audit"
    create_new_output_dir(output)
    with pytest.raises(FileExistsError):
        create_new_output_dir(output)
