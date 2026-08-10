import pytest
import torch

from scripts.engram.natural_generation_rescue_utils import (
    ScaffoldAlignment,
    align_model_short_to_unrestricted,
    assert_shadow_only,
    assert_target_free_generation_prompts,
    choose_backtracking_proposal,
    construct_target_path,
    deterministic_best_prefix,
    exact_rollback,
    load_candidate_state,
    project_shadow,
    relative_cap_reached,
    save_candidate_state,
    snapshot_identity,
)
from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir, temporary_parameter_delta
from scripts.engram.stage1_behavioral_margin_utils import relative_parameter_displacement


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        vocab = {1: "The", 2: " answer", 3: " is", 4: " target", 5: ".", 6: " extra", 7: "THE"}
        return "".join(vocab.get(int(item), "") for item in ids)


def test_model_short_answer_to_unrestricted_alignment():
    result = align_model_short_to_unrestricted([6, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5], tokenizer=ToyTokenizer())
    assert result == ScaffoldAlignment(1, 6, "exact_token", 1.0)
    normalized = align_model_short_to_unrestricted([6, 1, 2, 3, 4, 5], [7, 2, 3, 4], tokenizer=ToyTokenizer())
    assert normalized == ScaffoldAlignment(2, 5, "normalized_token", 1.0)


def test_deterministic_best_prefix_fallback_uses_earliest_tie():
    rows = [
        {"prefix_length": 2, "m_first": 1.0, "m_4": 0.0, "nll_target": 0.0},
        {"prefix_length": 1, "m_first": 1.01, "m_4": 0.0, "nll_target": 0.0},
    ]
    assert deterministic_best_prefix(rows)["prefix_length"] == 1


def test_full_sequence_tokenization_and_shifted_span_labels():
    result = construct_target_path(torch.tensor([[9, 8]]), [7, 6], torch.tensor([4, 3]), [5])
    assert result["target_span"] == [4, 6]
    assert result["shifted_labels"][0, result["target_predictor_positions"]].tolist() == [4, 3]
    assert result["complete_ids"].tolist() == [[9, 8, 7, 6, 4, 3, 5]]


def test_target_absent_from_generation_inputs():
    assert_target_free_generation_prompts("Question: image?", "Question: image? short answer", "edited target")
    with pytest.raises(AssertionError):
        assert_target_free_generation_prompts("Question: edited target", "safe", "edited target")


def test_shadow_only_and_canonical_bank_immutability():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = snapshot_identity(parameter, "bank")
    shadow = torch.tensor([0.1, -0.1])
    assert torch.equal(parameter.detach(), torch.tensor([1.0, 2.0])) and shadow.norm() > 0
    assert_shadow_only(before, snapshot_identity(parameter, "bank"))


@pytest.mark.parametrize("cap", [0.003, 0.010])
def test_global_norm_displacement_projection(cap):
    anchor = torch.ones(100)
    projected = project_shadow(torch.ones(100), anchor, cap)
    assert relative_parameter_displacement(projected, anchor) == pytest.approx(cap, abs=1e-7)
    assert relative_cap_reached(projected, anchor, cap)


def test_deterministic_backtracking_acceptance():
    baseline = {"loss": 3.0, "natural_first_margin": -2.0, "sequence_contrast": 0.0}
    proposals = [
        {"factor": 1.0, "loss": 2.0, "natural_first_margin": -1.0, "prefix_preserved": False, "locality_top1_preserved": True, "locality_nll_drift": 0.0},
        {"factor": 0.5, "loss": 2.5, "natural_first_margin": -1.5, "prefix_preserved": True, "locality_top1_preserved": True, "locality_nll_drift": 0.001},
    ]
    assert choose_backtracking_proposal(baseline, proposals)["factor"] == 0.5


def test_candidate_reload_and_exact_rollback(tmp_path):
    path = tmp_path / "candidate" / "state.pt"
    save_candidate_state(path, torch.tensor([0.1, 0.2]), {"name": "rescue"})
    assert torch.equal(load_candidate_state(path)["delta"], torch.tensor([0.1, 0.2]))
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    snapshot = parameter.detach().clone()
    with temporary_parameter_delta(parameter, torch.tensor([0.1, 0.2])):
        pass
    assert exact_rollback(parameter, snapshot)


def test_output_directory_nonoverwrite(tmp_path):
    path = tmp_path / "rescue"
    create_new_output_dir(path)
    with pytest.raises(FileExistsError):
        create_new_output_dir(path)
