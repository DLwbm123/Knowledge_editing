from pathlib import Path

import pytest
import torch

from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir, temporary_parameter_delta
from scripts.engram.stage1_behavioral_margin_utils import (
    HARD_STOP_SPAN_AMBIGUOUS,
    HARD_STOP_SPAN_NOT_FOUND,
    AnswerSpan,
    NaturalAnswerSpanError,
    align_unique_answer_span,
    assert_bank_immutable,
    assert_candidate_namespace,
    assert_generation_inputs_target_free,
    assert_locality_preserved,
    assert_three_path_parity,
    canonical_optimizer_hash,
    clip_global_relative_displacement,
    exact_parameter_rollback,
    fresh_reproduction_equal,
    load_candidate_payload,
    match_exact_budget,
    normalized_shadow_step,
    prompt_plus_natural_prefix,
    relative_parameter_displacement,
    run_deterministic_twice,
    save_candidate_payload_exclusive,
    shifted_boundary_training_tensors,
    tensor_l2,
)


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        vocab = {1: "old", 2: " answer", 3: " x", 4: "!", 5: "new"}
        return "".join(vocab[int(item)] for item in ids)


def test_unique_natural_answer_span_alignment():
    span = align_unique_answer_span([3, 1, 2, 4], [1, 2], tokenizer=ToyTokenizer(), answer_text="old answer")
    assert span == AnswerSpan(1, 3, "exact_token", "old answer")


@pytest.mark.parametrize(
    ("response", "label"),
    [([3, 4], HARD_STOP_SPAN_NOT_FOUND), ([1, 2, 3, 1, 2], HARD_STOP_SPAN_AMBIGUOUS)],
)
def test_absent_or_ambiguous_answer_span_fails(response, label):
    with pytest.raises(NaturalAnswerSpanError) as exc:
        align_unique_answer_span(response, [1, 2], tokenizer=ToyTokenizer(), answer_text="old answer")
    assert exc.value.label == label


def test_prompt_plus_natural_prefix_positioning():
    prompt = torch.tensor([[9, 8]])
    span = AnswerSpan(2, 4, "exact_token", "old answer")
    assert prompt_plus_natural_prefix(prompt, [3, 4, 1, 2], span).tolist() == [[9, 8, 3, 4]]


def test_shifted_label_is_correct_at_boundary():
    inputs, labels, positions = shifted_boundary_training_tensors(torch.tensor([[9, 8, 7]]), torch.tensor([5, 6]))
    assert inputs.tolist() == [[9, 8, 7, 5]]
    assert labels[0, positions].tolist() == [5, 6]


def test_no_target_leakage_in_generation_inputs():
    unrestricted = torch.tensor([[1, 2]])
    short = torch.tensor([[1, 2, 3]])
    assert_generation_inputs_target_free(unrestricted, unrestricted.clone(), short, short.clone(), [5, 6])
    with pytest.raises(AssertionError):
        assert_generation_inputs_target_free(torch.tensor([[1, 2, 5, 6]]), unrestricted, short, short, [5, 6])


def test_b1_norm_exactly_matches_existing_delta():
    candidate = match_exact_budget(torch.tensor([3.0, 4.0]), 0.125)
    assert tensor_l2(candidate) == pytest.approx(0.125, abs=1e-8)


def test_bmax_relative_displacement_clipping():
    anchor = torch.ones(100)
    clipped = clip_global_relative_displacement(torch.ones(100), anchor, 0.003)
    assert relative_parameter_displacement(clipped, anchor) == pytest.approx(0.003, abs=1e-8)


def test_shadow_delta_only_optimization():
    weight = torch.nn.Parameter(torch.tensor([7.0, 8.0]))
    shadow = torch.zeros(2)
    before = weight.detach().clone()
    updated = normalized_shadow_step(shadow, torch.tensor([3.0, 4.0]), 0.2)
    assert torch.equal(weight.detach(), before)
    assert torch.equal(shadow, torch.zeros(2))
    assert tensor_l2(updated) == pytest.approx(0.2, abs=1e-8)


def test_original_bank_immutability():
    before = {"sha256": "abc", "files": [1]}
    assert_bank_immutable(before, {"sha256": "abc", "files": [1]})
    with pytest.raises(RuntimeError):
        assert_bank_immutable(before, {"sha256": "def", "files": [1]})


def test_candidate_namespace_isolated(tmp_path):
    original = tmp_path / "original"
    candidate = tmp_path / "experiment" / "candidate"
    assert_candidate_namespace(original, candidate)
    with pytest.raises(ValueError):
        assert_candidate_namespace(original, original / "candidate")


def test_deterministic_optimizer_replay():
    operation = lambda: normalized_shadow_step(torch.zeros(2), torch.tensor([3.0, 4.0]), 0.2)
    first, second = run_deterministic_twice(operation)
    assert torch.equal(first, second)
    assert canonical_optimizer_hash(first, 1, 0.2) == canonical_optimizer_hash(second, 1, 0.2)


def test_cached_no_cache_hf_parity():
    assert_three_path_parity([1, 2, 3], [1, 2, 3], [1, 2, 3])
    with pytest.raises(RuntimeError):
        assert_three_path_parity([1], [2], [1])


def test_exact_locality_token_sequence_preservation():
    row = {"token_ids": [1, 2], "normalized_output": "same", "first_top1_id": 1, "stop_reason": "eos", "cap_hit": False, "nll": 1.0}
    assert_locality_preserved(row, dict(row, nll=1.009))
    with pytest.raises(RuntimeError):
        assert_locality_preserved(row, dict(row, token_ids=[1, 3]))


def test_candidate_bank_reload(tmp_path):
    path = tmp_path / "candidate" / "payload.pt"
    payload = {"delta": torch.tensor([1.0, 2.0]), "namespace": "stage1"}
    save_candidate_payload_exclusive(path, payload)
    loaded = load_candidate_payload(path)
    assert torch.equal(loaded["delta"], payload["delta"])
    assert loaded["namespace"] == "stage1"


def test_fresh_process_reproduction():
    reference = {"token_ids": [1, 2], "nll": 1.0}
    assert fresh_reproduction_equal(reference, {"token_ids": [1, 2], "nll": 1.00001}, 1e-4)
    assert not fresh_reproduction_equal(reference, {"token_ids": [1, 3], "nll": 1.0}, 1e-4)


def test_exact_temporary_delta_rollback():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    snapshot = parameter.detach().clone()
    with temporary_parameter_delta(parameter, torch.tensor([0.25, -0.25])) as ledger:
        assert not torch.equal(parameter.detach(), snapshot)
    assert ledger["rollback_exact"]
    assert exact_parameter_rollback(parameter, snapshot)


def test_existing_output_directory_is_never_overwritten(tmp_path):
    path = tmp_path / "run"
    assert create_new_output_dir(path) == path.resolve()
    with pytest.raises(FileExistsError):
        create_new_output_dir(path)
