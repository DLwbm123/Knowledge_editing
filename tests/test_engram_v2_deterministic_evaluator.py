import math

import torch
import torch.nn.functional as F

from scripts.engram.engram_eval_utils import (
    compare_tensors,
    legacy_tail_metrics,
    nested_input_hash,
    shifted_teacher_forced_metrics,
    tensor_sha256,
)


def test_shifted_teacher_forced_metrics_matches_manual_cross_entropy():
    logits = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 3.0, -1.0], [0.0, -1.0, 2.0], [1.0, 1.0, 1.0]]])
    labels = torch.tensor([[-100, 1, 2, -100]])
    result = shifted_teacher_forced_metrics(logits, labels)
    expected = F.cross_entropy(torch.stack([logits[0, 0], logits[0, 1]]), torch.tensor([1, 2]))
    assert result["target_token_count"] == 2
    assert math.isclose(result["target_nll"], float(expected), rel_tol=0.0, abs_tol=1.0e-7)
    assert result["target_positions_after_shift"] == [[0, 0], [0, 1]]


def test_shifted_metric_rejects_unaligned_shapes_and_empty_mask():
    logits = torch.zeros(1, 3, 5)
    try:
        shifted_teacher_forced_metrics(logits, torch.zeros(1, 2, dtype=torch.long))
        raise AssertionError("shape mismatch must fail")
    except ValueError:
        pass
    try:
        shifted_teacher_forced_metrics(logits, torch.full((1, 3), -100, dtype=torch.long))
        raise AssertionError("empty target mask must fail")
    except RuntimeError:
        pass


def test_legacy_tail_alignment_is_not_equivalent_to_expanded_causal_labels():
    logits = torch.randn(1, 7, 11, generator=torch.Generator().manual_seed(3))
    expanded = torch.tensor([[-100, -100, -100, 4, 5, 6, -100]])
    raw = torch.tensor([[4, 5, 6]])
    correct = shifted_teacher_forced_metrics(logits, expanded)
    legacy = legacy_tail_metrics(logits, raw)
    assert correct["target_token_count"] == legacy["target_token_count"] == 3
    assert not math.isclose(correct["target_nll"], legacy["target_nll"], rel_tol=0.0, abs_tol=1.0e-8)


def test_tensor_and_nested_hashes_detect_mutation():
    tensor = torch.arange(6).reshape(2, 3)
    clone = tensor.clone()
    assert tensor_sha256(tensor) == tensor_sha256(clone)
    before = nested_input_hash({"x": tensor, "label": ["a"]})
    clone[0, 0] = 99
    after = nested_input_hash({"x": clone, "label": ["a"]})
    assert before != after
    comparison = compare_tensors(tensor, clone)
    assert not comparison["exact_equal"]
    assert comparison["max_abs_diff"] == 99.0
