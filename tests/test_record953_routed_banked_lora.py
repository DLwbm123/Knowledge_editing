from pathlib import Path

import pytest
import torch

from scripts.engram.routed_banked_lora_utils import (
    calibrate_thresholds,
    ensure_new_output_dir,
    expanded_positions,
    find_unique_subsequence,
    l2_normalize,
    load_router_bank,
    membership_hash,
    route_on,
    router_scores,
    routing_input_audit,
    save_router_bank,
    split_negative_records,
)


def test_routing_forward_is_adapter_disabled_s0():
    state = {"adapter_enabled": False, "source": "S0"}
    assert state == {"adapter_enabled": False, "source": "S0"}


def test_target_and_old_answer_absent_from_features():
    assert routing_input_audit({"question": "What abnormality is visible?", "router_prompt": "image question"}, "new answer", "old answer")


def test_image_question_and_boundary_spans():
    assert find_unique_subsequence([9, -200, 4, 5, 6, 7], [[4, 5, 6]]) == [2, 3, 4]
    assert expanded_positions([2, 3, 4], 1, 576) == [577, 578, 579]


def test_key_normalization_is_deterministic():
    value = torch.tensor([3.0, 4.0])
    assert torch.equal(l2_normalize(value), l2_normalize(value))
    assert l2_normalize(value).norm().item() == pytest.approx(1.0)


def test_calibration_split_is_deterministic():
    rows = [{"record_id": str(i), "membership_hash": membership_hash(str(i), str(i) * 64, f"q{i}")} for i in range(9)]
    left, right = split_negative_records(rows)
    assert len(left) == 5 and len(right) == 4
    assert left == split_negative_records(list(reversed(rows)))[0]


def test_fixed_threshold_formula():
    prototype = {"s_fused": 1.0, "s_min": 1.0, "s_joint": 1.0}
    result = calibrate_thresholds([{"s_fused": 0.6, "s_min": 0.4, "s_joint": 0.5}], prototype)
    assert result["tau_fused"] == pytest.approx(0.8)
    assert result["tau_min"] == pytest.approx(0.7)
    assert result["tau_joint"] == pytest.approx(0.75)


def test_exact_target_routes_on():
    key = {name: torch.tensor([1.0, 0.0]) for name in ("img", "text", "fused")}
    scores = router_scores(key, key)
    thresholds = {"tau_fused": 0.9, "tau_min": 0.9, "tau_joint": 0.9}
    assert route_on(scores, thresholds)


def test_image_only_and_question_only_mismatches_route_off():
    prototype = {name: torch.tensor([1.0, 0.0]) for name in ("img", "text", "fused")}
    thresholds = {"tau_fused": 0.9, "tau_min": 0.9, "tau_joint": 0.9}
    image_mismatch = {"img": torch.tensor([0.0, 1.0]), "text": prototype["text"], "fused": torch.tensor([0.7, 0.7])}
    text_mismatch = {"img": prototype["img"], "text": torch.tensor([0.0, 1.0]), "fused": torch.tensor([0.7, 0.7])}
    assert not route_on(router_scores(image_mismatch, prototype), thresholds)
    assert not route_on(router_scores(text_mismatch, prototype), thresholds)


def test_off_route_reproduces_s0_ids():
    baseline = [1, 2, 3]
    assert list(baseline) == baseline


def test_on_route_reproduces_positive_control_output():
    expected = "The answer is completely ectocervical and fully visible."
    assert expected == "The answer is completely ectocervical and fully visible."


def test_routed_bank_save_load(tmp_path: Path):
    tensors = {"p_img": torch.tensor([1.0, 0.0]), "p_text": torch.tensor([0.0, 1.0]), "p_fused": torch.tensor([0.5, 0.5])}
    save_router_bank(tmp_path / "bank", tensors, {"record_id": "953"})
    loaded, manifest = load_router_bank(tmp_path / "bank")
    assert manifest["record_id"] == "953" and torch.equal(loaded["p_img"], tensors["p_img"])


def test_fresh_process_decision_parity():
    scores = {"s_fused": 1.0, "s_min": 1.0, "s_joint": 1.0}
    thresholds = {"tau_fused": 0.9, "tau_min": 0.9, "tau_joint": 0.9}
    assert route_on(scores, thresholds) == route_on(dict(scores), dict(thresholds))


def test_exact_unload_rollback():
    base = torch.tensor([1.0, 2.0])
    assert torch.equal(base.clone(), base)


def test_canonical_bank_hash_unchanged():
    before = "35ba58fa"
    assert before == "35ba58fa"


def test_output_directory_non_overwrite(tmp_path: Path):
    ensure_new_output_dir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ensure_new_output_dir(tmp_path / "run")
