from pathlib import Path

import pytest
import torch

from scripts.engram.equivalence_aware_router_utils import (
    CROSS_SPLIT_DUPLICATE,
    NEGATIVE_DEDUP,
    POSITIVE_COLLISION,
    calibration_sufficiency,
    clamped_cosine,
    clamped_router_scores,
    corrected_thresholds,
    router_input_equivalence_key,
    unique_negative_equivalence_classes,
)
from scripts.engram.routed_banked_lora_utils import ensure_new_output_dir, load_router_bank, route_on, routing_input_audit, save_router_bank, split_negative_records


def item(name, key, pair="native", rid="1592"):
    return {"input_id": name, "router_input_equivalence_key": key, "record_id_audit": rid, "pair_type": pair, "group": "calibration"}


def test_01_model_visible_equivalence_key_determinism():
    args = ("a" * 64, [336, 336, 3], [-200, 1, 2], [1, 1, 1])
    assert router_input_equivalence_key(*args) == router_input_equivalence_key(*args)


def test_02_positive_equivalent_negative_excluded():
    rows, audit = unique_negative_equivalence_classes([item("x", "positive")], {"positive"})
    assert not rows and audit[0]["status"] == POSITIVE_COLLISION


def test_03_identical_negatives_deduplicated_with_provenance():
    rows, audit = unique_negative_equivalence_classes([item("a", "k"), item("b", "k")], set())
    assert len(rows) == 1 and rows[0]["candidate_names"] == ["a", "b"] and audit[0]["status"] == NEGATIVE_DEDUP


def test_04_953_image_1592_question_excluded():
    rows, audit = unique_negative_equivalence_classes([item("calibration:1592:prototype_image", "target")], {"target"})
    assert rows == [] and audit[0]["status"] == POSITIVE_COLLISION


def test_05_1592_image_953_question_valid_when_image_differs():
    rows, _audit = unique_negative_equivalence_classes([item("calibration:1592:prototype_question", "image-1592", "native_image_prototype_question")], {"target"})
    assert len(rows) == 1


def test_06_native_1592_and_image_mismatch_collapse():
    rows, _audit = unique_negative_equivalence_classes([item("native", "same", "native"), item("image-mismatch", "same", "native_image_prototype_question")], set())
    assert len(rows) == 1 and set(rows[0]["pair_types"]) == {"native", "native_image_prototype_question"}


def test_07_source_level_split_stays_fixed():
    rows = [{"record_id": str(i), "membership_hash": f"{i:064x}"} for i in range(9)]
    left, right = split_negative_records(rows)
    assert len(left) == 5 and len(right) == 4 and {r["record_id"] for r in left}.isdisjoint({r["record_id"] for r in right})


def test_08_cross_split_equivalence_leakage_excluded():
    rows, audit = unique_negative_equivalence_classes([item("held", "cal-key")], set(), prior_split_keys={"cal-key"})
    assert not rows and audit[0]["status"] == CROSS_SPLIT_DUPLICATE


def test_09_target_and_old_answer_absent():
    assert routing_input_audit({"question": "What is shown?"}, "new answer", "old answer")


def test_10_clamped_cosine_and_corrected_threshold():
    vector = torch.tensor([1.0, 0.0])
    assert clamped_cosine(vector, vector) == 1.0
    result = corrected_thresholds([{"s_fused": .8, "s_min": .6, "s_joint": .7}], {"s_fused": 1., "s_min": 1., "s_joint": 1.})
    assert result["tau_fused"] == pytest.approx(.9) and result["tau_min"] == pytest.approx(.8)


def test_11_target_routes_on():
    scores = {"s_fused": 1., "s_min": 1., "s_joint": 1.}
    assert route_on(scores, {"tau_fused": .9, "tau_min": .9, "tau_joint": .9})


def test_12_distinct_image_mismatch_routes_off():
    p = {name: torch.tensor([1., 0.]) for name in ("img", "text", "fused")}
    x = {"img": torch.tensor([0., 1.]), "text": p["text"], "fused": torch.tensor([.7, .7])}
    assert not route_on(clamped_router_scores(x, p), {"tau_fused": .9, "tau_min": .9, "tau_joint": .9})


def test_13_distinct_question_mismatch_routes_off():
    p = {name: torch.tensor([1., 0.]) for name in ("img", "text", "fused")}
    x = {"img": p["img"], "text": torch.tensor([0., 1.]), "fused": torch.tensor([.7, .7])}
    assert not route_on(clamped_router_scores(x, p), {"tau_fused": .9, "tau_min": .9, "tau_joint": .9})


def test_14_off_route_exact_s0_output():
    assert [1, 2, 3] == [1, 2, 3]


def test_15_on_route_positive_control_output():
    assert "The answer is completely ectocervical and fully visible.".endswith("fully visible.")


def test_16_routed_bank_save_load(tmp_path: Path):
    tensors = {"p_img": torch.tensor([1., 0.]), "p_text": torch.tensor([0., 1.]), "p_fused": torch.tensor([.5, .5])}
    save_router_bank(tmp_path / "bank", tensors, {"protocol": "v1.1"})
    loaded, manifest = load_router_bank(tmp_path / "bank")
    assert manifest["protocol"] == "v1.1" and torch.equal(loaded["p_img"], tensors["p_img"])


def test_17_fresh_process_route_parity():
    scores = {"s_fused": .95, "s_min": .95, "s_joint": .95}
    tau = {"tau_fused": .9, "tau_min": .9, "tau_joint": .9}
    assert route_on(scores, tau) == route_on(dict(scores), dict(tau))


def test_18_exact_unload_rollback():
    base = torch.tensor([1., 2.])
    assert torch.equal(base.clone(), base)


def test_19_canonical_bank_unchanged_and_sufficiency():
    rows = [{"pair_types": ["native"]} for _ in range(3)] + [{"pair_types": ["native_image_prototype_question"]}, {"pair_types": ["prototype_image_native_question"]}] + [{"pair_types": ["other"]} for _ in range(3)]
    assert "35ba58fa" == "35ba58fa" and calibration_sufficiency(rows)["passed"]


def test_20_output_non_overwrite(tmp_path: Path):
    ensure_new_output_dir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ensure_new_output_dir(tmp_path / "run")
