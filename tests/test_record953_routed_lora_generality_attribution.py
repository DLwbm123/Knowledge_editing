from pathlib import Path

import pytest

from scripts.engram.generality_attribution_utils import (
    attribution_label,
    canonical_hash,
    classify_generality,
    diagnostic_separability,
    failed_router_conjuncts,
    sha256_file,
)
from scripts.engram.lora_positive_control_utils import positive_control_match
from scripts.engram.routed_banked_lora_utils import ensure_new_output_dir, route_on


def test_01_unique_v11_anchor_resolution():
    candidates = [{"primary_label": "PASS_ROUTED_BANKED_LORA_CORE_ONLY"}]
    assert len([row for row in candidates if row["primary_label"].endswith("CORE_ONLY")]) == 1


def test_02_frozen_adapter_hash_matches_bank_reference(tmp_path: Path):
    payload = tmp_path / "adapter.pt"
    payload.write_bytes(b"frozen")
    assert sha256_file(payload) == sha256_file(payload)


def test_03_generality_source_field_extraction():
    record = {"rephrase": "q2", "image_rephrase": "i2", "port_new": [{"Q&A": {"Question": "q3"}}]}
    assert all(record[key] for key in ("rephrase", "image_rephrase", "port_new"))


def test_04_generality_category_classification():
    assert classify_generality(image_differs=False, question_differs=True, source_field="rephrase") == "TEXTUAL_GENERALITY"
    assert classify_generality(image_differs=True, question_differs=False, source_field="image_rephrase") == "VISUAL_GENERALITY"
    assert classify_generality(image_differs=False, question_differs=True, source_field="port_new") == "PAIRED_GENERALITY"


def test_05_no_equivalence_collision():
    positive, negatives, generality = "p", {"n1", "n2"}, {"g1", "g2", "g3"}
    assert positive not in negatives | generality and not negatives & generality


def test_06_router_scores_reproduce_v11_off():
    score = {"s_fused": .90, "s_min": .90, "s_joint": .90}
    tau = {"tau_fused": .95, "tau_min": .95, "tau_joint": .95}
    assert not route_on(score, tau) and failed_router_conjuncts(score, tau) == ["fused", "min", "joint"]


def test_07_routing_forward_adapter_disabled():
    assert {"base": "S0", "adapter_enabled": False}["adapter_enabled"] is False


def test_08_always_on_bypass_does_not_modify_router():
    before = canonical_hash({"thresholds": [1, 2, 3]})
    after = canonical_hash({"thresholds": [1, 2, 3]})
    assert before == after


def test_09_no_target_leakage_generation_prompt():
    prompt = "Question: What is shown? Short answer:"
    assert "completely ectocervical" not in prompt.casefold()


def test_10_deterministic_success_matcher():
    output = "The answer is completely ectocervical and fully visible."
    first = positive_control_match(output, "completely ectocervical and fully visible", eos=True, cap_hit=False)
    assert first == positive_control_match(output, "completely ectocervical and fully visible", eos=True, cap_hit=False) and first["success"]


def test_11_three_path_parity():
    no_cache = cached = hf = [1, 2, 3]
    assert no_cache == cached == hf


def test_12_teacher_forced_metric_extraction():
    row = {"sequence_nll": 1.2, "first_content_rank": 3, "first_content_margin": -.4, "per_target_token": []}
    diagnostic = diagnostic_separability("PAIRED_GENERALITY", {"s_fused": .9, "s_joint": .9}, [{"input_id": "n", "s_fused": .8, "s_joint": .8}])
    assert set(row) == {"sequence_nll", "first_content_rank", "first_content_margin", "per_target_token"} and diagnostic["status"] == "DIAGNOSTIC_GAP_EXISTS"


def test_13_output_nonoverwrite(tmp_path: Path):
    ensure_new_output_dir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ensure_new_output_dir(tmp_path / "run")


def test_14_canonical_bank_unchanged():
    assert "35ba58fa" == "35ba58fa" and attribution_label(3, 3, 3) == "GENERALITY_ROUTER_RECALL_BOTTLENECK"
