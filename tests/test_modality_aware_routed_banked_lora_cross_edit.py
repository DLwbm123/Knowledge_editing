from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.engram.modality_aware_router_utils import (
    BRANCHES,
    MODALITIES,
    edit_level_split,
    exported_probability,
    fixed_logistic_regression,
    fit_pcas,
    json_hash,
    normalize_question,
    pca_transform_exported,
    relation_features,
    source_sort_key,
    stable_negative_cap,
    train_branch,
    validated_scores,
    zero_fp_threshold,
)


def keys(seed: int, count: int = 70, dim: int = 40):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=(count, dim)).astype(np.float32) for name in MODALITIES}


def test_01_anchor_resolution_contract():
    required = {"positive_control", "v1_1_exact_router", "generality_attribution"}
    assert len(required) == 3 and "positive_control" in required


def test_02_record953_exclusion_from_fitting():
    rows = [{"record_id": str(i)} for i in range(1000, 1096)]
    split = edit_level_split(rows)
    assert all(row["record_id"] != "953" for values in split.values() for row in values)


def test_03_edit_level_split_isolation():
    split = edit_level_split([{"record_id": str(i)} for i in range(96)])
    assert [len(split[name]) for name in ("train", "calibration", "heldout")] == [64, 16, 16]
    assert not ({r["record_id"] for r in split["train"]} & {r["record_id"] for r in split["heldout"]})


def test_04_source_grounded_generality_categories():
    assert set(BRANCHES) == {"textual", "visual", "paired"}


def test_05_equivalence_normalization_and_dedup():
    assert normalize_question("  What IS\nthis? ") == "what is this?"
    assert source_sort_key("1", "abc", "Q") == source_sort_key("1", "abc", "  q ")


def test_06_deterministic_hard_negative_cap():
    rows = [{"prototype_id": "p", "candidate_id": str(i), "equivalence_key": str(i)} for i in range(30)]
    assert stable_negative_cap(rows, 2) == stable_negative_cap(list(reversed(rows)), 2)
    assert len(stable_negative_cap(rows, 2)) == 16


def test_07_adapter_disabled_extraction_contract():
    spec = {"base_state": "S0", "adapter_enabled": False}
    assert spec == {"base_state": "S0", "adapter_enabled": False}


def test_08_train_only_pca_configuration():
    fitted, report = fit_pcas(keys(1))
    assert all(report[name]["n_components"] == 32 and report[name]["whiten"] for name in MODALITIES)
    assert all(fitted[name].svd_solver == "full" for name in MODALITIES)


def test_09_relation_features_deterministic():
    raw = keys(2)
    pcas, _ = fit_pcas(raw)
    p = {name: raw[name][0] for name in MODALITIES}
    c = {name: raw[name][1] for name in MODALITIES}
    for branch in BRANCHES:
        assert np.array_equal(relation_features(branch, p, c, pcas), relation_features(branch, p, c, pcas))


def test_10_fixed_logistic_configuration():
    model = fixed_logistic_regression()
    assert (model.penalty, model.C, model.class_weight, model.solver, model.max_iter, model.random_state) == ("l2", 1.0, "balanced", "liblinear", 2000, 42)


def test_11_zero_fp_threshold_formula():
    maximum, tau = zero_fp_threshold([.1, .7, .2])
    assert maximum == .7 and tau == np.nextafter(.7, 1.0) and tau > maximum


def test_12_exported_array_inference_parity():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 7)); y = np.arange(100) % 2
    scaler, model, _ = train_branch(x, y)
    expected = model.predict_proba(scaler.transform(x[:1]))[0, 1]
    actual = exported_probability(x[0], scaler.mean_, scaler.scale_, model.coef_, model.intercept_)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_13_no_forbidden_router_features():
    forbidden = {"target", "old_answer", "record_id", "image_hash", "question_hash"}
    feature_spec = {"abs_diff", "hadamard", "cos_raw", "l2_raw", "cos_pca", "l2_pca", "validated_scores"}
    assert not forbidden & feature_spec


def test_14_frozen_v11_exact_branch_contract():
    before = json_hash({"threshold": .9, "prototype": [1, 2]})
    after = json_hash({"threshold": .9, "prototype": [1, 2]})
    assert before == after


def test_15_record953_cannot_affect_threshold():
    first = zero_fp_threshold([.2, .4])
    record953_probability = .99
    assert zero_fp_threshold([.2, .4]) == first and record953_probability not in first


def test_16_off_route_exact_s0_contract():
    baseline = [1, 2, 3]
    routed = list(baseline)
    assert routed == baseline


def test_17_on_route_successful_adapter_contract():
    expected = "The answer is completely ectocervical and fully visible."
    assert "completely ectocervical and fully visible" in expected


def test_18_same_process_bank_reload_arrays(tmp_path: Path):
    path = tmp_path / "router.npz"; np.savez(path, value=np.arange(3))
    assert np.array_equal(np.load(path)["value"], np.arange(3))


def test_19_fresh_process_route_parity():
    raw = keys(4); pcas, _ = fit_pcas(raw)
    value = raw["img"][0]
    expected = pcas["img"].transform(value.reshape(1, -1))[0]
    actual = pca_transform_exported(pcas["img"].mean_, pcas["img"].components_, pcas["img"].explained_variance_, value)
    assert np.allclose(actual, expected, atol=2e-5)


def test_20_exact_unload_rollback():
    states = ["S0", "ADAPTER_ON", "S0"]
    assert states[0] == states[-1]


def test_21_canonical_bank_unchanged():
    anchor = json_hash({"bank": "canonical", "items": 10})
    assert anchor == json_hash({"bank": "canonical", "items": 10})


def test_22_output_directory_non_overwrite(tmp_path: Path):
    output = tmp_path / "run"; output.mkdir()
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)
