import pytest

from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir
from scripts.engram.stage1p_preflight_audit_utils import (
    CLASS_MODEL_KNOWN,
    assert_bank_hash_unchanged,
    classify_record,
    raw_row_for_id,
    resolve_answer_mapping,
    select_first_model_known,
    target_absent_from_prompt,
    verify_image_hash_propagation,
    verify_raw_processed_pairing,
)


def record(record_id=953):
    return {
        "id": record_id,
        "src": "question",
        "pred": "old answer",
        "alt": "new answer",
        "rephrase": "question again",
        "image": "category/part6_row_1.png",
        "image_rephrase": "category/part6_row_2.png",
        "loc": "text loc",
        "loc_ans": "text answer",
        "m_loc": "locality/loc.png",
        "m_loc_q": "image loc",
        "m_loc_a": "image answer",
        "clinical_VQA_task": "diagnosis",
        "department": "medicine",
        "perceptual_granularity": "image",
        "modality": "xray",
        "original_task": "CLS_2D",
        "port_new": [],
    }


def test_raw_source_to_processed_index_consistency():
    raw = record()
    index, found = raw_row_for_id([record(1), raw], "953")
    assert index == 1 and found == raw
    processed = dict(raw, image="images/0000_image_part6_row_1.png", image_rephrase="images/0000_image_rephrase_part6_row_2.png", m_loc="images/0000_m_loc_loc.png")
    assert verify_raw_processed_pairing(raw, processed)["passed"]


def test_image_hash_propagation():
    assert verify_image_hash_propagation("abc", "abc")
    assert not verify_image_hash_propagation("abc", "def")


def test_answer_option_mapping():
    direct = resolve_answer_mapping(record())
    assert direct["resolved"] and not direct["ambiguous"]
    mapped = resolve_answer_mapping(dict(record(), options=["x", "old answer"], answer_index=1))
    assert mapped["resolved"]
    bad = resolve_answer_mapping(dict(record(), options=["x"], answer_index=0))
    assert bad["ambiguous"]


def test_no_target_leakage_into_generation_prompt():
    assert target_absent_from_prompt("Question: what is shown? Short answer:", "new answer")
    assert not target_absent_from_prompt("Question: what is shown? new answer", "new answer")


def test_deterministic_fixed_order_selection():
    order = ["953", "1293", "1592"]
    classifications = {"953": "PAIRING_VALID_MODEL_UNKNOWN", "1293": CLASS_MODEL_KNOWN, "1592": CLASS_MODEL_KNOWN}
    assert select_first_model_known(order, classifications) == "1293"
    assert classify_record(pairing_valid=True, answer_mapping_ambiguous=False, view_matches=[{"token_boundary_contains": True}]) == CLASS_MODEL_KNOWN


def test_output_directory_nonoverwrite(tmp_path):
    path = tmp_path / "audit"
    create_new_output_dir(path)
    with pytest.raises(FileExistsError):
        create_new_output_dir(path)


def test_bank_hash_unchanged():
    assert_bank_hash_unchanged("abc", "abc")
    with pytest.raises(RuntimeError):
        assert_bank_hash_unchanged("abc", "def")
