import pytest

from methods.liveedit_med.blind_set_builder import freeze_candidates


def test_blind_freeze_excludes_observed_and_deduplicates_eqkey():
    rows = [
        {"record_id": "old", "selection_hash": "0", "router_input_equivalence_key": "a"},
        {"record_id": "1", "selection_hash": "1", "router_input_equivalence_key": "b"},
        {"record_id": "2", "selection_hash": "2", "router_input_equivalence_key": "b"},
        {"record_id": "3", "selection_hash": "3", "router_input_equivalence_key": "c"},
    ]
    result = freeze_candidates(rows, excluded_ids={"old"}, count=2)
    assert [row["record_id"] for row in result["selected"]] == ["1", "3"]
    assert len(result["manifest_hash"]) == 64


def test_blind_freeze_refuses_to_pad():
    with pytest.raises(RuntimeError, match="INSUFFICIENT_TRULY_UNUSED"):
        freeze_candidates([{"record_id": "1", "selection_hash": "1", "router_input_equivalence_key": "x"}], excluded_ids=set(), count=2)
