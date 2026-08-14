from scripts.engram.run_engram_v2_10edit_generation_gate import (
    classify_decision,
    correctness_score,
    exact_and_contains,
    metadata_prefix_equal,
    repeated_ngram,
)


def passing_summary():
    return {
        "base_exact": 0,
        "base_contains": 0,
        "engram_exact": 1,
        "engram_contains": 1,
        "toward": 1,
        "away": 0,
        "locality_token_agreement": 9,
        "locality_normalized_agreement": 9,
        "locality_correctness_losses": 0,
        "fresh_target_agreement": 10,
        "fresh_locality_agreement": 10,
        "state_verified": True,
        "finite": True,
    }


def test_exact_contains_and_score_follow_shared_normalization():
    exact, contains = exact_and_contains("Answer: Benign melanocyte.", "benign melanocyte")
    assert exact and contains
    assert correctness_score(exact, contains) == 2
    assert correctness_score(False, True) == 1
    assert correctness_score(False, False) == 0


def test_repeated_ngram_only_flags_consecutive_repetition():
    assert repeated_ngram("lung disease lung disease")
    assert not repeated_ngram("lung disease with chronic changes")


def test_generation_gate_requires_transfer_locality_and_fresh_process():
    summary = passing_summary()
    assert classify_decision(summary) == "ENGRAM_V2_10EDIT_GENERATION_PASS"
    for key, value in (
        ("toward", 0),
        ("locality_token_agreement", 8),
        ("fresh_target_agreement", 9),
        ("state_verified", False),
    ):
        failed = dict(summary)
        failed[key] = value
        assert classify_decision(failed) == "ENGRAM_V2_10EDIT_GENERATION_FAIL"


def test_checker_metadata_is_prefix_truncated_before_comparison():
    final_metadata = ["edit1", "edit2", "edit3"]
    assert metadata_prefix_equal(final_metadata, ["edit1"], 1)
    assert metadata_prefix_equal(final_metadata, ["edit1", "edit2"], 2)
    assert not metadata_prefix_equal(final_metadata, final_metadata, 2)
