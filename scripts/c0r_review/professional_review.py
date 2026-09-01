#!/usr/bin/env python3
"""Mechanically normalize private professional-review rows into C0R verdicts."""

import re

from launch_local_review_fast import PRESETS


REVIEW_ID = re.compile(r"^target_review_[0-9]{4}$")


def normalize_row(row):
    review_id = str(row.get("review_id", "")).strip()
    code = str(row.get("verdict_code", "")).strip()
    note = str(row.get("note", "")).strip()
    if not REVIEW_ID.fullmatch(review_id):
        raise ValueError("invalid review_id")
    if code == "6":
        raise ValueError("custom verdict requires explicit structured human fields")
    if code not in PRESETS:
        raise ValueError("verdict_code must be 1..5")
    if len(note) > 2000:
        raise ValueError("human note exceeds output schema limit")
    preset = PRESETS[code]
    verdict = {
        "review_id": review_id,
        "valid": preset["valid"] == "true",
        "confidence": preset["confidence"],
        "relation": preset["relation"],
        "issue_type": preset["issue_type"],
        "recommended_action": preset["recommended_action"],
        "reason": note or preset["reason"],
    }
    if code in {"4", "5"}:
        if len(note) < 12:
            raise ValueError("invalid/uncertain verdict requires a 12+ character human note")
        verdict["issue_type"] = "other"
    return verdict


def normalize_rows(rows, expected_count=200):
    rows = list(rows)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows")
    if [row.get("review_sequence") for row in rows] != list(range(1, expected_count + 1)):
        raise ValueError("review sequence changed")
    verdicts = [normalize_row(row) for row in rows]
    if len({row["review_id"] for row in verdicts}) != expected_count:
        raise ValueError("duplicate review_id")
    return verdicts
