#!/usr/bin/env python3
"""Mechanically normalize private professional-review rows into C0R verdicts."""

import collections
import random
import re

from launch_local_review_fast import PRESETS
from resolve_authorized_images import outcome


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


def compare_pilot_overlap(pilot_rows, expert_rows):
    expert = {row["review_id"]: row for row in expert_rows}
    fields = ("valid", "confidence", "relation", "issue_type", "recommended_action", "reason")
    structured = []
    outcome_conflicts = []
    for pilot in pilot_rows:
        current = expert.get(pilot["review_id"])
        if current is None or any(pilot[key] != current[key] for key in fields):
            structured.append(pilot["review_id"])
        if current is None or outcome(pilot) != outcome(current):
            outcome_conflicts.append(pilot["review_id"])
    return {"overlap": len(pilot_rows), "structured_difference_ids": structured, "outcome_conflict_ids": outcome_conflicts}


def build_blind_reviewer_b_selection(focus_ids, verdicts, dataset_by_id, selection_seed, queue_seed, control_count=20):
    known = {row["review_id"] for row in verdicts}
    if not set(focus_ids) <= known or known != set(dataset_by_id):
        raise ValueError("reviewer B bindings changed")
    mandatory = set(focus_ids)
    mandatory.update(row["review_id"] for row in verdicts if outcome(row) != "VALID" or row["confidence"] != "high")
    groups = collections.defaultdict(list)
    for row in verdicts:
        rid = row["review_id"]
        if rid not in mandatory and outcome(row) == "VALID" and row["confidence"] == "high":
            groups[dataset_by_id[rid]].append(rid)
    rng = random.Random(selection_seed)
    for group in groups.values():
        rng.shuffle(group)
    controls = []
    while len(controls) < control_count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(controls) < control_count:
                controls.append(groups[key].pop())
    if len(controls) != control_count:
        raise ValueError("insufficient high-valid controls")
    queue = sorted(mandatory | set(controls))
    random.Random(queue_seed).shuffle(queue)
    return {"mandatory_ids": sorted(mandatory), "control_ids": controls, "queue": queue}
