#!/usr/bin/env python3
"""CPU-only helpers for a single-reviewer target amendment plan."""

import collections

from resolve_authorized_images import outcome as reviewer_outcome


def classify_nonretained(authoritative_repair_available, frozen_reserve_available):
    if authoritative_repair_available:
        return "AUTHORITATIVE_SOURCE_REPAIR_AVAILABLE"
    if frozen_reserve_available:
        return "PRE_FORMAL_FROZEN_RESERVE_AVAILABLE"
    return "EXCLUSION_REQUIRED"


def final_census(verdicts):
    counts = collections.Counter(reviewer_outcome(row) for row in verdicts)
    return {key: counts[key] for key in ("VALID", "CONFIRMED_INVALID", "UNRESOLVED")}


def calculate_event_impact(events, nonretained_positions):
    positions = set(nonretained_positions)
    if not positions:
        raise ValueError("at least one nonretained position is required")
    earliest = min(positions)
    single = [row for row in events if row["mode"] == "single" and row["formal_edit_position"] in positions]
    sequential = [row for row in events if row["mode"] == "sequential" and int(row["checkpoint"]) >= earliest]
    single_groups = collections.Counter(row["anonymous_group_id"] for row in single)
    sequential_groups = collections.Counter(row["anonymous_group_id"] for row in sequential)
    return {
        "earliest_affected_position": earliest,
        "affected_single_events": len(single),
        "affected_sequential_events": len(sequential),
        "single_per_group": dict(sorted(single_groups.items())),
        "sequential_per_group": dict(sorted(sequential_groups.items())),
        "single_group_symmetric": len(single_groups) == 4 and len(set(single_groups.values())) == 1,
        "sequential_group_symmetric": len(sequential_groups) == 4 and len(set(sequential_groups.values())) == 1,
    }
