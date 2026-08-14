"""Leakage-safe deterministic future-blind set selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_candidates(candidates: Sequence[Mapping[str, Any]], *, excluded_ids: set[str], count: int = 16) -> dict[str, Any]:
    eligible = [dict(row) for row in candidates if str(row["record_id"]) not in excluded_ids]
    eligible.sort(key=lambda row: (str(row["selection_hash"]), str(row["record_id"])))
    seen_ids: set[str] = set()
    seen_eq: set[str] = set()
    selected = []
    for row in eligible:
        rid, eqkey = str(row["record_id"]), str(row["router_input_equivalence_key"])
        if rid in seen_ids or eqkey in seen_eq:
            continue
        seen_ids.add(rid); seen_eq.add(eqkey); selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"LIVEEDIT_MED_INSUFFICIENT_TRULY_UNUSED_BLIND_EDITS:{len(selected)}<{count}")
    manifest = {
        "protocol": "FUTURE_BLIND_MEDICAL_SET_V1", "selection_rule": "stable_hash_then_record_id_unique_eqkey",
        "requested_count": count, "selected_count": len(selected), "excluded_id_count": len(excluded_ids),
        "edited_checkpoint_loaded": False, "selected": selected,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest


def collect_observed_ids(source_records: Mapping[str, Any], extra_ids: Sequence[str] = ()) -> set[str]:
    result = {str(row["record_id"]) for rows in source_records["records"].values() for row in rows}
    result.update(map(str, extra_ids))
    return result
