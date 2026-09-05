#!/usr/bin/env python3
"""Freeze DEV16 paraphrases and hard-aware scope roles before model scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.medtrace.build_scope_census import image_identity  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def text_key(value: str) -> str:
    return "".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def stable_order(primary_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{primary_id}\0{row['source_dataset']}\0{image_identity(row['image_name'])}\0{row['source_qid']}".encode()
    ).hexdigest())


def validate_augmentation(dev: list[dict[str, Any]], review: dict[str, Any], qual: list[dict[str, Any]], old_roles: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    if review.get("generation", {}).get("input_visibility") != "SOURCE_QUESTION_ONLY__NO_TARGET__NO_MODEL_OUTPUTS__NO_PROBE_SCORES":
        raise RuntimeError("generality augmentation visibility boundary is invalid")
    records = {row["record_id"]: row for row in review.get("records", [])}
    expected = {event["edit_record"]["record_id"] for event in dev}
    if set(records) != expected or len(records) != 16:
        raise RuntimeError("augmentation must cover all 16 DEV edits exactly once")
    forbidden = set()
    for event in dev:
        edit = event["edit_record"]
        forbidden.update(text_key(value) for value in (edit["question"], edit["official_rephrase"]))
        forbidden.update(text_key(row["question"]) for row in event["probes"])
    for event in qual:
        forbidden.add(text_key(event["edit_record"]["question"]))
        forbidden.update(text_key(row["question"]) for row in event["probes"])
    for role in ("calibration", "evaluation"):
        forbidden.update(text_key(row["question"]) for row in old_roles["positives"][role])
    result: dict[str, list[dict[str, str]]] = {}
    for record_id, row in records.items():
        candidates = row.get("candidates", [])
        keys = [text_key(item["question"]) for item in candidates]
        if len(candidates) != 4 or len(set(keys)) != 4 or any(key in forbidden for key in keys):
            raise RuntimeError(f"invalid, duplicate or forbidden paraphrase for {record_id}")
        if any(not item.get("family") or not item.get("question", "").strip() for item in candidates):
            raise RuntimeError(f"incomplete paraphrase evidence for {record_id}")
        result[record_id] = [{"question": item["question"], "family": item["family"], "review_status": "APPROVED_EQUIVALENT"} for item in candidates]
    return result


def freeze_scope_roles(census: dict[str, Any], old_roles: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    candidates = {int(row["source_qid"]): row for row in census["negative_candidates_full"] if row.get("source_qid") is not None}
    hard_ids = set(map(int, review["same_question_different_image"]["approved_qids"]))
    challenge_ids = set(map(int, review["same_image_challenge"]["approved_qids"]))
    excluded_ids = {int(row["qid"]) for row in review["excluded"]}
    if hard_ids & challenge_ids or hard_ids & excluded_ids or challenge_ids & excluded_ids:
        raise RuntimeError("hard review categories overlap")
    if hard_ids | challenge_ids | excluded_ids != {
        int(row["source_qid"]) for row in census["negative_candidates_full"]
        if row["fact_relation"] in {"same_question_different_image_conflicting_source_answer", "same_image_other_source_fact"}
    }:
        raise RuntimeError("hard review does not exactly cover recovered candidates")
    hard = [candidates[qid] for qid in hard_ids]
    challenge = [candidates[qid] for qid in challenge_ids]
    if any(row["fact_relation"] != "same_question_different_image_conflicting_source_answer" or row["scope_relation_verification_status"] != "SOURCE_RELATION_VERIFIED" for row in hard):
        raise RuntimeError("invalid same-question hard-negative evidence")
    if any(row["fact_relation"] != "same_image_other_source_fact" or row["scope_relation_verification_status"] != "SOURCE_RELATION_VERIFIED" for row in challenge):
        raise RuntimeError("invalid same-image challenge evidence")
    primary_id = census["primary"]["record_id"]
    hard = stable_order(primary_id, hard)
    hard_by_role = {"fit": hard[:4], "calibration": hard[4:8], "evaluation": hard[8:13]}
    if {role: len(rows) for role, rows in hard_by_role.items()} != {"fit": 4, "calibration": 4, "evaluation": 5}:
        raise RuntimeError("insufficient independent same-question hard groups")
    hard_groups = [image_identity(row["image_name"]) for row in hard]
    if len(hard_groups) != len(set(hard_groups)):
        raise RuntimeError("same-question hard source group reused")

    old_groups = {
        image_identity(row["image_name"])
        for rows in old_roles["negatives"].values() for row in rows
    }
    by_group: dict[str, dict[str, Any]] = {}
    for row in census["negative_candidates_full"]:
        group = image_identity(row["image_name"])
        if row["fact_relation"] == "broad_unrelated_source_qa" and row["scope_relation_verification_status"] == "SOURCE_RELATION_VERIFIED" and group not in old_groups and group not in hard_groups:
            by_group.setdefault(group, row)
    broad = stable_order(primary_id, list(by_group.values()))
    if len(broad) < 51:
        raise RuntimeError("insufficient source-group-disjoint broad negatives")
    broad_fit, broad_cal, broad_eval = broad[:20], broad[20:36], broad[36:51]
    mixed_fit = hard_by_role["fit"] + broad_fit[:16]
    calibration = hard_by_role["calibration"] + broad_cal
    evaluation = hard_by_role["evaluation"] + broad_eval
    all_role_groups = {
        "fit": {image_identity(row["image_name"]) for row in broad_fit + hard_by_role["fit"]},
        "calibration": {image_identity(row["image_name"]) for row in calibration},
        "evaluation": {image_identity(row["image_name"]) for row in evaluation},
    }
    if any(all_role_groups[a] & all_role_groups[b] for a, b in (("fit", "calibration"), ("fit", "evaluation"), ("calibration", "evaluation"))):
        raise RuntimeError("negative source group crosses roles")
    return {
        "schema_version": "medtrace-hard-scope-roles-private-v1",
        "status": "FROZEN_BEFORE_ACTIVATION_SCORING__EQKEY_PENDING",
        "primary": old_roles["primary"],
        "positives": old_roles["positives"],
        "negative_roles": {
            "broad_fit_control": broad_fit,
            "mixed_fit": mixed_fit,
            "calibration": calibration,
            "evaluation": evaluation,
            "same_image_challenge": stable_order(primary_id, challenge),
        },
        "selection": "source-group first; stable hash; same-question hard quota 4/4/5; mixed broad is a 16-row subset of the 20-row broad-fit control; no model outputs or scores",
        "patient_disjoint_claimed": False,
        "hard_review": review,
    }


def relation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["fact_relation"] for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-inputs", type=Path, required=True)
    parser.add_argument("--qual-inputs", type=Path, required=True)
    parser.add_argument("--augmentation-review", type=Path, required=True)
    parser.add_argument("--old-scope-roles", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--hard-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-counts", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.public_counts.exists():
        raise FileExistsError("ablation data output already exists")
    dev, qual = read_jsonl(args.dev_inputs), read_jsonl(args.qual_inputs)
    augmentation_review = json.loads(args.augmentation_review.read_text())
    old_roles = json.loads(args.old_scope_roles.read_text())
    census = json.loads(args.census.read_text())
    hard_review = json.loads(args.hard_review.read_text())
    paraphrases = validate_augmentation(dev, augmentation_review, qual, old_roles)
    scope = freeze_scope_roles(census, old_roles, hard_review)
    private = {
        "schema_version": "medtrace-generality-hard-scope-data-private-v1",
        "status": "FROZEN_BEFORE_MODEL_SCORING__EQKEY_PENDING",
        "generality_paraphrases": paraphrases,
        "generality_review": augmentation_review,
        "scope": scope,
    }
    atomic_json(args.out, private)
    counts = {
        "schema_version": "medtrace-generality-hard-scope-role-counts-public-v1",
        "status": "FROZEN_BEFORE_MODEL_SCORING__EQKEY_PENDING",
        "generality": {"edit_count": len(paraphrases), "approved_paraphrases": sum(map(len, paraphrases.values())), "per_edit": 4, "independent_clinical_review": False},
        "scope": {
            "positive_counts": {role: len(scope["positives"][role]) for role in ("fit", "calibration", "evaluation")},
            "negative_counts": {role: len(scope["negative_roles"][role]) for role in scope["negative_roles"]},
            "negative_relation_counts": {role: relation_counts(rows) for role, rows in scope["negative_roles"].items()},
            "same_question_hard_group_counts": {"fit": 4, "calibration": 4, "evaluation": 5},
            "excluded_same_fact_cross_language": len(hard_review["excluded"]),
            "eqkey_status": "PENDING_MODEL_PREPROCESSING",
            "patient_disjoint_claimed": False,
        },
        "private_artifacts_withheld": True,
    }
    if len(hard_review["excluded"]) != 1:
        raise RuntimeError("unexpected hard-review exclusion count")
    atomic_json(args.public_counts, counts)


if __name__ == "__main__":
    main()
