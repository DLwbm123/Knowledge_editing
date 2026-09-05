#!/usr/bin/env python3
"""Build a private, source-evidenced scope census and freeze pilot roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def image_identity(value: str) -> str:
    image = Path(value)
    return (image.parent.name if image.name.lower() == "source.jpg" else image.name).lower()


def source_key(dataset: str, image: str, question: str, answer: str) -> tuple[str, str, str, str]:
    return dataset.upper(), image_identity(image), normalize(question), normalize(answer)


def scan_candidates(
    primary: dict[str, Any], source_rows: list[dict[str, Any]], excluded: set[tuple[str, str, str, str]],
    image_root: Path | None = None,
) -> list[dict[str, Any]]:
    primary_image = image_identity(primary["relative_image_path"])
    candidates = []
    for source_index, row in enumerate(source_rows):
        image = str(row.get("img_name") or "")
        if source_key("SLAKE", image, row.get("question", ""), row.get("answer", "")) in excluded:
            continue
        same_image = image_identity(image) == primary_image
        same_question = normalize(row.get("question")) == normalize(primary["question"])
        same_answer = normalize(row.get("answer")) == normalize(primary["gold_answer"])
        triple, base_type = row.get("triple"), row.get("base_type")
        if same_image and same_question and same_answer:
            relation = "same_source_equivalent_candidate"
            evidence = "source QA matches the primary image, question and answer"
            verified = "SOURCE_EQUIVALENCE_CANDIDATE__TEXT_REVIEW_REQUIRED"
        elif same_image and not same_question and triple and triple != primary.get("source_triple"):
            relation = "same_image_other_source_fact"
            evidence = "source QA on the primary image has a different annotated relation and question"
            verified = "SOURCE_RELATION_VERIFIED"
        elif same_question and not same_image and not same_answer and triple:
            relation = "same_question_different_image_conflicting_source_answer"
            evidence = "source QA repeats the question on another image and supplies a distinct annotated answer"
            verified = "SOURCE_RELATION_VERIFIED"
        elif not same_image and not same_question and not same_answer and triple and base_type and triple != primary.get("source_triple"):
            relation = "broad_unrelated_source_qa"
            evidence = "source QA has a different image, question, answer and annotated relation"
            verified = "SOURCE_RELATION_VERIFIED"
        else:
            continue
        candidates.append({
            "source_dataset": "SLAKE", "source_split": "train", "source_index": source_index,
            "source_qid": row.get("qid"), "image_name": image,
            "image_path": str(image_root / image) if image_root else image,
            "question": row.get("question"), "source_answer": row.get("answer"),
            "source_triple": triple, "source_base_type": base_type,
            "fact_relation": relation, "evidence_type": "original_source_annotation", "evidence": evidence,
            "source_annotation_status": "FOUND", "scope_relation_verification_status": verified,
            "verification_status": verified + "__EQKEY_PENDING", "role": "UNASSIGNED_PRE_EQKEY",
        })
    return sorted(candidates, key=lambda row: (str(row["source_qid"]), row["source_index"]))


def stratified_cap(candidates: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    order = ("same_source_equivalent_candidate", "same_image_other_source_fact", "same_question_different_image_conflicting_source_answer", "broad_unrelated_source_qa")
    ordered = [row for relation in order for row in candidates if row["fact_relation"] == relation]
    return ordered[:limit]


def select_candidates(
    primary: dict[str, Any], source_rows: list[dict[str, Any]], excluded: set[tuple[str, str, str, str]], limit: int = 120,
) -> list[dict[str, Any]]:
    return stratified_cap(scan_candidates(primary, source_rows, excluded), limit)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        result[row["fact_relation"]] = result.get(row["fact_relation"], 0) + 1
    return result


def freeze_roles(census: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    candidates = review.get("candidates", [])
    if len(candidates) > 24 or review.get("review_input_visibility") != "SOURCE_QUESTION_ONLY__NO_MODEL_OUTPUTS":
        raise RuntimeError("augmentation review boundary is invalid")
    approved = [row for row in candidates if row.get("review_status") == "APPROVED_EQUIVALENT"]
    roles = ("fit", "calibration", "evaluation")
    positives = {role: [row for row in approved if row.get("role") == role] for role in roles}
    if any(len(positives[role]) != 4 for role in roles):
        raise RuntimeError("scope pilot requires four reviewed positives per role")
    families = {role: {row["rewrite_family"] for row in rows} for role, rows in positives.items()}
    if any(families[a] & families[b] for a, b in (("fit", "calibration"), ("fit", "evaluation"), ("calibration", "evaluation"))):
        raise RuntimeError("rewrite families cross scope roles")
    verified = [row for row in census["negative_candidates_full"] if row["scope_relation_verification_status"] == "SOURCE_RELATION_VERIFIED"]
    by_image = {}
    for row in verified:
        by_image.setdefault((row["source_dataset"], image_identity(row["image_name"])), row)
    ordered = sorted(by_image.values(), key=lambda row: hashlib.sha256(
        f"{census['primary']['record_id']}\0{row['source_dataset']}\0{image_identity(row['image_name'])}".encode()
    ).hexdigest())
    if len(ordered) < 60:
        raise RuntimeError(f"only {len(ordered)} source-image-disjoint verified negatives are available")
    negatives = {role: [dict(row, role=role) for row in ordered[offset:offset + 20]] for role, offset in (("fit", 0), ("calibration", 20), ("evaluation", 40))}
    return {
        "schema_version": "medtrace-scope-pilot-role-freeze-private-v1",
        "status": "EXPLORATORY_EVALUABLE__EQKEY_PENDING",
        "method": "MEDTRACE_NATIVE_TEXT_SCOPE_AUGMENTATION_PILOT",
        "primary": census["primary"], "augmentation": review,
        "positives": positives, "negatives": negatives,
        "role_counts": {role: {"positive": 4, "negative": 20} for role in roles},
        "patient_disjoint_claimed": False,
        "selection": "stable hash over primary ID, source dataset and full source-image identity; one row per image; no model outputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-inputs", type=Path, required=True)
    parser.add_argument("--qual-inputs", type=Path, required=True)
    parser.add_argument("--formal-records", type=Path, required=True)
    parser.add_argument("--formal-probes", type=Path, required=True)
    parser.add_argument("--slake-train", type=Path, required=True)
    parser.add_argument("--slake-validation", type=Path, required=True)
    parser.add_argument("--slake-test", type=Path, required=True)
    parser.add_argument("--vqarad", type=Path, required=True)
    parser.add_argument("--primary-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--augmentation-review", type=Path)
    parser.add_argument("--roles-out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if bool(args.augmentation_review) != bool(args.roles_out):
        raise RuntimeError("augmentation review and roles output must be supplied together")
    dev = read_jsonl(args.dev_inputs)
    primary_event = next((row for row in dev if row["edit_record"]["record_id"] == args.primary_id), None)
    if primary_event is None:
        raise RuntimeError("primary edit is absent from frozen DEV16")
    primary = dict(primary_event["edit_record"])
    if primary["dataset"].upper() != "SLAKE":
        raise RuntimeError("this bounded census expects the frozen SLAKE primary")
    slake = {name: json.loads(path.read_text()) for name, path in {"train": args.slake_train, "validation": args.slake_validation, "test": args.slake_test}.items()}
    vqarad = json.loads(args.vqarad.read_text())
    primary_image = image_identity(primary["relative_image_path"])
    matches = [(split_name, row) for split_name, rows in slake.items() for row in rows if image_identity(str(row.get("img_name") or "")) == primary_image and normalize(row.get("question")) == normalize(primary["question"]) and normalize(row.get("answer")) == normalize(primary["gold_answer"])]
    if len(matches) != 1 or matches[0][0] != "train":
        raise RuntimeError("primary source QA did not resolve uniquely to SLAKE train")
    source = matches[0][1]
    primary |= {"source_triple": source.get("triple"), "source_base_type": source.get("base_type")}
    excluded: set[tuple[str, str, str, str]] = set()
    for event in read_jsonl(args.qual_inputs):
        edit = event["edit_record"]
        excluded.add(source_key(edit["dataset"], edit["relative_image_path"], edit["question"], edit["gold_answer"]))
        excluded.update(source_key(row["dataset"], row["image_path"], row["question"], row["reference"]) for row in event["probes"])
    for row in read_jsonl(args.formal_records):
        excluded.add(source_key(row["dataset"], row["relative_image_path"], row["question"], row["gold_answer"]))
    excluded.update(source_key(row["dataset"], row["image_path"], row["question"], row["reference"]) for row in read_jsonl(args.formal_probes))
    full = scan_candidates(primary, slake["train"], excluded, args.slake_train.parent / "imgs")
    retained = stratified_cap(full)
    payload = {
        "schema_version": "medtrace-scope-source-census-private-v2", "status": "SOURCE_CENSUS_V2_COMPLETE",
        "primary": {"record_id": primary["record_id"], "dataset": "SLAKE", "source_split": "train", "source_qid": source.get("qid"), "image_name": primary_image, "image_path": primary["image_path"], "question": primary["question"], "target": primary["gold_answer"], "official_rephrase": primary["official_rephrase"], "source_triple": source.get("triple")},
        "source_counts": {"slake_train": len(slake["train"]), "slake_validation": len(slake["validation"]), "slake_test": len(slake["test"]), "vqarad_all": len(vqarad)},
        "search_scope": {"loaded_inventory_rows": sum(map(len, slake.values())) + len(vqarad), "actually_searched_allowed_rows": len(slake["train"]), "allowed_supervision_sources": ["SLAKE/train"], "audit_only_sources": ["SLAKE/validation", "SLAKE/test", "VQA-RAD/all"]},
        "input_sha256": {name: sha256_file(path) for name, path in {"dev_inputs": args.dev_inputs, "qual_inputs": args.qual_inputs, "formal_records": args.formal_records, "formal_probes": args.formal_probes, "slake_train": args.slake_train, "slake_validation": args.slake_validation, "slake_test": args.slake_test, "vqarad": args.vqarad}.items()},
        "excluded_model_visible_source_keys": len(excluded),
        "positive_inventory": {"native": 1, "official_rephrase": int(bool(primary.get("official_rephrase"))), "new_source_equivalent_records": counts(full).get("same_source_equivalent_candidate", 0), "new_source_equivalent_search_status": "SEARCHED__TEXT_REVIEW_REQUIRED", "augmentation_required_for_role_isolated_positive_sets": True},
        "eligible_candidate_count_before_cap": len(full), "eligible_candidates_by_relation_before_cap": counts(full),
        "retained_candidate_count_after_cap": len(retained), "retained_candidates_by_relation_after_cap": counts(retained),
        "negative_candidates": retained, "negative_candidates_full": full,
        "role_freeze_status": "PENDING_EQKEY_AND_POSITIVE_AUGMENTATION", "scope_scoring_started": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    if args.augmentation_review:
        if args.roles_out.exists():
            raise FileExistsError(args.roles_out)
        roles = freeze_roles(payload, json.loads(args.augmentation_review.read_text()))
        temporary = args.roles_out.with_suffix(args.roles_out.suffix + ".tmp")
        temporary.write_text(json.dumps(roles, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.roles_out)


if __name__ == "__main__":
    main()
