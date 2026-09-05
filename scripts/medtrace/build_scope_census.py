#!/usr/bin/env python3
"""Build a private, source-evidenced scope census without scoring a model."""

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


def select_candidates(
    primary: dict[str, Any], source_rows: list[dict[str, Any]], excluded: set[tuple[str, str, str, str]], limit: int = 120,
) -> list[dict[str, Any]]:
    primary_image = image_identity(primary["relative_image_path"])
    candidates = []
    for source_index, row in enumerate(source_rows):
        image = str(row.get("img_name") or "")
        key = source_key("SLAKE", image, row.get("question", ""), row.get("answer", ""))
        if key in excluded or normalize(row.get("answer")) == normalize(primary["gold_answer"]):
            continue
        if image_identity(image) == primary_image:
            relation = "same_image_other_source_fact"
            evidence = "source QA on the primary image with a different question and source answer"
        elif normalize(row.get("question")) == normalize(primary["question"]):
            relation = "same_question_different_image_conflicting_source_answer"
            evidence = "source QA repeats the question on another image with a different source answer"
        elif row.get("triple") != primary.get("source_triple") and row.get("base_type") != primary.get("source_base_type"):
            relation = "broad_unrelated_source_qa"
            evidence = "source QA has a different image, structured triple, base type, question, and answer"
        else:
            continue
        candidates.append({
            "source_dataset": "SLAKE", "source_split": "train", "source_index": source_index,
            "source_qid": row.get("qid"), "image_name": image, "question": row.get("question"),
            "source_answer": row.get("answer"), "source_triple": row.get("triple"),
            "fact_relation": relation, "evidence_type": "original_source_annotation",
            "evidence": evidence, "verification_status": "SOURCE_LABEL_VERIFIED__EQKEY_PENDING",
            "role": "UNASSIGNED_PRE_EQKEY",
        })
    candidates.sort(key=lambda row: (row["fact_relation"], str(row["source_qid"]), row["source_index"]))
    return candidates[:limit]


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
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    dev = read_jsonl(args.dev_inputs)
    primary_event = next((row for row in dev if row["edit_record"]["record_id"] == args.primary_id), None)
    if primary_event is None:
        raise RuntimeError("primary edit is absent from frozen DEV16")
    primary = primary_event["edit_record"]
    if primary["dataset"].upper() != "SLAKE":
        raise RuntimeError("this bounded census expects the frozen SLAKE primary")
    slake = {
        "train": json.loads(args.slake_train.read_text()),
        "validation": json.loads(args.slake_validation.read_text()),
        "test": json.loads(args.slake_test.read_text()),
    }
    primary_image = image_identity(primary["relative_image_path"])
    matches = [
        (split, row) for split, rows in slake.items() for row in rows
        if image_identity(str(row.get("img_name") or "")) == primary_image
        and normalize(row.get("question")) == normalize(primary["question"])
        and normalize(row.get("answer")) == normalize(primary["gold_answer"])
    ]
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
    candidates = select_candidates(primary, slake["train"], excluded)
    by_relation: dict[str, int] = {}
    for row in candidates:
        by_relation[row["fact_relation"]] = by_relation.get(row["fact_relation"], 0) + 1
    payload = {
        "schema_version": "medtrace-scope-source-census-private-v1",
        "status": "SOURCE_CENSUS_COMPLETE__POSITIVE_AUGMENTATION_REQUIRED",
        "primary": {
            "record_id": primary["record_id"], "dataset": "SLAKE", "source_split": "train",
            "source_qid": source.get("qid"), "image_name": primary_image,
            "question": primary["question"], "target": primary["gold_answer"],
            "official_rephrase": primary["official_rephrase"], "source_triple": source.get("triple"),
        },
        "source_counts": {
            "slake_train": len(slake["train"]), "slake_validation": len(slake["validation"]),
            "slake_test": len(slake["test"]), "vqarad_all": len(json.loads(args.vqarad.read_text())),
        },
        "input_sha256": {name: sha256_file(path) for name, path in {
            "dev_inputs": args.dev_inputs, "qual_inputs": args.qual_inputs,
            "formal_records": args.formal_records, "formal_probes": args.formal_probes,
            "slake_train": args.slake_train, "slake_validation": args.slake_validation,
            "slake_test": args.slake_test, "vqarad": args.vqarad,
        }.items()},
        "excluded_model_visible_source_keys": len(excluded),
        "positive_inventory": {
            "native": 1, "official_rephrase": int(bool(primary.get("official_rephrase"))),
            "new_source_equivalent_records": 0, "augmentation_required_for_role_isolated_positive_sets": True,
        },
        "negative_candidate_count": len(candidates), "negative_candidates_by_relation": by_relation,
        "negative_candidates": candidates,
        "role_freeze_status": "PENDING_EQKEY_AND_POSITIVE_AUGMENTATION",
        "scope_scoring_started": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)


if __name__ == "__main__":
    main()
