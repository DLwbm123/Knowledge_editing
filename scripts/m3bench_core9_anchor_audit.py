#!/usr/bin/env python3
"""Audit whether the frozen amended edits can anchor public Core-9 metadata."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("T2L", "T3L", "T3G", "T4L", "T4G")
DROP_REASONS = (
    "anchor_absent",
    "question_role_mismatch",
    "concept_mismatch",
    "paired_image_missing",
    "paired_question_missing",
    "gold_missing",
    "duplicate_relation",
    "unsupported_direction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", "".join(ch if ch.isalnum() else " " for ch in str(value).casefold())).strip()


def phrase_in(needle: object, haystack: object) -> bool:
    needle_tokens = normalize(needle).split()
    haystack_tokens = normalize(haystack).split()
    width = len(needle_tokens)
    return bool(width) and any(haystack_tokens[i : i + width] == needle_tokens for i in range(len(haystack_tokens) - width + 1))


def parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def record_image_id(record: dict) -> str:
    relative = str(record["relative_image_path"])
    return relative.split("/", 1)[0] if str(record["dataset"]).upper() == "SLAKE" else Path(relative).name


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def source_qa_index(metadata_root: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    qas: dict[str, list[dict]] = defaultdict(list)
    datasets: dict[str, str] = {}
    for dataset, name in (("SLAKE", "slake_metadata.csv"), ("VQA-RAD", "vqarad_metadata.csv")):
        for row in read_csv(metadata_root / name):
            image_id = row["image_id"]
            datasets[image_id] = dataset
            for qa in parse_list(row.get("qa_list")):
                if isinstance(qa, dict):
                    qas[image_id].append(qa)
    return qas, datasets


def image_path(image_id: str, datasets: dict[str, str], slake_root: Path, vqarad_root: Path) -> Path | None:
    dataset = datasets.get(image_id)
    if dataset == "SLAKE":
        return slake_root / image_id / "source.jpg"
    if dataset == "VQA-RAD":
        return vqarad_root / image_id
    return None


def add_candidate(state: dict, key: tuple, relation: dict) -> None:
    if key in state["seen"]:
        state["drops"]["duplicate_relation"] += 1
        return
    state["seen"].add(key)
    state["relations"].append(relation)


def task_state() -> dict:
    return {"metadata_rows": [], "relations": [], "drops": Counter(), "seen": set()}


def audit_records(
    records: list[dict], metadata_root: Path, slake_root: Path, vqarad_root: Path
) -> tuple[list[dict], dict]:
    t2l = defaultdict(list)
    for index, row in enumerate(read_csv(metadata_root / "t2l_cross_image_pairs.csv")):
        row = dict(row, _row=index)
        t2l[row["image_id_1"]].append(row)
        t2l[row["image_id_2"]].append(row)
    t3 = defaultdict(list)
    for index, row in enumerate(read_csv(metadata_root / "t3_cross_modality_pairs.csv")):
        t3[row["image_A"]].append(dict(row, _row=index))
    t4l = defaultdict(list)
    for index, row in enumerate(read_csv(metadata_root / "t4l_compositional_locality.csv")):
        t4l[row["image_id"]].append(dict(row, _row=index))
    t4g = defaultdict(list)
    for index, row in enumerate(read_csv(metadata_root / "t4g_compositional_generality.csv")):
        t4g[row["single_image"]].append(dict(row, _row=index))
    source_qas, datasets = source_qa_index(metadata_root)

    details = []
    summary = {
        task: {
            "retained_edits_with_metadata_anchor": 0,
            "candidate_edit_count": 0,
            "candidate_probe_relations": 0,
            "drop_reason_counts": Counter(),
        }
        for task in TASKS
    }

    for record in records:
        image_id = record_image_id(record)
        states = {task: task_state() for task in TASKS}

        rows = t2l.get(image_id, [])
        states["T2L"]["metadata_rows"] = [row["_row"] for row in rows]
        if not rows:
            states["T2L"]["drops"]["anchor_absent"] += 1
        for row in rows:
            paired = row["image_id_2"] if row["image_id_1"] == image_id else row["image_id_1"]
            paired_path = image_path(paired, datasets, slake_root, vqarad_root)
            if paired_path is None or not paired_path.is_file():
                states["T2L"]["drops"]["paired_image_missing"] += 1
                continue
            for qa in parse_list(row.get("qa_list")):
                if not isinstance(qa, dict) or not normalize(qa.get("question")):
                    states["T2L"]["drops"]["paired_question_missing"] += 1
                    continue
                if normalize(qa["question"]) == normalize(record["question"]):
                    states["T2L"]["drops"]["question_role_mismatch"] += 1
                    continue
                if not normalize(qa.get("answer")):
                    states["T2L"]["drops"]["gold_missing"] += 1
                    continue
                key = (image_id, qa.get("qid", ""), normalize(qa["question"]), normalize(qa["answer"]))
                add_candidate(states["T2L"], key, {"metadata_row": row["_row"], "probe_image_id": image_id, "probe_qid": qa.get("qid", "")})

        rows = t3.get(image_id, [])
        for task in ("T3L", "T3G"):
            states[task]["metadata_rows"] = [row["_row"] for row in rows]
            if not rows:
                states[task]["drops"]["anchor_absent"] += 1
        for row in rows:
            diseases = {normalize(value) for value in parse_list(row.get("diseases")) if normalize(value)}
            for entry in parse_list(row.get("same_disease_images_in_other_modalities")):
                if not isinstance(entry, dict):
                    for task in ("T3L", "T3G"):
                        states[task]["drops"]["paired_image_missing"] += 1
                    continue
                paired = str(entry.get("image_id", ""))
                paired_path = image_path(paired, datasets, slake_root, vqarad_root)
                if paired_path is None or not paired_path.is_file():
                    for task in ("T3L", "T3G"):
                        states[task]["drops"]["paired_image_missing"] += 1
                    continue
                modality_a = normalize(row.get("modality_A"))
                modality_b = normalize(entry.get("modality"))
                if not modality_a or not modality_b or modality_a == modality_b:
                    for task in ("T3L", "T3G"):
                        states[task]["drops"]["unsupported_direction"] += 1
                    continue
                disease = normalize(entry.get("disease"))
                if not disease or (diseases and disease not in diseases):
                    for task in ("T3L", "T3G"):
                        states[task]["drops"]["concept_mismatch"] += 1
                    continue
                matches = [qa for qa in source_qas.get(paired, []) if normalize(qa.get("question")) == normalize(record["question"])]
                if not matches:
                    for task in ("T3L", "T3G"):
                        states[task]["drops"]["paired_question_missing"] += 1
                    continue
                for qa in matches:
                    if not normalize(qa.get("answer")):
                        for task in ("T3L", "T3G"):
                            states[task]["drops"]["gold_missing"] += 1
                        continue
                    key = (image_id, paired, qa.get("qid", ""), normalize(qa["question"]), disease)
                    relation = {"metadata_row": row["_row"], "probe_image_id": paired, "probe_qid": qa.get("qid", ""), "disease": entry.get("disease", ""), "probe_modality": entry.get("modality", "")}
                    add_candidate(states["T3L"], key, relation)
                    add_candidate(states["T3G"], key, relation)

        rows = t4l.get(image_id, [])
        states["T4L"]["metadata_rows"] = [row["_row"] for row in rows]
        if not rows:
            states["T4L"]["drops"]["anchor_absent"] += 1
        for row in rows:
            if normalize(row.get("question_a")) != normalize(record["question"]):
                states["T4L"]["drops"]["question_role_mismatch"] += 1
                continue
            if not normalize(row.get("lesion_a")) or not normalize(row.get("lesion_b")) or normalize(row["lesion_a"]) == normalize(row["lesion_b"]):
                states["T4L"]["drops"]["concept_mismatch"] += 1
                continue
            if not normalize(row.get("question_b")):
                states["T4L"]["drops"]["paired_question_missing"] += 1
                continue
            if not normalize(row.get("answer_b")):
                states["T4L"]["drops"]["gold_missing"] += 1
                continue
            key = (image_id, normalize(row["question_b"]), normalize(row["answer_b"]))
            add_candidate(states["T4L"], key, {"metadata_row": row["_row"], "probe_image_id": image_id})

        rows = t4g.get(image_id, [])
        states["T4G"]["metadata_rows"] = [row["_row"] for row in rows]
        if not rows:
            states["T4G"]["drops"]["anchor_absent"] += 1
        for row in rows:
            lesion = row.get("single_lesion", "")
            if not (phrase_in(lesion, record["question"]) or phrase_in(lesion, record["gold_answer"])):
                states["T4G"]["drops"]["concept_mismatch"] += 1
                continue
            paired = row.get("multi_image", "")
            paired_path = image_path(paired, datasets, slake_root, vqarad_root)
            if paired_path is None or not paired_path.is_file():
                states["T4G"]["drops"]["paired_image_missing"] += 1
                continue
            matches = [qa for qa in source_qas.get(paired, []) if phrase_in(lesion, qa.get("question"))]
            if not matches:
                states["T4G"]["drops"]["paired_question_missing"] += 1
                continue
            for qa in matches:
                if not normalize(qa.get("answer")):
                    states["T4G"]["drops"]["gold_missing"] += 1
                    continue
                key = (image_id, paired, qa.get("qid", ""), normalize(qa["question"]), normalize(qa["answer"]))
                add_candidate(states["T4G"], key, {"metadata_row": row["_row"], "probe_image_id": paired, "probe_qid": qa.get("qid", ""), "single_lesion": lesion})

        task_details = {}
        for task, state in states.items():
            if state["metadata_rows"]:
                summary[task]["retained_edits_with_metadata_anchor"] += 1
            if state["relations"]:
                summary[task]["candidate_edit_count"] += 1
            summary[task]["candidate_probe_relations"] += len(state["relations"])
            summary[task]["drop_reason_counts"].update(state["drops"])
            task_details[task] = {
                "metadata_rows": state["metadata_rows"],
                "candidate_probe_relations": state["relations"],
                "drop_reason_counts": {reason: state["drops"].get(reason, 0) for reason in DROP_REASONS},
            }
        details.append({
            "record_id": record["record_id"],
            "original_position": record["original_position"],
            "amended_position": record["amended_position"],
            "dataset": record["dataset"],
            "image_id": image_id,
            "tasks": task_details,
        })

    clean_summary = {}
    for task, values in summary.items():
        clean_summary[task] = {
            **{key: value for key, value in values.items() if key != "drop_reason_counts"},
            "drop_reason_counts": {reason: values["drop_reason_counts"].get(reason, 0) for reason in DROP_REASONS},
        }
    family_anchors = {
        "T2L": clean_summary["T2L"]["retained_edits_with_metadata_anchor"],
        "T3": clean_summary["T3L"]["retained_edits_with_metadata_anchor"],
        "T4": len({row["record_id"] for row in details if row["tasks"]["T4L"]["metadata_rows"] or row["tasks"]["T4G"]["metadata_rows"]}),
    }
    return details, {
        "schema_version": "m3bench-core9-anchor-overlap-v1",
        "created_at_utc": utc_now(),
        "record_count": len(records),
        "tasks": clean_summary,
        "family_anchor_counts": family_anchors,
        "status": "PASS" if all(family_anchors.values()) else "BLOCKED__NO_ANCHOR_INTERSECTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--slake-image-root", type=Path, required=True)
    parser.add_argument("--vqarad-image-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-records-sha256", required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.with_name(args.output_root.name + ".tmp").exists():
        raise RuntimeError("refusing to reuse Core-9 recovery root")
    if sha256(args.records) != args.expected_records_sha256:
        raise RuntimeError("amended-189 record hash mismatch")
    records = read_jsonl(args.records)
    if len(records) != 189 or [row["amended_position"] for row in records] != list(range(1, 190)):
        raise RuntimeError("amended-189 sequence is not exact")

    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    actual = {name: sha256(args.benchmark_root / name) for name in lock["files"]}
    if actual != lock["files"]:
        raise RuntimeError("public source snapshot differs from source lock")

    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        details, census = audit_records(
            records,
            args.benchmark_root / "metadata/selected_processed_files",
            args.slake_image_root,
            args.vqarad_image_root,
        )
        atomic_jsonl(temporary / "inputs/core9_recovery/ANCHOR_OVERLAP_189.jsonl", details)
        atomic_json(temporary / "reports/core9_recovery/ANCHOR_OVERLAP_CENSUS.json", census)
        atomic_json(temporary / "reports/core9_recovery/SPEC_AUTHORITY_DECISION.json", {
            "schema_version": "m3bench-core9-spec-authority-v1",
            "operational_scope": "M3BENCH_PUBLIC_RELEASE_ALIGNED_AMENDED189_CORE9",
            "data_construction": f"public release {lock['public_commit']} plus official metadata",
            "metric_primary": "macro per eligible edit request where unambiguous",
            "t5_status": "M3BENCH_T5_SEPARATE_EXTENSION_BLOCKED__PADCHEST_GR_ASSETS_UNAVAILABLE",
            "paper_exact_claim_permitted": False,
        })
        atomic_json(temporary / "governance/OPERATOR_DECISION_CORE9.json", {
            "decision": "APPROVE_M3BENCH_PUBLIC_RELEASE_ALIGNED_CORE9_WITH_T5_SEPARATE_EXTENSION",
            "parent_commit": args.parent_commit,
            "amended_record_count": 189,
            "amended_records_sha256": args.expected_records_sha256,
            "gpu_authorized": 3,
        })
        atomic_json(temporary / "locks/PUBLIC_SOURCE_LOCK.json", {**lock, "observed_sha256": actual})
        marker = "M3BENCH_CORE9_ANCHOR_OVERLAP_PASS" if census["status"] == "PASS" else "M3BENCH_CORE9_BLOCKED__NO_ANCHOR_INTERSECTION"
        atomic_text(temporary / marker, census["status"] + "\n")
        os.replace(temporary, args.output_root)
        print(json.dumps({"status": census["status"], "family_anchor_counts": census["family_anchor_counts"], "tasks": census["tasks"]}, sort_keys=True))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
