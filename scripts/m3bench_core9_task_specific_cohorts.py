#!/usr/bin/env python3
"""Freeze T2L/T3L/T3G/T4G cohorts from public metadata and base verdicts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.m3bench_core9_task_specific_t4l import canonical_hash, normalize, phrase_in, sha256


TASKS = ("T2L", "T3L", "T3G", "T4G")


def parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            pass
    return []


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    write_new(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def relation_id(task: str, edit_id: str) -> str:
    return task.casefold() + "-" + canonical_hash((task, edit_id))[:24]


def qa_index(inventory: list[dict]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in inventory:
        if any(lineage.get("source_task") == "PUBLIC_SOURCE_QA" for lineage in row["lineage"]):
            result[(row["dataset"], row["image_id"])].append(row)
    return result


def synthetic_index(inventory: list[dict]) -> dict[tuple[int, str], list[dict]]:
    result: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in inventory:
        for lineage in row["lineage"]:
            if lineage.get("source_task") == "T2L":
                result[(int(lineage["source_metadata_row"]), row["image_id"])].append(row)
    return result


def dataset_for(image_id: str) -> str:
    return "VQA-RAD" if image_id.endswith(".jpg") else "SLAKE"


def make_record(task: str, edit: dict, probes: list[dict], metadata_rows: list[int]) -> dict:
    return {
        "event_id": relation_id(task, edit["query_id"]),
        "task": task,
        "edit_query_id": edit["query_id"],
        "probe_query_ids": sorted({row["query_id"] for row in probes}),
        "source_metadata_rows": sorted(set(metadata_rows)),
        "expected_probe_pre_correct": task in {"T2L", "T3L"},
        "macro_aggregation_unit": "eligible_edit_request",
        "method_outputs_used_for_selection": False,
    }


def build_t2l(metadata_root: Path, by_image: dict, synthetic: dict, verdict: dict) -> tuple[list[dict], list[dict]]:
    candidates, audit = [], []
    grouped: dict[str, dict] = {}
    for row_number, relation in enumerate(read_csv(metadata_root / "t2l_cross_image_pairs.csv"), 1):
        for image_id in (relation["image_id_1"], relation["image_id_2"]):
            dataset = dataset_for(image_id)
            probes = [row for row in synthetic.get((row_number, image_id), []) if verdict[row["query_id"]]]
            for edit in by_image.get((dataset, image_id), []):
                if verdict[edit["query_id"]]:
                    continue
                eligible_probes = [row for row in probes if normalize(row["question"]) != normalize(edit["question"])]
                state = grouped.setdefault(edit["query_id"], {"edit": edit, "probes": {}, "rows": []})
                state["rows"].append(row_number)
                state["probes"].update({row["query_id"]: row for row in eligible_probes})
                audit.append({"task": "T2L", "edit_query_id": edit["query_id"], "source_metadata_row": row_number, "candidate_probe_count": len(probes), "eligible_probe_count": len(eligible_probes)})
    for state in grouped.values():
        probes = list(state["probes"].values())
        candidates.append({**make_record("T2L", state["edit"], probes, state["rows"]), "eligible": bool(probes)})
    return sorted(candidates, key=lambda row: row["event_id"]), audit


def build_t3(metadata_root: Path, by_image: dict, verdict: dict) -> tuple[dict[str, list[dict]], list[dict]]:
    grouped = {"T3L": {}, "T3G": {}}
    audit = []
    for row_number, relation in enumerate(read_csv(metadata_root / "t3_cross_modality_pairs.csv"), 1):
        anchor_id = relation["image_A"]
        anchor_dataset = dataset_for(anchor_id)
        anchor_modality = normalize(relation.get("modality_A"))
        diseases = {normalize(value) for value in parse_list(relation.get("diseases")) if normalize(value)}
        for edit in by_image.get((anchor_dataset, anchor_id), []):
            if verdict[edit["query_id"]] or not any(phrase_in(disease, edit["question"]) or phrase_in(disease, edit["gold_answer"]) for disease in diseases):
                continue
            for pair in parse_list(relation.get("same_disease_images_in_other_modalities")):
                if not isinstance(pair, dict):
                    continue
                disease = normalize(pair.get("disease"))
                if disease not in diseases or normalize(pair.get("modality")) == anchor_modality:
                    continue
                pair_id = str(pair.get("image_id", ""))
                matches = [row for row in by_image.get((dataset_for(pair_id), pair_id), []) if normalize(row["question"]) == normalize(edit["question"])]
                for probe in matches:
                    task = "T3L" if verdict[probe["query_id"]] else "T3G"
                    state = grouped[task].setdefault(edit["query_id"], {"edit": edit, "probes": {}, "rows": []})
                    state["rows"].append(row_number)
                    state["probes"][probe["query_id"]] = probe
                    audit.append({"task": task, "edit_query_id": edit["query_id"], "probe_query_id": probe["query_id"], "source_metadata_row": row_number, "paired_gold_source": "paired_image_source_qa", "probe_pre_correct": verdict[probe["query_id"]]})
    result = {}
    for task in ("T3L", "T3G"):
        result[task] = sorted(
            ({**make_record(task, state["edit"], list(state["probes"].values()), state["rows"]), "eligible": bool(state["probes"])} for state in grouped[task].values()),
            key=lambda row: row["event_id"],
        )
    if {row["probe_query_id"] for row in audit if row["task"] == "T3L"} & {row["probe_query_id"] for row in audit if row["task"] == "T3G"}:
        raise RuntimeError("T3 probe belongs to both locality and generality")
    return result, audit


def build_t4g(metadata_root: Path, by_image: dict, verdict: dict) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, dict] = {}
    audit = []
    for row_number, relation in enumerate(read_csv(metadata_root / "t4g_compositional_generality.csv"), 1):
        single_id, multi_id = relation["single_image"], relation["multi_image"]
        lesion = relation["single_lesion"]
        edits = [
            row for row in by_image.get((dataset_for(single_id), single_id), [])
            if not verdict[row["query_id"]] and (phrase_in(lesion, row["question"]) or phrase_in(lesion, row["gold_answer"]))
        ]
        probes = [
            row for row in by_image.get((dataset_for(multi_id), multi_id), [])
            if not verdict[row["query_id"]] and phrase_in(lesion, row["question"])
        ]
        for edit in edits:
            state = grouped.setdefault(edit["query_id"], {"edit": edit, "probes": {}, "rows": []})
            state["rows"].append(row_number)
            state["probes"].update({row["query_id"]: row for row in probes})
            audit.append({"task": "T4G", "edit_query_id": edit["query_id"], "source_metadata_row": row_number, "candidate_probe_count": len(probes), "eligible_probe_count": len(probes), "target_finding": lesion})
    candidates = [
        {**make_record("T4G", state["edit"], list(state["probes"].values()), state["rows"]), "eligible": bool(state["probes"])}
        for state in grouped.values()
    ]
    return sorted(candidates, key=lambda row: row["event_id"]), audit


def macro_per_edit(values: dict[str, list[bool]]) -> float | None:
    per_edit = [sum(items) / len(items) for items in values.values() if items]
    return sum(per_edit) / len(per_edit) if per_edit else None


def manifest(task: str, candidates: list[dict], inventory: dict[str, dict]) -> dict:
    formal = [row for row in candidates if row["eligible"]]
    probe_counts = [len(row["probe_query_ids"]) for row in formal]
    all_query_ids = {row["edit_query_id"] for row in candidates} | {value for row in candidates for value in row["probe_query_ids"]}
    return {
        "schema_version": "m3bench-task-specific-cohort-v1",
        "task": task,
        "candidate_edit_count": len(candidates),
        "eligible_edit_count": len(formal),
        "candidate_probe_count": sum(len(row["probe_query_ids"]) for row in candidates),
        "eligible_probe_count": sum(probe_counts),
        "unique_image_count": len({(inventory[value]["dataset"], inventory[value]["image_id"]) for value in all_query_ids}),
        "zero_probe_edit_count": sum(not row["probe_query_ids"] for row in candidates),
        "mean_probes_per_eligible_edit": statistics.mean(probe_counts) if probe_counts else 0,
        "median_probes_per_eligible_edit": statistics.median(probe_counts) if probe_counts else 0,
        "macro_aggregation_unit": "eligible_edit_request",
        "pooled_micro_is_secondary_only": True,
        "method_outputs_used_for_selection": False,
        "status": "PASS" if formal and probe_counts else "BLOCKED__ZERO_ELIGIBLE_COHORT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--base-verdicts", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse cohort output directory")
    inventory_rows = read_jsonl(args.inventory)
    inventory = {row["query_id"]: row for row in inventory_rows}
    verdict_rows = read_jsonl(args.base_verdicts)
    verdict = {row["query_id"]: bool(row["is_correct"]) for row in verdict_rows}
    if len(verdict) != len(verdict_rows) or set(verdict) != set(inventory):
        raise RuntimeError("base verdict coverage mismatch")
    by_image = qa_index(inventory_rows)
    t2l, t2_audit = build_t2l(args.metadata_root, by_image, synthetic_index(inventory_rows), verdict)
    t3, t3_audit = build_t3(args.metadata_root, by_image, verdict)
    t4g, t4_audit = build_t4g(args.metadata_root, by_image, verdict)
    task_rows = {"T2L": t2l, "T3L": t3["T3L"], "T3G": t3["T3G"], "T4G": t4g}
    audits = {"T2L": t2_audit, "T3L": [row for row in t3_audit if row["task"] == "T3L"], "T3G": [row for row in t3_audit if row["task"] == "T3G"], "T4G": t4_audit}
    args.output_dir.mkdir(parents=True)
    summary = {}
    event_ids = []
    for task, candidates in task_rows.items():
        formal = [row for row in candidates if row["eligible"]]
        write_jsonl(args.output_dir / f"{task}_CANDIDATES.jsonl", candidates)
        write_jsonl(args.output_dir / f"{task}_FORMAL_RECORDS.jsonl", formal)
        write_jsonl(args.output_dir / f"{task}_BASE_ELIGIBILITY_AUDIT.jsonl", audits[task])
        value = manifest(task, candidates, inventory)
        value["sequence_or_cohort_sha256"] = hashlib.sha256("".join(row["event_id"] for row in formal).encode()).hexdigest()
        write_json(args.output_dir / f"{task}_MANIFEST.json", value)
        summary[task] = value
        event_ids.extend(row["event_id"] for row in formal)
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("duplicate task event keys")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
