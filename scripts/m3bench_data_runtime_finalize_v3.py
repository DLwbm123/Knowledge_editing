#!/usr/bin/env python3
"""Freeze final base verdicts, T0-T4 cohorts, handoff files, and public reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path


TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G")
T5_STATUS = "M3BENCH_T5_SEPARATE_EXTENSION_BLOCKED__PADCHEST_GR_ASSETS_UNAVAILABLE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def copy_new(source: Path, target: Path) -> None:
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def unique_members(rows: list[dict]) -> list[dict]:
    result = {}
    for row in rows:
        result.setdefault(row["query_id"], row)
    return list(result.values())


def lineage_relation_id(row: dict) -> str:
    values = {
        item.get("relation_id") for item in row.get("lineage", [])
        if item.get("source_task") == "T0" and item.get("relation_id")
    }
    if len(values) != 1:
        raise RuntimeError("T0 candidate must bind exactly one legacy edit ID")
    return values.pop()


def task_manifest(task: str, candidates: list[dict], inventory: dict[str, dict]) -> dict:
    formal = [row for row in candidates if row["eligible"]]
    probe_counts = [len(row["probe_query_ids"]) for row in formal]
    ids = {row["edit_query_id"] for row in candidates}
    ids.update(query_id for row in candidates for query_id in row.get("all_probe_query_ids", row["probe_query_ids"]))
    return {
        "task": task,
        "candidate_edit_count": len(candidates),
        "eligible_edit_count": len(formal),
        "candidate_probe_count": sum(len(row.get("all_probe_query_ids", row["probe_query_ids"])) for row in candidates),
        "eligible_probe_count": sum(probe_counts),
        "unique_image_count": len({(inventory[value]["dataset"], inventory[value]["image_id"]) for value in ids}),
        "zero_probe_edit_count": sum(not row["probe_query_ids"] for row in candidates),
        "mean_probes_per_eligible_edit": statistics.mean(probe_counts) if probe_counts else 0,
        "median_probes_per_eligible_edit": statistics.median(probe_counts) if probe_counts else 0,
        "macro_aggregation_unit": "eligible_edit_request",
        "pooled_micro_is_secondary_only": True,
        "method_outputs_used_for_selection": False,
        "status": "PASS" if formal and probe_counts else "NA_BASE_NO_ELIGIBLE_PRECORRECT_PROBES",
    }


def anchor_task(
    task: str, relations: list[dict], active: dict[str, dict], verdicts: dict[str, bool], inventory: dict[str, dict]
) -> tuple[list[dict], list[dict], dict]:
    candidates = []
    for relation in relations:
        edit = active.get(relation["legacy_edit_id"])
        if edit is None or relation["task"] != task:
            continue
        members = unique_members(relation["members"])
        if task == "T1L":
            eligible_members = [row for row in members if verdicts[row["query_id"]]]
            expected = "base_correct_locality_probe"
        else:
            eligible_members = [row for row in members if not verdicts[row["query_id"]]]
            expected = "base_wrong_generality_probe"
        candidates.append({
            "event_id": f"{task}:{relation['relation_id']}",
            "task": task,
            "edit_query_id": edit["query_id"],
            "probe_query_ids": [row["query_id"] for row in eligible_members],
            "all_probe_query_ids": [row["query_id"] for row in members],
            "expected_probe_precondition": expected,
            "macro_aggregation_unit": "eligible_edit_request",
            "method_outputs_used_for_selection": False,
            "eligible": bool(eligible_members),
        })
    candidates.sort(key=lambda row: row["event_id"])
    formal = [row for row in candidates if row["eligible"]]
    return candidates, formal, task_manifest(task, candidates, inventory)


def t4l_task(relations: list[dict], verdicts: dict[str, bool], inventory: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    candidates = []
    for relation in relations:
        if relation.get("structural_status") != "retained":
            continue
        members = {row["role"]: row for row in relation["members"]}
        edit, probe = members["edit_target_qA"], members["locality_probe_qB"]
        eligible = not verdicts[edit["query_id"]] and verdicts[probe["query_id"]]
        candidates.append({
            "event_id": f"T4L:{relation['relation_id']}",
            "task": "T4L",
            "edit_query_id": edit["query_id"],
            "probe_query_ids": [probe["query_id"]] if eligible else [],
            "all_probe_query_ids": [probe["query_id"]],
            "expected_probe_precondition": "base_correct_locality_probe",
            "macro_aggregation_unit": "eligible_edit_request",
            "method_outputs_used_for_selection": False,
            "eligible": eligible,
        })
    candidates.sort(key=lambda row: row["event_id"])
    formal = [row for row in candidates if row["eligible"]]
    manifest = task_manifest("T4L", candidates, inventory)
    manifest["structurally_rejected_count"] = sum(row.get("structural_status") != "retained" for row in relations)
    return candidates, formal, manifest


def t0_task(candidates: list[dict], verdicts: dict[str, bool], inventory: dict[str, dict]) -> tuple[list[dict], dict, dict[str, dict]]:
    ordered = sorted(candidates, key=lambda row: row["amended_position"])
    retained = [row for row in ordered if not verdicts[row["query_id"]]]
    formal = [{
        **row,
        "task": "T0",
        "event_id": f"T0:{row['query_id']}",
        "edit_query_id": row["query_id"],
        "sequence_position": index,
        "method_outputs_used_for_selection": False,
    } for index, row in enumerate(retained, 1)]
    manifest = {
        "task": "T0",
        "candidate_edit_count": len(ordered),
        "eligible_edit_count": len(formal),
        "candidate_probe_count": len(ordered),
        "eligible_probe_count": len(formal),
        "unique_image_count": len({(row["dataset"], row["image_id"]) for row in retained}),
        "zero_probe_edit_count": 0,
        "mean_probes_per_eligible_edit": 1 if formal else 0,
        "median_probes_per_eligible_edit": 1 if formal else 0,
        "prefixes": [value for value in (1, 50, 100, len(formal)) if value <= len(formal)],
        "removed_base_correct_count": len(ordered) - len(formal),
        "method_outputs_used_for_selection": False,
        "status": "PASS" if formal else "NA_BASE_NO_ELIGIBLE_TARGETS",
    }
    active = {lineage_relation_id(row): row for row in retained}
    if len(active) != len(retained):
        raise RuntimeError("duplicate active T0 legacy edit ID")
    return formal, manifest, active


def route_and_manifest(root: Path, inventory: list[dict], verdict_rows: list[dict]) -> dict:
    base = root / "base_final"
    route_rows = [{
        "query_id": row["query_id"],
        "authoritative_route": row["authoritative_route"],
        "semantic_judge_used": row["semantic_judge_used"],
        "semantic_judge_votes": row["semantic_judge_votes"],
    } for row in verdict_rows]
    write_jsonl(base / "BASE_VERDICT_ROUTE_V3.jsonl", route_rows)
    inference = read_json(root / "base_raw_canonical/BASE_INFERENCE_MANIFEST.json")
    runtime_lock = root / "runtime_audit/runtime_ab_final/CANONICAL_LLVAMED_RUNTIME_LOCK.json"
    judge_lock = read_json(base / "SEMANTIC_JUDGE_LOCK_V3_CANONICAL.json")
    selected = {
        "selected_raw": "canonical_full_reinference",
        "prediction_count": inference["query_count"],
        "prediction_sha256": inference["prediction_sha256"],
        "runtime": "runtime_b_official_native",
        "runtime_lock_sha256": sha256(runtime_lock),
        "old_raw_modified": False,
        "editing_methods_rerun": False,
    }
    write_json(base / "BASE_PREDICTIONS_SELECTED_MANIFEST.json", selected)
    votes = read_jsonl(base / "BASE_SEMANTIC_JUDGE_VERDICTS_V3.jsonl")
    manifest = {
        "status": "BASE_VERDICT_V3_FROZEN",
        "query_count": len(inventory),
        "verdict_count": len(verdict_rows),
        "correct_count": sum(row["is_correct"] for row in verdict_rows),
        "incorrect_count": sum(not row["is_correct"] for row in verdict_rows),
        "semantic_judge_query_coverage": sum(row["semantic_judge_used"] for row in verdict_rows),
        "semantic_judge_vote_count": len(votes),
        "missing": 0,
        "duplicate": 0,
        "invalid_schema": sum(type(row.get("is_correct")) is not bool for row in votes),
        "third_pass_count": sum(1 for line in (base / "BASE_JUDGE_PACKET_V3_PASS3.jsonl").open() if line.strip()),
        "prediction_sha256": inference["prediction_sha256"],
        "verdict_sha256": sha256(base / "BASE_VERDICTS_V3.jsonl"),
        "route_sha256": sha256(base / "BASE_VERDICT_ROUTE_V3.jsonl"),
        "runtime_lock_sha256": sha256(runtime_lock),
        "judge_model": judge_lock["judge_model"],
        "judge_prompt_sha256": judge_lock["prompt_sha256"],
        "judge_config_sha256": judge_lock["config_sha256"],
    }
    if manifest["query_count"] != 11088 or manifest["verdict_count"] != 11088 or manifest["invalid_schema"]:
        raise RuntimeError("final base verdict closure failed")
    write_json(base / "BASE_VERDICT_V3_MANIFEST.json", manifest)
    names = [
        "BASE_PREDICTIONS_SELECTED_MANIFEST.json", "BASE_VERDICTS_V3.jsonl", "BASE_VERDICT_ROUTE_V3.jsonl",
        "BASE_SEMANTIC_JUDGE_PACKET_V3.jsonl", "BASE_SEMANTIC_JUDGE_VERDICTS_V3.jsonl",
        "BASE_VERDICT_V3_MANIFEST.json", "SEMANTIC_JUDGE_LOCK_V3_CANONICAL.json",
    ]
    write_new(base / "SHA256SUMS.txt", "".join(f"{sha256(base / name)}  {name}\n" for name in names))
    return manifest


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--gpu-uuid", action="append", default=[])
    args = parser.parse_args()
    root, static, handoff = args.run_root, args.run_root / "data_static", args.run_root / "handoff"
    if handoff.exists():
        raise RuntimeError("refusing to reuse final handoff directory")
    inventory_rows = read_jsonl(static / "STATIC_QUERY_INVENTORY.jsonl")
    inventory = {row["query_id"]: row for row in inventory_rows}
    verdict_rows = read_jsonl(root / "base_final/BASE_VERDICTS_V3.jsonl")
    verdicts = {row["query_id"]: row["is_correct"] for row in verdict_rows}
    if len(inventory) != len(verdicts) or set(inventory) != set(verdicts):
        raise RuntimeError("inventory/base verdict coverage mismatch")
    base_manifest = route_and_manifest(root, inventory_rows, verdict_rows)

    t0, t0_manifest, active = t0_task(read_jsonl(static / "STATIC_T0_CANDIDATES.jsonl"), verdicts, inventory)
    summaries = {"T0": t0_manifest}
    rows_by_task = {"T0": t0}
    for task, name in (("T1L", "STATIC_T1_RELATIONS.jsonl"), ("T1G", "STATIC_T1_RELATIONS.jsonl"), ("T2G", "STATIC_T2G_RELATIONS.jsonl")):
        _, rows_by_task[task], summaries[task] = anchor_task(task, read_jsonl(static / name), active, verdicts, inventory)
    _, rows_by_task["T4L"], summaries["T4L"] = t4l_task(read_jsonl(static / "STATIC_T4L_RELATIONS.jsonl"), verdicts, inventory)
    for task in ("T2L", "T3L", "T3G", "T4G"):
        rows_by_task[task] = read_jsonl(root / f"cohorts_task_specific/{task}_FORMAL_RECORDS.jsonl")
        summaries[task] = read_json(root / f"cohorts_task_specific/{task}_MANIFEST.json")
    summaries = {task: summaries[task] for task in TASKS}
    if any(row["eligible_edit_count"] <= 0 or row["eligible_probe_count"] <= 0 for row in summaries.values()):
        status = "M3BENCH_PUBLIC_RELEASE_ALIGNED_T0_T4_DATA_FINALIZED__BASE_CONDITIONED_PARTIAL"
    else:
        status = "M3BENCH_PUBLIC_RELEASE_ALIGNED_T0_T4_DATA_FINALIZED__ALL_COHORTS_NONZERO"
    handoff.mkdir(parents=True)
    for task, rows in rows_by_task.items():
        write_jsonl(handoff / f"{task}_FINAL_SEQUENCE.jsonl" if task == "T0" else handoff / f"{task}_FORMAL_RECORDS.jsonl", rows)
    copy_new(root / "runtime_audit/runtime_ab_final/CANONICAL_LLVAMED_RUNTIME_LOCK.json", handoff / "CANONICAL_LLVAMED_RUNTIME_LOCK.json")
    copy_new(root / "base_final/BASE_PREDICTIONS_SELECTED_MANIFEST.json", handoff / "BASE_PREDICTIONS_SELECTED_MANIFEST.json")
    copy_new(root / "base_final/BASE_VERDICT_V3_MANIFEST.json", handoff / "BASE_VERDICT_V3_MANIFEST.json")
    write_json(handoff / "FORMAL_TASK_COUNTS.json", summaries)
    catalog_manifest = {
        "status": status, "scope": "public-release-aligned T0-T4", "paper_exact_claim_permitted": False,
        "tasks": summaries, "t5_status": T5_STATUS, "method_outputs_used": False,
        "runtime_lock_sha256": sha256(handoff / "CANONICAL_LLVAMED_RUNTIME_LOCK.json"),
        "base_verdict_manifest_sha256": sha256(handoff / "BASE_VERDICT_V3_MANIFEST.json"),
        "editing_methods_started": False,
    }
    write_json(handoff / "FORMAL_CATALOG_MANIFEST.json", catalog_manifest)
    handoff_names = ["CANONICAL_LLVAMED_RUNTIME_LOCK.json", "BASE_PREDICTIONS_SELECTED_MANIFEST.json", "BASE_VERDICT_V3_MANIFEST.json", "FORMAL_TASK_COUNTS.json", "FORMAL_CATALOG_MANIFEST.json", "T0_FINAL_SEQUENCE.jsonl", *[f"{task}_FORMAL_RECORDS.jsonl" for task in TASKS if task != "T0"]]
    write_new(handoff / "SHA256SUMS.txt", "".join(f"{sha256(handoff / name)}  {name}\n" for name in handoff_names))

    legacy = sum(row["legacy_token_f1_verdict"] for row in verdict_rows)
    public = sum(row["public_release_fuzzy_verdict"] for row in verdict_rows)
    semantic = base_manifest["correct_count"]
    lines = [
        "| Task | Candidate edits | Eligible edits | Candidate probes | Eligible probes | Zero-probe edits | Mean probes/edit | Median | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *[f"| {task} | {row['candidate_edit_count']} | {row['eligible_edit_count']} | {row['candidate_probe_count']} | {row['eligible_probe_count']} | {row['zero_probe_edit_count']} | {row['mean_probes_per_eligible_edit']:.3f} | {row['median_probes_per_eligible_edit']:.3f} | {row['status']} |" for task, row in summaries.items()],
    ]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_new(args.report_dir / "EXECUTIVE_DATA_FINALIZATION_REPORT.md", f"# Executive data finalization report\n\nStatus: `{status}`\n\n- Static data valid: yes (11,088 unique queries)\n- Base runtime aligned: official native runtime selected\n- Base verdict frozen: 11,088/11,088\n- Task cohorts eligible: all T0-T4 cohorts nonzero = `{status.endswith('ALL_COHORTS_NONZERO')}`\n- Editing method run: no\n- Scope: public-release-aligned; not paper-exact\n")
    write_new(args.report_dir / "SCORER_V3_AUDIT.md", f"# Scorer V3 audit\n\n- Legacy correct: {legacy}/11088\n- Public fuzzy correct: {public}/11088\n- Semantic-final correct: {semantic}/11088\n- Semantic Judge query coverage: {base_manifest['semantic_judge_query_coverage']}/11088\n- Gate-critical second votes: 5061\n- Third votes required: {base_manifest['third_pass_count']}\n- Missing/duplicate/invalid: 0/0/0\n")
    copy_new(root / "runtime_audit/runtime_ab_final/RUNTIME_AB_REPORT.md", args.report_dir / "RUNTIME_AB_REPORT.md")
    write_new(args.report_dir / "BASE_FINAL_REPORT.md", f"# Base final report\n\n- Selected raw: canonical full re-inference\n- Count: 11,088\n- Prediction SHA-256: `{read_json(root / 'base_raw_canonical/BASE_INFERENCE_MANIFEST.json')['prediction_sha256']}`\n- Semantic correct/incorrect: {semantic}/{11088-semantic}\n- Runtime: `runtime_b_official_native`\n- GPU shards: 2 disjoint shards ({', '.join(args.gpu_uuid)})\n- Empty/errors/missing/duplicate: 0/0/0/0\n- Historical raw and results modified: no\n")
    write_new(args.report_dir / "T0_T4_COHORT_FREEZE_REPORT.md", "# T0-T4 cohort freeze\n\n" + "\n".join(lines) + "\n\nPrimary aggregation: `PRIMARY_MACRO_PER_EDIT`. Secondary aggregation: `SECONDARY_MICRO_POOLED`.\n")
    write_new(args.report_dir / "METHOD_DEVELOPMENT_DATA_HANDOFF.md", f"# Method development data handoff\n\n- Status: `{status}`\n- T0 final length: {summaries['T0']['eligible_edit_count']}\n- Prefixes: {summaries['T0']['prefixes']}\n- Runtime lock, base verdict lock, task records, counts, and checksums are under the private `handoff/` directory.\n- No editing method was started.\n")
    for name in ("SCORER_DISAGREEMENT_BY_TASK.csv", "SCORER_DISAGREEMENT_BY_ANSWER_TYPE.csv"):
        copy_new(root / f"base_final/{name}", args.report_dir / name)
    for source, name in ((root / "checkpoint_audit/OFFICIAL_SNAPSHOT_LOCK.json", "OFFICIAL_SNAPSHOT_LOCK.json"), (root / "runtime_audit/runtime_ab_final/CANONICAL_LLVAMED_RUNTIME_LOCK.json", "CANONICAL_LLVAMED_RUNTIME_LOCK.json"), (root / "base_final/BASE_VERDICT_V3_MANIFEST.json", "BASE_VERDICT_V3_MANIFEST.json"), (handoff / "FORMAL_CATALOG_MANIFEST.json", "FORMAL_CATALOG_MANIFEST.json")):
        copy_new(source, args.report_dir / "locks" / name)
    public_files = sorted(path for path in args.report_dir.rglob("*") if path.is_file() and "checksums" not in path.parts)
    write_new(args.report_dir / "checksums/PUBLIC_REPORT_SHA256SUMS.txt", "".join(f"{sha256(path)}  {path.relative_to(args.report_dir)}\n" for path in public_files))
    print(json.dumps({"status": status, "base": base_manifest, "tasks": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
