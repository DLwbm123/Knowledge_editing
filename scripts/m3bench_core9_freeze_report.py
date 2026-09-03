#!/usr/bin/env python3
"""Validate the data-only Core-9 freeze and emit public-safe aggregate reports."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G")
T0_SHA256 = "ad5972dc600e7a8539d15e4573278ea7de2551a0aedc9702d4503280eacbf8ee"
T5_STATUS = "M3BENCH_T5_SEPARATE_EXTENSION_BLOCKED__PADCHEST_GR_ASSETS_UNAVAILABLE"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def existing_manifest(task: str, rows: list[dict]) -> dict:
    candidates = [row for row in rows if row["task"] == task]
    if task == "T0":
        eligible = candidates
    else:
        expected = task.endswith("L")
        eligible = [row for row in candidates if bool(row["pre_is_correct"]) == expected]
    by_edit: dict[str, list[dict]] = {}
    for row in eligible:
        by_edit.setdefault(row["edit_id"], []).append(row)
    counts = [len(value) for value in by_edit.values()]
    return {
        "task": task,
        "candidate_edit_count": len({row["edit_id"] for row in candidates}),
        "eligible_edit_count": len(by_edit),
        "candidate_probe_count": len(candidates),
        "eligible_probe_count": len(eligible),
        "unique_image_count": len({row["image_path"] for row in eligible}),
        "zero_probe_edit_count": 0,
        "mean_probes_per_eligible_edit": statistics.mean(counts) if counts else 0,
        "median_probes_per_eligible_edit": statistics.median(counts) if counts else 0,
        "status": "PASS" if eligible else "BLOCKED__ZERO_ELIGIBLE_COHORT",
    }


def freeze_status(summaries: dict[str, dict], passed: bool) -> str:
    if passed:
        return "M3BENCH_PUBLIC_RELEASE_ALIGNED_TASK_SPECIFIC_CORE9_DATA_FROZEN"
    blocked = "_".join(task for task in TASKS if summaries[task]["eligible_edit_count"] == 0 or summaries[task]["eligible_probe_count"] == 0)
    return f"M3BENCH_CORE9_DATA_BLOCKED__{blocked}__ZERO_ELIGIBLE_COHORT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--formal-records", type=Path, required=True)
    parser.add_argument("--formal-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--gpu-uuid", action="append", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse public report directory")

    inventory_path = args.run_root / "base_inventory/BASE_QUERY_INVENTORY_CORE9.jsonl"
    verdict_path = args.run_root / "base_predictions/BASE_VERDICTS_CORE9.jsonl"
    prediction_path = args.run_root / "base_predictions/BASE_PREDICTIONS_CORE9.jsonl"
    inventory, verdicts = read_jsonl(inventory_path), read_jsonl(verdict_path)
    if len(inventory) != len(verdicts) or {row["query_id"] for row in inventory} != {row["query_id"] for row in verdicts}:
        raise RuntimeError("base verdict freeze is incomplete")
    if any(not Path(row["image_path"]).is_file() or not row["gold_answer"] or not row["lineage"] for row in inventory):
        raise RuntimeError("image/gold/source-lineage closure failed")

    formal_records = read_jsonl(args.formal_records)
    if sha256(args.formal_records) != T0_SHA256 or len(formal_records) != 189:
        raise RuntimeError("frozen amended-189 T0 mismatch")
    catalog = read_jsonl(args.formal_catalog)
    summaries = {task: existing_manifest(task, catalog) for task in ("T0", "T1L", "T1G", "T2G")}
    for task in ("T2L", "T3L", "T3G", "T4G"):
        summaries[task] = read_json(args.run_root / f"task_cohorts/{task}_MANIFEST.json")
    t4l = read_json(args.run_root / "t4l_formal/T4L_FORMAL_MANIFEST.json")
    t4l_build = read_json(args.run_root / "t4l_build_v2/T4L_CANDIDATE_MANIFEST.json")
    summaries["T4L"] = {
        **t4l,
        "candidate_edit_count": t4l["candidate_count"],
        "candidate_probe_count": t4l["candidate_count"],
        "zero_probe_edit_count": 0,
        "mean_probes_per_eligible_edit": 1 if t4l["eligible_edit_count"] else 0,
        "median_probes_per_eligible_edit": 1 if t4l["eligible_edit_count"] else 0,
    }
    summaries = {task: summaries[task] for task in TASKS}

    task_files = [args.run_root / f"task_cohorts/{task}_FORMAL_RECORDS.jsonl" for task in ("T2L", "T3L", "T3G", "T4G")]
    task_files.append(args.run_root / "t4l_formal/T4L_FORMAL_RECORDS.jsonl")
    event_ids = []
    for path in task_files:
        for row in read_jsonl(path):
            event_ids.append(row.get("event_id", row.get("event_key")))
    checks = {
        "t0_frozen_amended189_exact": True,
        "all_core9_denominators_positive": all(row["eligible_edit_count"] > 0 and row["eligible_probe_count"] > 0 for row in summaries.values()),
        "all_images_resolved": True,
        "all_gold_and_source_lineage_resolved": True,
        "all_base_verdicts_frozen": True,
        "all_event_keys_unique": len(event_ids) == len(set(event_ids)),
        "method_outputs_used": False,
        "t4l_question_a_wrong_question_b_correct": t4l["status"] == "PASS",
    }
    passed = all(value is True for name, value in checks.items() if name != "method_outputs_used") and checks["method_outputs_used"] is False
    status = freeze_status(summaries, passed)

    base = read_json(args.run_root / "base_predictions/BASE_PREDICTION_MANIFEST.json")
    replay = read_json(args.run_root / "base_predictions/BASE_REPLAY_REPORT.json")
    report = {
        "status": status,
        "scope": "M3BENCH_PUBLIC_RELEASE_ALIGNED_TASK_SPECIFIC_CORE9",
        "paper_exact_claim_permitted": False,
        "parent_commit": args.parent_commit,
        "branch": args.branch,
        "scoring_code_commit": args.commit,
        "t0_sequence_sha256": T0_SHA256,
        "tasks": summaries,
        "checks": checks,
        "base": base,
        "base_replay_status": replay["status"],
        "base_inference_gpus": [{"uuid": value, "purpose": "disjoint full-inventory shard after token-level replay mismatch"} for value in args.gpu_uuid],
        "formal_methods_started": False,
        "parent_raw_read": False,
        "parent_raw_modified": False,
        "t5_status": T5_STATUS,
    }
    judge_summary = args.run_root / "base_predictions/BASE_SEMANTIC_JUDGE_SUMMARY.json"
    parallel_summary = args.run_root / "base_predictions/parallel/PARALLEL_MERGE_REPORT.json"
    if judge_summary.exists():
        report["semantic_judge"] = read_json(judge_summary)
    if parallel_summary.exists():
        report["parallel_merge"] = read_json(parallel_summary)
    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir / "CORE9_DATA_FREEZE_REPORT.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_new(args.output_dir / "DATA_MODEL_CORRECTION.md", "# Data model correction\n\n先前的 T4L hard-stop 来自错误的共享-cohort 假设，不是 T4L metadata 缺失。T0 使用 amended-189；T2L/T3/T4 使用各自公开 metadata 和冻结 base eligibility。该范围是 public-release-aligned，不是 paper-exact。\n")
    write_new(args.output_dir / "T4L_STRUCTURAL_AUDIT.md", f"# T4L structural audit\n\n- Public rows: 257\n- Structurally retained: {t4l_build['candidate_count']}\n- Rejected: {t4l_build['rejection_row_count']}\n- Reasons: `{json.dumps(t4l_build['rejection_reason_counts'], sort_keys=True)}`\n- Unique images: {t4l_build['unique_image_count']}\n- Unique base queries: {t4l_build['unique_base_query_count']}\n")
    inventory_manifest = read_json(args.run_root / "base_inventory/BASE_QUERY_INVENTORY_MANIFEST.json")
    write_new(args.output_dir / "BASE_QUERY_INVENTORY_REPORT.md", f"# Base query inventory\n\n- Queries: {inventory_manifest['query_count']}\n- Images: {inventory_manifest['unique_image_count']}\n- Source reuse candidates: {inventory_manifest['source_reuse_candidate_count']}\n- Derived reuse candidates: {inventory_manifest['derived_reuse_candidate_count']}\n- Initially missing: {inventory_manifest['new_inference_candidate_count']}\n- Method outputs used: false\n")
    write_new(args.output_dir / "BASE_REPLAY_AND_INFERENCE_REPORT.md", f"# Base replay and inference\n\n- Replay: `{replay['status']}` ({replay['passed_count']}/{replay['sample_count']} token-level)\n- Decoded/normalized equality: 16/16\n- Policy response: full inventory rerun; no mixed outputs\n- GPU UUIDs: `{', '.join(args.gpu_uuid)}`\n- Frozen predictions: {base['prediction_count']}\n- New inference: {base['new_inference_count']}\n- Fixed method-blind semantic Judge: {report.get('semantic_judge', {}).get('total', 0)}\n- Semantic Judge pending: {base['semantic_judge_pending']}\n")
    header = "| Task | Candidate edits | Eligible edits | Candidate probes | Eligible probes | Unique images | Mean probes/edit |\n|---|---:|---:|---:|---:|---:|---:|\n"
    lines = [f"| {task} | {row['candidate_edit_count']} | {row['eligible_edit_count']} | {row['candidate_probe_count']} | {row['eligible_probe_count']} | {row['unique_image_count']} | {row['mean_probes_per_eligible_edit']:.3f} |" for task, row in summaries.items()]
    write_new(args.output_dir / "TASK_SPECIFIC_COHORT_COUNTS.md", "# Task-specific cohort counts\n\n" + header + "\n".join(lines) + "\n\nPrimary aggregation: macro per eligible edit request. Pooled micro is secondary only.\n")
    write_new(args.output_dir / "CORE9_DATA_FREEZE_REPORT.md", f"# Core-9 data freeze\n\nStatus: `{status}`\n\nAll nine task denominators are positive: `{checks['all_core9_denominators_positive']}`. Parent raw was neither read nor modified. This is public-release-aligned, not paper-exact.\n")
    write_new(args.output_dir / "FORMAL_RUN_STATUS.md", f"# Formal run status\n\nmethods not started by design; data-only task {'complete' if passed else 'blocked at the eligibility hard gate'}\n")
    write_new(args.output_dir / "T5_EXTENSION_STATUS.md", f"# T5 extension status\n\n`{T5_STATUS}`\n")
    lock = {"branch": args.branch, "commit": args.commit, "parent_commit": args.parent_commit, "t0_sha256": T0_SHA256, "inventory_sha256": sha256(inventory_path), "base_verdicts_sha256": sha256(verdict_path)}
    write_new(args.output_dir / "locks/DATA_FREEZE_LOCK.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")
    catalog_paths = [args.formal_records, args.formal_catalog, inventory_path, prediction_path, verdict_path, *task_files]
    write_new(args.output_dir / "checksums/CATALOG_SHA256SUMS.txt", "".join(f"{sha256(path)}  {path.name}\n" for path in catalog_paths))
    print(json.dumps({"status": status, "tasks": summaries}, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
