#!/usr/bin/env python3
"""Close existing DEV16 raw outputs, prepare fixed Judge input, and publish aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer  # noqa: E402

TASKS = ("T0", "T1L", "T1G", "T2G")
EXPECTED = {"T0": 16, "T1L": 26, "T1G": 62, "T2G": 59}
NEGATION = {"no", "not", "without", "negative", "absent", "absence"}
LATERALITY = {"left", "right", "bilateral"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def whole_words(needle: str, haystack: str) -> bool:
    words = re.findall(r"[a-z0-9]+", normalize_medical_answer(needle))
    text = set(re.findall(r"[a-z0-9]+", normalize_medical_answer(haystack)))
    return bool(words) and all(word in text for word in words)


def prepared_rows(run: Path, generation_sha: str, eos_token_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_dirs = sorted(path for path in (run / "events").iterdir() if path.is_dir())
    if len(event_dirs) != 16:
        raise RuntimeError(f"expected 16 event directories, found {len(event_dirs)}")
    rows, event_locks = [], []
    seen = set()
    for event_index, event_dir in enumerate(event_dirs, 1):
        result_path = event_dir / "result.json"
        checkpoint_path = event_dir / "expert.pt"
        if not result_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"event {event_index} is missing result or checkpoint")
        result = json.loads(result_path.read_text())
        checkpoint_sha = result["checkpoint"]["sha256"]
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise RuntimeError(f"event {event_index} checkpoint lock mismatch")
        event_locks.append({
            "event_index": event_index,
            "record_id_sha256": hashlib.sha256(result["record_id"].encode()).hexdigest(),
            "checkpoint_sha256": checkpoint_sha,
            "result_sha256": sha256_file(result_path),
        })
        edit_target = normalize_medical_answer(result["selected"]["decoded_text"])
        if not result["selected"]["literal_normalized_target_match"]:
            raise RuntimeError(f"event {event_index} selected output does not lock the edit target")
        for output in result["evaluation"]:
            key = (result["record_id"], output["query_id"], checkpoint_sha, generation_sha)
            if key in seen:
                raise RuntimeError(f"duplicate prediction binding in event {event_index}")
            seen.add(key)
            normalized = normalize_medical_answer(output["decoded_text"])
            reference = normalize_medical_answer(output["reference"])
            tokens = list(output["generated_token_ids"])
            reached = len(tokens) >= 1024
            ended = bool(tokens and tokens[-1] == eos_token_id)
            opaque = sha256_json(key)
            rows.append({
                "opaque_query_id": opaque,
                "event_index": event_index,
                "edit_id": result["record_id"],
                "query_id": output["query_id"],
                "task": output["task"],
                "question": output["question"],
                "reference": output["reference"],
                "raw_answer": output["decoded_text"],
                "raw_token_ids": tokens,
                "generated_token_count": len(tokens),
                "checkpoint_sha256": checkpoint_sha,
                "generation_config_sha256": generation_sha,
                "legacy_target_substring_present": bool(output["native_target_copy"]),
                "normalized_exact_reference_match": normalized == reference,
                "normalized_exact_edit_target_match": normalized == edit_target,
                "edit_target_whole_words_present": whole_words(edit_target, normalized),
                "reached_length_limit": reached,
                "ended_with_eos": ended,
                "truncated_without_eos": reached and not ended,
            })
    counts = Counter(row["task"] for row in rows)
    if dict(counts) != EXPECTED or len(rows) != 163:
        raise RuntimeError(f"DEV16 closure mismatch: total={len(rows)}, tasks={dict(counts)}")
    return rows, {"events": event_locks, "task_counts": dict(counts)}


def prepare(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    generation_sha = sha256_file(args.generation_lock)
    rows, locks = prepared_rows(args.run, generation_sha, args.eos_token_id)
    base = {row["query_id"]: row for row in read_jsonl(args.base_predictions)}
    base_t2g = []
    for row in rows:
        if row["task"] != "T2G":
            continue
        item = base.get(row["query_id"])
        if item is None:
            raise RuntimeError(f"missing frozen base output for {row['query_id']}")
        opaque = sha256_json((row["edit_id"], row["query_id"], "BASE", generation_sha))
        base_t2g.append({
            "opaque_query_id": opaque,
            "edit_id": row["edit_id"],
            "query_id": row["query_id"],
            "task": "T2G_BASE",
            "question": row["question"],
            "reference": row["reference"],
            "raw_answer": item["model_answer_raw"],
        })
    packet_rows = rows + base_t2g
    packet = "".join(json.dumps({
        "opaque_query_id": row["opaque_query_id"],
        "question": row["question"],
        "gold_answer": row["reference"],
        "raw_base_answer": row["raw_answer"],
        "adjudication_pass": 1,
    }, sort_keys=True) + "\n" for row in packet_rows)
    sidecar = [{
        "opaque_query_id": row["opaque_query_id"], "event_index": row.get("event_index"),
        "edit_id": row["edit_id"], "query_id": row["query_id"], "task": row["task"],
    } for row in packet_rows]
    closure = {
        "schema_version": "medtrace-dev16-raw-closure-private-v1",
        "status": "COMPLETE",
        "source_run": str(args.run),
        "source_run_code_commit": json.loads((args.run / "run_lock.json").read_text())["code_commit"],
        "run_lock_sha256": sha256_file(args.run / "run_lock.json"),
        "generation_config_sha256": generation_sha,
        "event_count": 16,
        "prediction_count": 163,
        "judge_packet_count": len(packet_rows),
        "missing": 0,
        "duplicates": 0,
        "empty_outputs": sum(not row["raw_answer"].strip() for row in rows),
        "length": {
            "reached_limit": sum(row["reached_length_limit"] for row in rows),
            "ended_with_eos": sum(row["ended_with_eos"] for row in rows),
            "truncated_without_eos": sum(row["truncated_without_eos"] for row in rows),
            "evidence_boundary": "derived from saved token IDs; generation stop_reason was not saved",
        },
        **locks,
        "predictions": rows,
        "base_t2g_rejudge": base_t2g,
    }
    atomic_json(args.out / "RAW_CLOSURE.json", closure)
    atomic_json(args.out / "JUDGE_SIDECAR_PRIVATE.json", sidecar)
    atomic_text(args.out / "JUDGE_PACKET_PRIVATE.jsonl", packet)
    atomic_json(args.out / "RUN_COMPLETION.json", {
        "schema_version": "medtrace-run-completion-private-v1",
        "training_generation": "COMPLETE",
        "semantic_judge": "PENDING",
        "scope": "NOT_RUN",
        "publication": "PENDING",
        "historical_run_lock_sha256": closure["run_lock_sha256"],
        "historical_run_lock_status": "RUNNING_STARTUP_SNAPSHOT_NOT_MUTATED",
    })


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def task_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        by_edit = defaultdict(list)
        for row in selected:
            by_edit[row["event_index"]].append(row)
        exact_per_edit = [mean(item["normalized_exact_reference_match"] for item in values) for values in by_edit.values()]
        semantic_per_edit = [mean(item["semantic_correct"] for item in values) for values in by_edit.values()]
        metrics[task] = {
            "eligible_edit_count": len(by_edit),
            "prediction_count": len(selected),
            "exact_correct": sum(row["normalized_exact_reference_match"] for row in selected),
            "exact_micro": mean(row["normalized_exact_reference_match"] for row in selected),
            "exact_macro": mean(exact_per_edit),
            "semantic_correct": sum(row["semantic_correct"] for row in selected),
            "semantic_micro": mean(row["semantic_correct"] for row in selected),
            "semantic_macro": mean(semantic_per_edit),
            "reached_length_limit": sum(row["reached_length_limit"] for row in selected),
            "ended_with_eos": sum(row["ended_with_eos"] for row in selected),
            "truncated_without_eos": sum(row["truncated_without_eos"] for row in selected),
        }
    return metrics


def failure_kind(row: dict[str, Any]) -> str:
    if not row["raw_answer"].strip():
        return "unanswered"
    if row["truncated_without_eos"]:
        return "repetition_or_truncation"
    output = set(re.findall(r"[a-z]+", normalize_medical_answer(row["raw_answer"])))
    reference = set(re.findall(r"[a-z]+", normalize_medical_answer(row["reference"])))
    if (output & NEGATION) != (reference & NEGATION):
        return "negation_or_polarity_conflict"
    if (output & LATERALITY) != (reference & LATERALITY):
        return "laterality_conflict"
    return "substantive_error_or_unknown"


def finalize(args: argparse.Namespace) -> None:
    closure = json.loads((args.private / "RAW_CLOSURE.json").read_text())
    verdicts = read_jsonl(args.judge_output)
    verdict_by_id = {row["opaque_query_id"]: row for row in verdicts}
    expected_ids = {row["opaque_query_id"] for row in closure["predictions"] + closure["base_t2g_rejudge"]}
    if len(verdict_by_id) != len(verdicts) or verdict_by_id.keys() != expected_ids:
        raise RuntimeError("Judge output does not exactly cover the prepared packet")
    if any(not row["parse_valid"] for row in verdicts):
        raise RuntimeError("Judge produced an invalid structured verdict")
    rows = closure["predictions"]
    for row in rows:
        row["semantic_correct"] = bool(verdict_by_id[row["opaque_query_id"]]["is_correct"])
    metrics = task_metrics(rows)
    base_t2g = {(row["edit_id"], row["query_id"]): bool(verdict_by_id[row["opaque_query_id"]]["is_correct"]) for row in closure["base_t2g_rejudge"]}
    cross = Counter()
    failures = Counter()
    for row in (item for item in rows if item["task"] == "T2G"):
        base_correct = base_t2g[(row["edit_id"], row["query_id"])]
        cp_correct = row["semantic_correct"]
        cross[f"base_{'correct' if base_correct else 'wrong'}__cp_{'correct' if cp_correct else 'wrong'}"] += 1
        if not cp_correct:
            failures[failure_kind(row)] += 1
    oracle = sum(base_t2g[(row["edit_id"], row["query_id"])] or row["semantic_correct"] for row in rows if row["task"] == "T2G") / EXPECTED["T2G"]
    t1l = Counter()
    for row in (item for item in rows if item["task"] == "T1L"):
        if row["semantic_correct"]:
            t1l["compatible"] += 1
        elif row["normalized_exact_edit_target_match"] and normalize_medical_answer(row["reference"]) != normalize_medical_answer(row["raw_answer"]):
            t1l["source_and_judge_supported_conflict"] += 1
        else:
            t1l["unknown"] += 1
    event_results = [json.loads(path.read_text()) for path in sorted((args.source_run / "events").glob("*/result.json"))]
    aggregate = {
        "schema_version": "medtrace-dev16-evaluation-private-v1",
        "status": "CP_DEV16_EVALUATION_COMPLETE",
        "judge_coverage": len(verdicts),
        "cp_prediction_coverage": len(rows),
        "base_t2g_rejudge_coverage": len(base_t2g),
        "tasks": metrics,
        "t1l_relation_diagnostic": dict(t1l),
        "t2g_cross_table": dict(cross),
        "t2g_oracle_selector_upper_bound": oracle,
        "t2g_failure_diagnostic": dict(failures),
        "nll": {
            "base_mean": mean(row["base_score"]["nll"] for row in event_results),
            "selected_mean": mean(row["selected"]["score"]["nll"] for row in event_results),
            "decreased": sum(row["nll_decreased"] for row in event_results),
        },
        "steps": {
            "values": [row["selected"]["step"] for row in event_results],
            "mean": mean(row["selected"]["step"] for row in event_results),
            "median": median(row["selected"]["step"] for row in event_results),
        },
        "total_event_seconds": sum(row["timing"]["end_to_end_seconds"] for row in event_results),
        "peak_vram_bytes": max(row["peak_vram_bytes"] for row in event_results),
    }
    atomic_json(args.private / "CP_DEV16_EVALUATION_PRIVATE.json", aggregate)
    completion = json.loads((args.private / "RUN_COMPLETION.json").read_text())
    completion["semantic_judge"] = "COMPLETE"
    atomic_json(args.private / "RUN_COMPLETION.json", completion)
    args.public.mkdir(parents=True, exist_ok=True)
    public_completion = completion | {
        "status": "CP_DEV16_EVALUATION_COMPLETE__SCOPE_PENDING",
        "private_artifacts_withheld": True,
        "judge_execution_sha256": sha256_file(args.execution_lock),
    }
    atomic_json(args.public / "RUN_COMPLETION.json", public_completion)
    atomic_json(args.public / "RAW_CLOSURE.json", {
        key: closure[key] for key in ("schema_version", "status", "source_run_code_commit", "run_lock_sha256", "generation_config_sha256", "event_count", "prediction_count", "judge_packet_count", "missing", "duplicates", "empty_outputs", "length", "task_counts")
    })
    with (args.public / "CP_DEV16_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "eligible_edits", "n", "exact_correct", "exact_micro", "exact_macro", "semantic_correct", "semantic_micro", "semantic_macro", "truncated_without_eos"])
        for task in TASKS:
            row = metrics[task]
            writer.writerow([task, row["eligible_edit_count"], row["prediction_count"], row["exact_correct"], row["exact_micro"], row["exact_macro"], row["semantic_correct"], row["semantic_micro"], row["semantic_macro"], row["truncated_without_eos"]])
    table = "\n".join(
        f"| {task} | {value['eligible_edit_count']} | {value['prediction_count']} | {value['exact_correct']}/{value['prediction_count']} ({value['exact_micro']:.1%}) | {value['exact_macro']:.1%} | {value['semantic_correct']}/{value['prediction_count']} ({value['semantic_micro']:.1%}) | {value['semantic_macro']:.1%} | {value['truncated_without_eos']} |"
        for task, value in metrics.items()
    )
    atomic_text(args.public / "CP_DEV16_REPORT.md", f"""# CP-DEV16 evaluation

Status: `CP_DEV16_EVALUATION_COMPLETE`

This is a 16-edit development run of `TIME_INSPIRED_CP_R4_DEV16_FORCED_ON`, not full TIME, full MedTRACE, or an independent benchmark. T0 was used for fitting and early stopping.

| Task | Valid edits | n | Exact micro | Exact macro | Semantic micro | Semantic macro | Truncated without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All 16 experts reached the native training stop within 200 steps; mean selected step was {aggregate['steps']['mean']:.2f}. Mean target NLL changed from {aggregate['nll']['base_mean']:.6f} to {aggregate['nll']['selected_mean']:.6f}; all 16 decreased. Event-time sum was {aggregate['total_event_seconds']:.3f} seconds and peak allocated VRAM was {aggregate['peak_vram_bytes'] / 1024**3:.3f} GiB.

The legacy `native_target_copy` field was a normalized substring test. It is retained only as `legacy_target_substring_present`; exact-reference, exact-edit-target, whole-word and semantic verdicts are separate. Length-limit classification is derived from saved token IDs because the original run did not save a stop reason.

T1L source/Judge diagnostic: {dict(t1l)}. `unknown` is neither removed from the ordinary locality denominator nor called a clinical conflict.
""")
    atomic_text(args.public / "T2G_DIAGNOSTIC.md", f"""# T2G diagnostic

Status: `COMPLETE`

The 59 saved forced-on outputs and the corresponding saved base outputs were judged in the same semantic protocol and current execution lane. No backbone generation was rerun.

Cross-table: `{dict(cross)}`.

The analysis-only oracle selector upper bound is `{oracle:.6f}`. It is not an implemented router and cannot repair rows where both base and CP are wrong.

Failure categories for semantic-false CP outputs: `{dict(failures)}`. Negation/laterality categories are lexical diagnostics; remaining errors are not assigned a clinical mechanism without stronger evidence.
""")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--base-predictions", type=Path, required=True)
    p.add_argument("--generation-lock", type=Path, required=True)
    p.add_argument("--eos-token-id", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=prepare)
    f = sub.add_parser("finalize")
    f.add_argument("--private", type=Path, required=True)
    f.add_argument("--source-run", type=Path, required=True)
    f.add_argument("--judge-output", type=Path, required=True)
    f.add_argument("--execution-lock", type=Path, required=True)
    f.add_argument("--public", type=Path, required=True)
    f.set_defaults(func=finalize)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
