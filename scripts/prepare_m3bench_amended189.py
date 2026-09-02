#!/usr/bin/env python3
"""Freeze the approved exclusion-only sequence and run its static preflight."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G", "T5")
SEQUENCE_LABEL = "M3BENCH_AMENDED_EXCLUSION_ONLY_189"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def distribution(rows: list[dict]) -> dict:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(str(row.get(key, "unknown")) for row in rows).items()))

    return {
        "dataset": counts("dataset"),
        "question_type": counts("question_type"),
        "target_whitespace_token_length": dict(
            sorted(collections.Counter(len(str(row["gold_answer"]).split()) for row in rows).items())
        ),
        "token_length_definition": "Unicode-whitespace token count; model-token count is checked at runtime",
    }


def expected_counts(catalog: list[dict], count: int) -> dict:
    per_task = {task: sum(row["task"] == task for row in catalog) for task in TASKS}
    prefixes = {}
    for prefix in (1, 50, 100, count):
        selected = [row for row in catalog if int(row["sequence_position"]) <= prefix]
        task_counts = {task: sum(row["task"] == task for row in selected) for task in TASKS}
        prefixes[str(prefix)] = {
            "raw_outputs_per_task_per_method": task_counts,
            "raw_outputs_per_method": len(selected),
            "raw_outputs_all_methods": 4 * len(selected),
        }
    return {
        "schema_version": "m3bench-formal-editor-expected-counts-v2",
        "sequence_label": SEQUENCE_LABEL,
        "formal_records_per_method": count,
        "methods": ["LoRA", "GRACE", "BalanceEdit", "BELoRA-50-independent-reimplementation"],
        "single": {
            "records_per_method": count,
            "raw_outputs_per_task_per_method": per_task,
            "raw_outputs_per_method": len(catalog),
            "raw_outputs_all_methods": 4 * len(catalog),
        },
        "sequential": {"trajectories": 4, "edits_per_trajectory": count, "prefixes": prefixes},
    }


def coverage(catalog: list[dict], metadata_root: Path, padchest_root: Path) -> dict:
    metadata = {
        "T2L": metadata_root / "t2l_cross_image_pairs.csv",
        "T3L": metadata_root / "t3_cross_modality_pairs.csv",
        "T3G": metadata_root / "t3_cross_modality_pairs.csv",
        "T4L": metadata_root / "t4l_compositional_locality.csv",
        "T4G": metadata_root / "t4g_compositional_generality.csv",
    }
    rows = {}
    for task in TASKS:
        members = [row for row in catalog if row["task"] == task]
        if task == "T0":
            eligible = len(members)
        elif task.endswith("L"):
            eligible = sum(row.get("pre_is_correct") is True for row in members)
        elif task.endswith("G"):
            eligible = sum(row.get("pre_is_correct") is False for row in members)
        else:
            eligible = len(members)
        datasets = collections.Counter(row.get("dataset", "unknown") for row in members)
        if members:
            source_status = "AVAILABLE_FROM_FROZEN_PARENT"
        elif task in metadata:
            source_status = (
                "METADATA_PRESENT_BUT_FROZEN_PARENT_SCOPE_RESOLVED_ZERO"
                if metadata[task].is_file()
                else "SOURCE_METADATA_MISSING"
            )
        elif task == "T5":
            source_status = (
                "PADCHEST_ROOT_PRESENT_BUT_NO_FROZEN_TASK_CATALOG"
                if padchest_root.is_dir()
                else "PADCHEST_SOURCE_MISSING"
            )
        else:
            source_status = "FROZEN_PARENT_TASK_EMPTY"
        rows[task] = {
            "candidate_count": len(members),
            "eligible_denominator": eligible,
            "dataset_breakdown": dict(sorted(datasets.items())),
            "source_asset_status": source_status,
        }
    zero = [task for task, row in rows.items() if row["eligible_denominator"] == 0]
    return {
        "schema_version": "m3bench-amended189-data-task-coverage-v1",
        "sequence_label": SEQUENCE_LABEL,
        "tasks": rows,
        "zero_denominator_tasks": zero,
        "status": "PASS" if not zero else "BLOCKED__DATA_OR_TASK_COVERAGE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--review-reconstruction-root", type=Path, required=True)
    parser.add_argument("--single-reviewer-amendment-root", type=Path, required=True)
    parser.add_argument("--effect-repair-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--padchest-root", type=Path, required=True)
    parser.add_argument("--output-run-root", type=Path, required=True)
    parser.add_argument("--authority-source-branch", required=True)
    parser.add_argument("--authority-source-commit", required=True)
    parser.add_argument("--approved-at-utc", default=utc_now())
    args = parser.parse_args()

    output = args.output_run_root
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError("refusing to reuse amended run root")
    temporary.mkdir(parents=True)

    try:
        source_records_path = args.source_run_root / "inputs/frozen/FORMAL_EDITOR_RECORDS_200.jsonl"
        source_catalog_path = args.source_run_root / "inputs/frozen/FORMAL_PROBE_CATALOG.jsonl"
        source_records = read_jsonl(source_records_path)
        source_catalog = read_jsonl(source_catalog_path)
        if len(source_records) != 200 or [row["formal_sequence_position"] for row in source_records] != list(range(1, 201)):
            raise RuntimeError("source formal-200 sequence is not exact")

        census = read_json(args.single_reviewer_amendment_root / "SINGLE_REVIEWER_FINAL_TARGET_CENSUS.json")
        nonretained = read_jsonl(args.single_reviewer_amendment_root / "NONRETAINED_TARGETS_PRIVATE.jsonl")
        counts = census["reviewer_a_counts"]
        if counts != {"VALID": 189, "CONFIRMED_INVALID": 10, "UNRESOLVED": 1} or len(nonretained) != 11:
            raise RuntimeError("Reviewer A frozen census mismatch")

        pointer_path = args.review_reconstruction_root / "sessions/Reviewer_A/ACTIVE_REVIEWER_A_OUTPUT.json"
        pointer = read_json(pointer_path)
        reviewer_output = args.review_reconstruction_root / pointer["output"]
        reviewer_freeze = args.review_reconstruction_root / pointer["freeze_manifest"]
        reviewer_rows = read_jsonl(reviewer_output)
        if pointer.get("status") != "FROZEN" or pointer.get("records") != 200 or len(reviewer_rows) != 200:
            raise RuntimeError("Reviewer A output is not frozen at 200 records")
        if collections.Counter(row["valid"] for row in reviewer_rows) != {True: 189, False: 11}:
            raise RuntimeError("Reviewer A output count mismatch")

        excluded_positions = {int(row["formal_position"]) for row in nonretained}
        if len(excluded_positions) != 11 or 19 not in excluded_positions or 57 in excluded_positions or 67 in excluded_positions:
            raise RuntimeError("frozen amendment position contract mismatch")
        by_position = {int(row["formal_sequence_position"]): row for row in source_records}
        if any(by_position[int(row["formal_position"])]["record_id"] != row["formal_target_id"] for row in nonretained):
            raise RuntimeError("private nonretained mapping differs from formal-200")

        retained_source = [row for row in source_records if int(row["formal_sequence_position"]) not in excluded_positions]
        records = []
        position_map = []
        original_to_amended = {}
        for amended_position, source in enumerate(retained_source, 1):
            original_position = int(source["formal_sequence_position"])
            original_to_amended[original_position] = amended_position
            row = dict(source)
            row.update(
                {
                    "original_position": original_position,
                    "amended_position": amended_position,
                    "formal_sequence_position": amended_position,
                    "sequence_label": SEQUENCE_LABEL,
                    "sequence_size": 189,
                }
            )
            records.append(row)
            position_map.append(
                {
                    "amended_position": amended_position,
                    "original_position": original_position,
                    "record_id": row["record_id"],
                    "reviewer_a_verdict": "VALID",
                    "retained": True,
                }
            )
        if len(records) != 189 or len({row["record_id"] for row in records}) != 189:
            raise RuntimeError("amended sequence cardinality mismatch")

        image_failures = []
        for row in records:
            path = Path(row["image_path"])
            if not path.is_file() or sha256(path) != row["image_sha256"]:
                image_failures.append(row["amended_position"])
        if image_failures:
            raise RuntimeError(f"image closure failed at amended positions {image_failures}")

        retained_ids = {row["record_id"] for row in records}
        catalog = []
        for source in source_catalog:
            if source["edit_id"] not in retained_ids:
                continue
            row = dict(source)
            row["original_position"] = int(source["sequence_position"])
            row["sequence_position"] = original_to_amended[row["original_position"]]
            row["sequence_label"] = SEQUENCE_LABEL
            catalog.append(row)
        catalog.sort(key=lambda row: (row["sequence_position"], row["task"], row["probe_index"]))

        governance = temporary / "governance/amended189"
        write_jsonl(governance / "FORMAL_EDITOR_RECORDS_189.jsonl", records)
        write_jsonl(governance / "FORMAL_POSITION_MAP_189.jsonl", position_map)
        write_json(
            governance / "EXCLUDED_TARGETS_11.json",
            {"count": 11, "items": sorted(nonretained, key=lambda row: int(row["formal_position"]))},
        )
        source_task_lock = read_json(args.source_run_root / "locks/FORMAL_TASK_MANIFEST_LOCK.json")
        decision = {
            "schema_version": "m3bench-operator-amendment-decision-v1",
            "decision": "APPROVE_EXCLUSION_ONLY_189",
            "sequence_label": SEQUENCE_LABEL,
            "authority_source_branch": args.authority_source_branch,
            "authority_source_commit": args.authority_source_commit,
            "original_sequence_sha256": source_task_lock["formal_sequence_sha256"],
            "original_count": 200,
            "retained_count": 189,
            "excluded_confirmed_invalid": 10,
            "excluded_unresolved": 1,
            "replacement_count": 0,
            "ordering": "retained_original_relative_order",
            "method_output_used_for_selection": False,
            "approved_at_utc": args.approved_at_utc,
            "reviewer_a_binding": {
                "active_pointer_sha256": sha256(pointer_path),
                "output_sha256": sha256(reviewer_output),
                "freeze_manifest_sha256": sha256(reviewer_freeze),
            },
        }
        write_json(governance / "OPERATOR_AMENDMENT_DECISION.json", decision)
        audit = {
            "schema_version": "m3bench-amended189-selection-audit-v1",
            "created_at_utc": utc_now(),
            "status": "PASS",
            "checks": {
                "formal_200_exact": True,
                "retained_189_exact": True,
                "amended_positions_dense": True,
                "relative_order_preserved": True,
                "record_ids_unique": True,
                "images_resolved_and_sha256_exact": True,
                "method_output_used_for_selection": False,
            },
            "original_distribution": distribution(source_records),
            "amended_distribution": distribution(records),
        }
        write_json(governance / "AMENDED_189_SELECTION_AUDIT.json", audit)

        inputs = temporary / "inputs/frozen"
        for name in (
            "LLAVA_MED_EDIT_TARGET_LOCK.json",
            "LLAVA_MED_MODULE_INVENTORY.json",
            "canonical_535.jsonl",
            "gpt56_sol_rubric.md",
            "llava_med_generation_frozen.json",
            "t1g_derived_probe_manifest_800.jsonl",
            "t2g_derived_probe_manifest_800.jsonl",
        ):
            inputs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source_run_root / "inputs/frozen" / name, inputs / name)
        write_jsonl(inputs / "FORMAL_EDITOR_RECORDS_189.jsonl", records)
        write_jsonl(inputs / "FORMAL_PROBE_CATALOG.jsonl", catalog)
        for task in TASKS:
            source = args.source_run_root / f"inputs/frozen/tasks/{task}_task_manifest.jsonl"
            rows = read_jsonl(source) if source.is_file() else []
            filtered = []
            for row in rows:
                edit_id = row.get("anchor", {}).get("record_id_or_derived_probe_id")
                if edit_id in retained_ids:
                    item = dict(row)
                    item["original_position"] = int(row["sequence_position"])
                    item["sequence_position"] = original_to_amended[item["original_position"]]
                    filtered.append(item)
            write_jsonl(inputs / f"tasks/{task}_task_manifest.jsonl", filtered)

        locks = temporary / "locks"
        locks.mkdir(parents=True)
        shutil.copyfile(
            args.source_run_root / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json",
            locks / "FORMAL_MODEL_AND_GENERATION_LOCK.json",
        )
        shutil.copyfile(
            args.effect_repair_root / "locks/EFFECT_REPAIRED_METHOD_CONFIG_BUNDLE.json",
            locks / "FORMAL_METHOD_CONFIG_BUNDLE.json",
        )
        counts_payload = expected_counts(catalog, 189)
        write_json(locks / "FORMAL_EXPECTED_COUNTS_189.json", counts_payload)
        write_json(locks / "FORMAL_EXPECTED_COUNTS.json", counts_payload)

        coverage_report = coverage(catalog, args.metadata_root, args.padchest_root)
        write_json(temporary / "preflight/FORMAL_EXPECTED_COUNTS_189.json", counts_payload)
        write_json(temporary / "preflight/DATA_TASK_COVERAGE_REPORT.json", coverage_report)
        write_text(governance / "M3BENCH_AMENDMENT_EXCLUSION_ONLY_189_FROZEN", "PASS\n")

        checksum_files = [
            governance / "OPERATOR_AMENDMENT_DECISION.json",
            governance / "FORMAL_EDITOR_RECORDS_189.jsonl",
            governance / "FORMAL_POSITION_MAP_189.jsonl",
            governance / "EXCLUDED_TARGETS_11.json",
            governance / "AMENDED_189_SELECTION_AUDIT.json",
        ]
        write_text(
            governance / "SHA256SUMS.txt",
            "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files),
        )
        write_json(
            temporary / "RUN_MANIFEST.json",
            {
                "schema_version": "m3bench-amended189-run-v1",
                "created_at_utc": utc_now(),
                "sequence_label": SEQUENCE_LABEL,
                "record_count": 189,
                "amendment_status": "M3BENCH_AMENDMENT_EXCLUSION_ONLY_189_FROZEN",
                "data_task_coverage_status": coverage_report["status"],
                "formal_gpu_started": False,
                "judge_started": False,
                "evaluator_started": False,
            },
        )
        for path in temporary.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        os.replace(temporary, output)
        print(json.dumps({"status": coverage_report["status"], "run_root": str(output), "records": 189}))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
