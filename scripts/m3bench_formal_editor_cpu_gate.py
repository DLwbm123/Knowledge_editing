#!/usr/bin/env python3
"""Build a non-overwriting CPU-only bridge from the final data handoff to the editor runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G")
SEQUENTIAL_TASKS = ("T0", "T1L", "T1G", "T2G")
METHODS = ("lora", "grace", "balancedit", "belora")


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


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_new(path: Path, value: object, *, jsonl: bool = False) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value)
        if jsonl
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def copy_new(source: Path, target: Path) -> None:
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def verify_checksum_manifest(root: Path) -> bool:
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        relative = Path(name.strip())
        if relative.is_absolute() or ".." in relative.parts or sha256(root / relative) != expected:
            return False
    return True


def legacy_record(query: dict, legacy: dict[str, dict]) -> dict | None:
    identifiers = {
        value
        for item in query.get("lineage", [])
        for value in (item.get("relation_id"), item.get("legacy_source_record_id"))
        if value
    }
    matches = []
    for identifier in identifiers:
        candidate = legacy.get(identifier)
        if candidate and all(
            candidate.get(key) == query.get(key)
            for key in ("dataset", "question", "gold_answer", "image_sha256")
        ):
            matches.append(candidate)
    unique = {row["record_id"]: row for row in matches}
    if len(unique) > 1:
        raise RuntimeError(f"ambiguous legacy binding for {query['query_id']}")
    return next(iter(unique.values()), None)


def editor_record(query: dict, task: str, position: int, legacy: dict[str, dict]) -> dict:
    source = legacy_record(query, legacy)
    positive_source = "legacy_official_rephrase" if source else "identity_fallback_no_frozen_rephrase"
    return {
        "record_id": query["query_id"],
        "dataset": query["dataset"],
        "question": query["question"],
        "gold_answer": query["gold_answer"],
        "official_rephrase": source["official_rephrase"] if source else query["question"],
        "image_path": query["image_path"],
        "relative_image_path": source.get("relative_image_path", query.get("image_id", "")) if source else query.get("image_id", ""),
        "formal_sequence_position": position,
        "question_type": task,
        "router_positive_source": positive_source,
    }


def probe_record(query: dict, *, edit_id: str, task: str, position: int, index: int) -> dict:
    return {
        "edit_id": edit_id,
        "probe_id": query["query_id"],
        "task": task,
        "dataset": query["dataset"],
        "question": query["question"],
        "reference": query["gold_answer"],
        "image_path": query["image_path"],
        "sequence_position": position,
        "variant_type": query.get("role", "probe"),
        "probe_index": index,
    }


def expected_counts(events: list[dict], sequential: list[dict], prefixes: list[int]) -> dict:
    single_tasks = Counter()
    event_tasks = Counter()
    for event in events:
        event_tasks[event["task"]] += 1
        single_tasks[event["task"]] += len(event["probes"])
    prefix_rows = {}
    for prefix in prefixes:
        selected = [row for row in sequential if row["sequence_position"] <= prefix]
        counts = Counter(row["task"] for row in selected)
        prefix_rows[str(prefix)] = {
            "raw_outputs_per_task_per_method": dict(sorted(counts.items())),
            "raw_outputs_per_method": len(selected),
            "raw_outputs_all_methods": len(METHODS) * len(selected),
        }
    return {
        "schema_version": "m3bench-public-aligned-formal-expected-counts-v1",
        "methods": list(METHODS),
        "single": {
            "events_per_task_per_method": dict(sorted(event_tasks.items())),
            "events_per_method": len(events),
            "raw_outputs_per_task_per_method": dict(sorted(single_tasks.items())),
            "raw_outputs_per_method": sum(single_tasks.values()),
            "raw_outputs_all_methods": len(METHODS) * sum(single_tasks.values()),
        },
        "sequential": {
            "ordered_edit_task": "T0",
            "ordered_edit_count": max(prefixes),
            "supported_probe_tasks": list(SEQUENTIAL_TASKS),
            "unsupported_task_specific_tasks": [task for task in TASKS if task not in SEQUENTIAL_TASKS],
            "prefixes": prefix_rows,
        },
    }


def build(args: argparse.Namespace) -> dict:
    output = args.output_root
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError("refusing to reuse CPU-gate output root")

    handoff_manifest = read_json(args.handoff_root / "FORMAL_CATALOG_MANIFEST.json")
    task_counts = read_json(args.handoff_root / "FORMAL_TASK_COUNTS.json")
    canonical_runtime = read_json(args.handoff_root / "CANONICAL_LLVAMED_RUNTIME_LOCK.json")
    official_snapshot = read_json(args.official_snapshot_lock)
    method_bundle_path = args.effect_repair_root / "locks/EFFECT_REPAIRED_METHOD_CONFIG_BUNDLE.json"
    method_bundle = read_json(method_bundle_path)
    effect_model = read_json(args.effect_repair_root / "locks/EFFECT_REPAIRED_MODEL_RUNTIME_LOCK.json")
    effect_generation = read_json(args.effect_repair_root / "locks/EFFECT_REPAIRED_GENERATION_LOCK.json")
    effect_targets = read_json(args.effect_repair_root / "locks/EFFECT_REPAIRED_TARGET_MODULE_LOCK.json")
    inventory_rows = read_jsonl(args.query_inventory)
    inventory = {row["query_id"]: row for row in inventory_rows}
    legacy_rows = read_jsonl(args.legacy_run_root / "inputs/frozen/FORMAL_EDITOR_RECORDS_200.jsonl")
    legacy = {row["record_id"]: row for row in legacy_rows}
    t0 = read_jsonl(args.handoff_root / "T0_FINAL_SEQUENCE.jsonl")
    prefixes = task_counts["T0"]["prefixes"]

    checks = {
        "handoff_checksum_manifest_pass": verify_checksum_manifest(args.handoff_root),
        "handoff_scope_public_release_aligned": handoff_manifest.get("scope") == "public-release-aligned T0-T4",
        "paper_exact_claim_disabled": handoff_manifest.get("paper_exact_claim_permitted") is False,
        "editing_not_previously_started": handoff_manifest.get("editing_methods_started") is False,
        "inventory_unique": len(inventory) == len(inventory_rows) == 11088,
        "legacy_formal_200_exact": len(legacy) == len(legacy_rows) == 200,
        "t0_count_exact": len(t0) == task_counts["T0"]["eligible_edit_count"],
        "t0_positions_dense": [row["sequence_position"] for row in t0] == list(range(1, len(t0) + 1)),
        "t0_prefixes_exact": prefixes == [value for value in (1, 50, 100, len(t0)) if value <= len(t0)],
        "method_outputs_not_used": handoff_manifest.get("method_outputs_used") is False,
        "canonical_runtime_official_native": canonical_runtime.get("selected_runtime") == "runtime_b_official_native",
        "snapshot_locks_agree": all(
            canonical_runtime.get(key) == official_snapshot.get(key)
            for key in ("llava_med_code_commit", "model_snapshot_sha", "vision_snapshot_sha")
        ) and official_snapshot.get("local_checkpoint_matches_official_snapshot") is True,
        "effect_runtime_matches_canonical_model": (
            effect_model.get("model_id") == "llava_med_v1_5_mistral_7b"
            and effect_model.get("language_block_count") == 32
            and effect_model.get("model_class") in official_snapshot.get("architecture", [])
        ),
        "effect_generation_matches_canonical": all(
            effect_generation.get(key) == value
            or (key == "temperature" and effect_generation.get(key) is None and value == 0)
            for key, value in canonical_runtime.get("generation", {}).items()
        ),
        "effect_targets_locked": bool(effect_targets.get("target_lists_sha256")),
        "effect_method_bundle_exact": set(method_bundle.get("method_configs", {})) == set(METHODS)
        and all(
            config.get("config_sha256")
            == canonical_sha256({key: value for key, value in config.items() if key != "config_sha256"})
            for config in method_bundle.get("method_configs", {}).values()
        ),
    }

    task_rows: dict[str, list[dict]] = {}
    referenced_ids = {row["query_id"] for row in t0}
    event_ids = {row["event_id"] for row in t0}
    task_checks = {}
    for task in TASKS[1:]:
        rows = read_jsonl(args.handoff_root / f"{task}_FORMAL_RECORDS.jsonl")
        task_rows[task] = rows
        ids = [row["event_id"] for row in rows]
        referenced_ids.update(row["edit_query_id"] for row in rows)
        referenced_ids.update(value for row in rows for value in row["probe_query_ids"])
        task_checks[task] = (
            len(rows) == task_counts[task]["eligible_edit_count"]
            and sum(len(row["probe_query_ids"]) for row in rows) == task_counts[task]["eligible_probe_count"]
            and len(ids) == len(set(ids))
            and all(row.get("eligible") is True for row in rows)
            and all(row.get("method_outputs_used_for_selection") is False for row in rows)
        )
        event_ids.update(ids)
    checks["task_files_match_frozen_counts"] = all(task_checks.values())
    checks["event_ids_unique"] = len(event_ids) == len(t0) + sum(len(rows) for rows in task_rows.values())
    checks["all_referenced_queries_resolve"] = referenced_ids <= inventory.keys()

    bound = [legacy_record(row, legacy) for row in t0]
    checks["t0_legacy_binding_exact"] = all(bound) and len({row["record_id"] for row in bound if row}) == len(t0)
    image_rows = [inventory[query_id] for query_id in sorted(referenced_ids) if query_id in inventory]
    checks["all_referenced_images_readable"] = all(Path(row["image_path"]).is_file() for row in image_rows)
    checks["all_referenced_image_hashes_exact"] = args.skip_image_hashes or all(
        sha256(Path(row["image_path"])) == row["image_sha256"] for row in image_rows
    )
    if not all(checks.values()):
        raise RuntimeError(f"CPU gate failed: {checks}")

    t0_records = [editor_record(row, "T0", row["sequence_position"], legacy) for row in t0]
    t0_positions = {row["query_id"]: row["sequence_position"] for row in t0}
    events = []
    sequential = []
    for row, record in zip(t0, t0_records, strict=True):
        probe = probe_record(row, edit_id=row["query_id"], task="T0", position=row["sequence_position"], index=1)
        events.append({
            "schema_version": "m3bench-formal-single-event-v1", "event_id": row["event_id"],
            "task": "T0", "event_position": row["sequence_position"], "edit_record": record, "probes": [probe],
        })
        sequential.append(probe)

    for task in TASKS[1:]:
        for event_position, row in enumerate(task_rows[task], 1):
            edit_query = inventory[row["edit_query_id"]]
            probes = [
                probe_record(inventory[query_id], edit_id=row["edit_query_id"], task=task, position=event_position, index=index)
                for index, query_id in enumerate(row["probe_query_ids"], 1)
            ]
            events.append({
                "schema_version": "m3bench-formal-single-event-v1", "event_id": row["event_id"],
                "task": task, "event_position": event_position,
                "edit_record": editor_record(edit_query, task, event_position, legacy), "probes": probes,
            })
            if task in SEQUENTIAL_TASKS:
                position = t0_positions[row["edit_query_id"]]
                sequential.extend({**probe, "sequence_position": position, "edit_id": row["edit_query_id"]} for probe in probes)

    sequential.sort(key=lambda row: (row["sequence_position"], TASKS.index(row["task"]), row["probe_index"]))
    counts = expected_counts(events, sequential, prefixes)
    positive_sources = Counter(event["edit_record"]["router_positive_source"] for event in events)
    preflight = {
        "schema_version": "m3bench-formal-editor-integration-cpu-gate-v1",
        "status": "M3BENCH_FORMAL_EDITOR_INTEGRATION_CPU_GATE_PASS__GPU_APPROVAL_REQUIRED",
        "checks": checks,
        "task_checks": task_checks,
        "t0_sequence_length": len(t0),
        "prefixes": prefixes,
        "single_event_count_per_method": len(events),
        "single_raw_output_count_per_method": counts["single"]["raw_outputs_per_method"],
        "router_positive_sources": dict(sorted(positive_sources.items())),
        "sequential_scope": "T0 ordered trajectory with T0/T1L/T1G/T2G probes",
        "task_specific_sequential_status": "NA_NO_FROZEN_CROSS_TASK_ORDER",
        "gpu_used": False,
        "model_loaded": False,
        "judge_started": False,
        "official_source_checkout_commit_verification": "DEFERRED_TO_GPU_APPROVAL_GATE",
    }
    judge_packet = {
        "schema_version": "m3bench-post-edit-judge-audit-packet-v1",
        "status": "READY_FOR_RAW_INGEST__NOT_EXECUTED",
        "independent_from_editor_runtime": True,
        "item_fields_allowed": ["opaque_event_id", "question", "gold_answer", "raw_model_answer", "task"],
        "item_fields_forbidden": ["method", "anonymous_group", "reviewer_a_verdict", "formal_position_anomaly", "other_method_answer", "expected_result"],
        "raw_answers_present": False,
        "verdicts_present": False,
        "judge_started": False,
        "expected_counts": counts,
    }
    gpu_plan = {
        "schema_version": "m3bench-formal-gpu-execution-plan-v1",
        "status": "GPU_APPROVAL_REQUIRED",
        "allowed_physical_gpus": [2, 3],
        "forbidden_physical_gpus": [1],
        "qualification_first": [
            "verify exact GPU UUID and one-visible-device guard",
            "run one T0 event and one identity-positive task-specific event per method in fresh non-formal outputs",
            "require nonempty generation, finite loss/gradients, base unchanged, save/reload parity, and route contract",
        ],
        "formal_after_qualification": [
            "run all single events in frozen task/file order using non-overwriting chunks",
            "run one T0 sequential trajectory per method at frozen prefixes",
            "do not invent sequential results for task-specific cohorts without a frozen order",
            "close raw artifacts before any Judge or evaluator",
        ],
        "judge_after_raw_closure_only": True,
    }

    temporary.mkdir(parents=True)
    try:
        inputs = temporary / "inputs/frozen"
        write_new(inputs / f"FORMAL_EDITOR_RECORDS_{len(t0_records)}.jsonl", t0_records, jsonl=True)
        write_new(inputs / "FORMAL_PROBE_CATALOG.jsonl", sequential, jsonl=True)
        write_new(inputs / "FORMAL_SINGLE_EVENT_CATALOG.jsonl", events, jsonl=True)
        copy_new(args.effect_repair_root / "runtime/LLAVA_MED_MODULE_INVENTORY.json", inputs / "LLAVA_MED_MODULE_INVENTORY.json")
        copy_new(args.effect_repair_root / "locks/EFFECT_REPAIRED_TARGET_MODULE_LOCK.json", inputs / "LLAVA_MED_EDIT_TARGET_LOCK.json")
        copy_new(args.effect_repair_root / "locks/EFFECT_REPAIRED_GENERATION_LOCK.json", inputs / "llava_med_generation_frozen.json")
        locks = temporary / "locks"
        copy_new(args.legacy_run_root / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json", locks / "FORMAL_MODEL_AND_GENERATION_LOCK.json")
        copy_new(method_bundle_path, locks / "FORMAL_METHOD_CONFIG_BUNDLE.json")
        for name in (
            "EFFECT_REPAIRED_MODEL_RUNTIME_LOCK.json",
            "EFFECT_REPAIRED_GENERATION_LOCK.json",
            "EFFECT_REPAIRED_TARGET_MODULE_LOCK.json",
            "EFFECT_REPAIRED_SOURCE_MANIFEST.json",
        ):
            copy_new(args.effect_repair_root / "locks" / name, locks / name)
        copy_new(args.handoff_root / "CANONICAL_LLVAMED_RUNTIME_LOCK.json", locks / "CANONICAL_LLVAMED_RUNTIME_LOCK.json")
        copy_new(args.official_snapshot_lock, locks / "OFFICIAL_SNAPSHOT_LOCK.json")
        copy_new(args.handoff_root / "FORMAL_CATALOG_MANIFEST.json", locks / "FORMAL_CATALOG_MANIFEST.json")
        copy_new(args.handoff_root / "FORMAL_TASK_COUNTS.json", locks / "FORMAL_TASK_COUNTS.json")
        write_new(locks / "FORMAL_EXPECTED_COUNTS.json", counts)
        write_new(temporary / "CPU_PREFLIGHT.json", preflight)
        write_new(temporary / "judge_audit/JUDGE_AUDIT_PACKET.json", judge_packet)
        write_new(temporary / "GPU_EXECUTION_PLAN.json", gpu_plan)
        (temporary / "M3BENCH_FORMAL_EDITOR_PREFLIGHT_PASS").write_text("PASS\n", encoding="utf-8")
        (temporary / "GPU_APPROVAL_REQUIRED").write_text("STOP\n", encoding="utf-8")
        for path in temporary.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--query-inventory", type=Path, required=True)
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--effect-repair-root", type=Path, required=True)
    parser.add_argument("--official-snapshot-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-image-hashes", action="store_true")
    result = build(parser.parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
