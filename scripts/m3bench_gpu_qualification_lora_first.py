#!/usr/bin/env python3
"""Freeze GPU qualification inputs and run the no-edit parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.m3bench_base_correctness_v3 import normalize


ALL_CANARY_TASKS = ("T0", "T1L", "T3L", "T3G", "T4L")
HASH64_CANARY_TASKS = ("T1G", "T2L", "T2G", "T4G")


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


def stable_key(value: str) -> tuple[str, str]:
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), value


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


def event_probe_occurrences(events: list[dict], task: str) -> list[dict]:
    result = []
    for event in events:
        if event["task"] != task:
            continue
        for index, probe in enumerate(event["probes"], 1):
            result.append({
                "query_id": probe["probe_id"],
                "task": task,
                "event_id": event["event_id"],
                "probe_index": index,
            })
    return result


def no_edit_selection(events: list[dict]) -> tuple[list[dict], dict]:
    selected, counts = [], {}
    for task in ALL_CANARY_TASKS:
        rows = event_probe_occurrences(events, task)
        selected.extend(rows)
        counts[task] = {"requested": "all", "occurrences": len(rows)}
    for task in HASH64_CANARY_TASKS:
        rows = sorted(
            event_probe_occurrences(events, task),
            key=lambda row: (*stable_key(row["query_id"]), row["event_id"], row["probe_index"]),
        )[:64]
        selected.extend(rows)
        counts[task] = {"requested": 64, "occurrences": len(rows)}
    by_id: dict[str, dict] = {}
    for row in selected:
        item = by_id.setdefault(row["query_id"], {"query_id": row["query_id"], "source_tasks": []})
        if row["task"] not in item["source_tasks"]:
            item["source_tasks"].append(row["task"])
    rows = sorted(by_id.values(), key=lambda row: stable_key(row["query_id"]))
    return rows, {"source_selection": counts, "deduplicated_query_count": len(rows)}


def qualification_selection(events: list[dict]) -> tuple[list[dict], dict]:
    t0 = [row for row in events if row["task"] == "T0" and row["edit_record"].get("router_positive_source") == "legacy_official_rephrase"]
    selected = sorted(t0, key=lambda row: stable_key(row["event_id"]))[:4]
    groups = {
        "T2L": lambda row: row["task"] == "T2L",
        "T3": lambda row: row["task"] in {"T3L", "T3G"},
        "T4L": lambda row: row["task"] == "T4L",
        "T4G": lambda row: row["task"] == "T4G",
    }
    for label, predicate in groups.items():
        candidates = [
            row for row in events
            if predicate(row) and row["edit_record"].get("router_positive_source") == "identity_fallback_no_frozen_rephrase"
        ]
        if not candidates:
            raise RuntimeError(f"no identity-fallback qualification candidate for {label}")
        selected.append(sorted(candidates, key=lambda row: stable_key(row["event_id"]))[0])
    if len(selected) != 8 or len({row["event_id"] for row in selected}) != 8:
        raise RuntimeError("QUAL8 selection is not eight unique events")
    fallback_counts = Counter(
        row["task"] for row in events
        if row["edit_record"].get("router_positive_source") == "identity_fallback_no_frozen_rephrase"
    )
    return selected, {"identity_fallback_event_count": sum(fallback_counts.values()), "by_task": dict(sorted(fallback_counts.items()))}


def calibration_selection(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    t0 = {row["edit_record"]["record_id"]: row for row in events if row["task"] == "T0"}
    t1l_ids = sorted(
        {row["edit_record"]["record_id"] for row in events if row["task"] == "T1L"},
        key=stable_key,
    )
    if len(t1l_ids) != 14:
        raise RuntimeError("expected exactly 14 T1L edit anchors")
    other_ids = sorted(set(t0) - set(t1l_ids), key=stable_key)[:18]
    if len(other_ids) != 18:
        raise RuntimeError("insufficient distinct non-T1L T0 anchors")
    dev_ids = set(t1l_ids[:7] + other_ids[:9])
    qual_ids = set(t1l_ids[7:] + other_ids[9:])
    if dev_ids & qual_ids or len(dev_ids) != 16 or len(qual_ids) != 16:
        raise RuntimeError("DEV16/QUAL16 overlap or size failure")

    related: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        if row["task"] in {"T1L", "T1G", "T2G"}:
            related[row["edit_record"]["record_id"]].append(row)

    def build(ids: set[str]) -> list[dict]:
        result = []
        for record_id in sorted(ids, key=lambda value: int(t0[value]["event_position"])):
            base = t0[record_id]
            probes = [probe for row in related[record_id] for probe in row["probes"]]
            result.append({
                "event_id": base["event_id"],
                "event_position": base["event_position"],
                "edit_record": base["edit_record"],
                "probes": probes,
                "probe_tasks": dict(sorted(Counter(probe["task"] for probe in probes).items())),
            })
        return result

    remaining = sorted(set(t0) - dev_ids - qual_ids, key=stable_key)[:16]
    sequence = [t0[value] for value in sorted(remaining, key=lambda value: int(t0[value]["event_position"]))]
    return build(dev_ids), build(qual_ids), sequence


def prepare(args: argparse.Namespace) -> None:
    events = read_jsonl(args.cpu_gate / "inputs/frozen/FORMAL_SINGLE_EVENT_CATALOG.jsonl")
    inventory_rows = read_jsonl(args.data_root / "data_static/STATIC_QUERY_INVENTORY.jsonl")
    base_rows = read_jsonl(args.data_root / "base_raw_canonical/BASE_PREDICTIONS_CANONICAL.jsonl")
    verdict_rows = read_jsonl(args.data_root / "base_final/BASE_VERDICTS_V3.jsonl")
    inventory = {row["query_id"]: row for row in inventory_rows}
    base = {row["query_id"]: row for row in base_rows}
    verdict = {row["query_id"]: row for row in verdict_rows}
    if not (len(inventory) == len(inventory_rows) == len(base) == len(base_rows) == len(verdict) == len(verdict_rows) == 11088):
        raise RuntimeError("frozen query/base/verdict inventory coverage mismatch")

    selected, selection_report = no_edit_selection(events)
    private = []
    for item in selected:
        query_id = item["query_id"]
        source, prediction, decision = inventory[query_id], base[query_id], verdict[query_id]
        private.append({
            **item,
            "dataset": source["dataset"],
            "question": source["question"],
            "gold_answer": source["gold_answer"],
            "image_path": source["image_path"],
            "image_sha256": source["image_sha256"],
            "frozen_base_decoded_text": prediction["model_answer_raw"],
            "frozen_base_normalized": prediction["normalized_answer"],
            "frozen_semantic_verdict": decision["is_correct"],
        })
    private_path = args.output_root / "private/NO_EDIT_PARITY_INPUTS.jsonl"
    write_new(private_path, private, jsonl=True)
    manifest = {
        "schema_version": "m3bench-no-edit-parity-manifest-v1",
        "status": "FROZEN_BEFORE_GPU_OUTPUT",
        **selection_report,
        "query_ids": [row["query_id"] for row in selected],
        "private_inputs_sha256": sha256(private_path),
        "frozen_base_token_ids_available": False,
        "token_id_evidence_policy": "fresh exact-official runtime is the token-ID oracle; frozen canonical raw anchors decoded and normalized text",
    }
    write_new(args.output_root / "NO_EDIT_PARITY_MANIFEST.json", manifest)

    qual8, fallback = qualification_selection(events)
    qual_private = args.output_root / "private/QUAL8_INPUTS.jsonl"
    write_new(qual_private, qual8, jsonl=True)
    write_new(args.output_root / "QUAL8_MANIFEST.json", {
        "schema_version": "m3bench-qual8-manifest-v1",
        "status": "FROZEN_BEFORE_METHOD_OUTPUT",
        "event_count": 8,
        "event_ids": [row["event_id"] for row in qual8],
        "router_positive_sources": dict(sorted(Counter(row["edit_record"]["router_positive_source"] for row in qual8).items())),
        "datasets": dict(sorted(Counter(row["edit_record"]["dataset"] for row in qual8).items())),
        "identity_fallback_inventory": fallback,
        "private_inputs_sha256": sha256(qual_private),
        "method_outputs_used_for_selection": False,
    })

    dev, qualification, sequence = calibration_selection(events)
    for name, rows in (("LORA_DEV16", dev), ("LORA_QUAL16", qualification), ("LORA_SEQ16", sequence)):
        private_file = args.output_root / f"private/{name}_INPUTS.jsonl"
        write_new(private_file, rows, jsonl=True)
        write_new(args.output_root / f"{name}_MANIFEST.json", {
            "schema_version": f"m3bench-{name.lower()}-manifest-v1",
            "status": "FROZEN_BEFORE_LORA_OUTPUT",
            "event_count": len(rows),
            "event_ids": [row["event_id"] for row in rows],
            "private_inputs_sha256": sha256(private_file),
            "method_outputs_used_for_selection": False,
        })
    print(json.dumps({"status": "PASS", "no_edit_queries": len(private), "qual8": len(qual8), "dev16": len(dev), "qual16": len(qualification), "seq16": len(sequence)}, sort_keys=True))


def _load_adapter(args: argparse.Namespace):
    from scripts.editor_paperspec_formal import assert_authorized_device, assert_official_llavamed_source

    source = assert_official_llavamed_source()
    assert_authorized_device()
    sys.path.insert(0, str(source))
    if args.runtime == "formal":
        from scripts.editor_paperspec_formal import load_runtime

        runtime = load_runtime(args.cpu_gate, "cuda:0")
        return runtime.adapter, runtime.generation_config

    from m3bench_repro.inference.llava_med import LlavaMedAdapter

    lock = read_json(args.cpu_gate / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json")
    config = read_json(args.cpu_gate / "inputs/frozen/llava_med_generation_frozen.json")
    adapter = LlavaMedAdapter(
        lock["generation_config"]["model_path"],
        lock["generation_config"]["vision_tower_path"],
        device="cuda:0",
        load_mode="official_native",
    )
    adapter.load()
    return adapter, config


def no_edit_run(args: argparse.Namespace) -> None:
    inputs = read_jsonl(args.inputs)
    output = args.output_root / f"private/NO_EDIT_{args.runtime.upper()}.jsonl"
    done = read_jsonl(output) if output.exists() else []
    if [row["query_id"] for row in done] != [row["query_id"] for row in inputs[: len(done)]]:
        raise RuntimeError("no-edit resume is not an exact manifest prefix")
    if len(done) == len(inputs):
        print(json.dumps({"status": "ALREADY_COMPLETE", "runtime": args.runtime, "completed": len(done)}))
        return
    adapter, generation = _load_adapter(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for ordinal, row in enumerate(inputs[len(done):], len(done) + 1):
            record = {
                "query_id": row["query_id"],
                "runtime": args.runtime,
                "raw_token_ids": [],
                "prompt_token_ids": [],
                "decoded_text": "",
                "normalized_text": "",
                "image_sha256": "",
                "empty": True,
                "error": None,
            }
            try:
                batch = adapter.prepare_inputs(row["image_path"], row["question"])
                result = adapter.generate_with_result(row["image_path"], row["question"], generation)
                record.update({
                    "raw_token_ids": list(result.raw_token_ids),
                    "prompt_token_ids": [int(value) for value in batch["input_ids"][0].detach().cpu().tolist()],
                    "decoded_text": result.decoded_text,
                    "normalized_text": normalize(result.decoded_text),
                    "image_sha256": batch["image_sha256"],
                    "empty": not bool(result.decoded_text.strip()),
                })
            except Exception as error:  # preserve the exact failing row and stop after fsync
                record["error"] = f"{type(error).__name__}: {error}"
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            progress = args.output_root / f"private/NO_EDIT_{args.runtime.upper()}_PROGRESS.json"
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({"completed": ordinal, "total": len(inputs)}) + "\n", encoding="utf-8")
            os.replace(temporary, progress)
            if record["error"]:
                raise RuntimeError(record["error"])
    print(json.dumps({"status": "PASS", "runtime": args.runtime, "completed": len(inputs)}))


def no_edit_compare(args: argparse.Namespace) -> None:
    inputs = read_jsonl(args.inputs)
    official = read_jsonl(args.output_root / "private/NO_EDIT_OFFICIAL.jsonl")
    formal = read_jsonl(args.output_root / "private/NO_EDIT_FORMAL.jsonl")
    if len(inputs) != len(official) or len(inputs) != len(formal):
        raise RuntimeError("no-edit output coverage mismatch")
    checks = Counter()
    mismatches = []
    for expected, left, right in zip(inputs, official, formal, strict=True):
        row_checks = {
            "query_id_exact": expected["query_id"] == left["query_id"] == right["query_id"],
            "official_formal_raw_token_ids_exact": left["raw_token_ids"] == right["raw_token_ids"],
            "official_formal_prompt_token_ids_exact": left["prompt_token_ids"] == right["prompt_token_ids"],
            "official_formal_decoded_exact": left["decoded_text"] == right["decoded_text"],
            "official_formal_normalized_exact": left["normalized_text"] == right["normalized_text"],
            "frozen_official_decoded_exact": expected["frozen_base_decoded_text"].strip() == left["decoded_text"],
            "frozen_formal_decoded_exact": expected["frozen_base_decoded_text"].strip() == right["decoded_text"],
            "frozen_official_normalized_exact": expected["frozen_base_normalized"] == left["normalized_text"],
            "frozen_formal_normalized_exact": expected["frozen_base_normalized"] == right["normalized_text"],
            "image_sha_exact": expected["image_sha256"] == left["image_sha256"] == right["image_sha256"],
            "empty_error_zero": not left["empty"] and not right["empty"] and left["error"] is None and right["error"] is None,
            "semantic_verdict_preserved_by_exact_output": expected["frozen_base_decoded_text"].strip() == right["decoded_text"],
        }
        checks.update(key for key, passed in row_checks.items() if passed)
        if not all(row_checks.values()):
            mismatches.append({"query_id": expected["query_id"], "failed_checks": [key for key, passed in row_checks.items() if not passed]})
    total = len(inputs)
    passed = not mismatches
    report = {
        "schema_version": "m3bench-no-edit-parity-report-v1",
        "status": "PASS__TWO_ANCHOR_NO_EDIT_PARITY" if passed else "M3BENCH_GPU_GATE_BLOCKED__NO_EDIT_PARITY_MISMATCH__RUNTIME",
        "query_count": total,
        "passed_query_count": total - len(mismatches),
        "check_pass_counts": dict(checks),
        "mismatches": mismatches,
        "frozen_token_id_reference_available": False,
        "evidence_limitation": "Frozen canonical base rows contain decoded text but no raw or prompt token IDs; token-ID exactness is therefore proven between a fresh exact-official run and the formal runtime, while frozen raw anchors decoded/normalized exactness.",
        "official_output_sha256": sha256(args.output_root / "private/NO_EDIT_OFFICIAL.jsonl"),
        "formal_output_sha256": sha256(args.output_root / "private/NO_EDIT_FORMAL.jsonl"),
    }
    write_new(args.output_root / "NO_EDIT_PARITY_REPORT.json", report)
    print(json.dumps({"status": report["status"], "query_count": total, "mismatch_count": len(mismatches)}, sort_keys=True))
    if not passed:
        raise SystemExit(3)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    prepare_command = sub.add_parser("prepare")
    prepare_command.add_argument("--cpu-gate", type=Path, required=True)
    prepare_command.add_argument("--data-root", type=Path, required=True)
    prepare_command.add_argument("--output-root", type=Path, required=True)
    prepare_command.set_defaults(func=prepare)
    run = sub.add_parser("no-edit-run")
    run.add_argument("--cpu-gate", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime", choices=("official", "formal"), required=True)
    run.set_defaults(func=no_edit_run)
    compare = sub.add_parser("no-edit-compare")
    compare.add_argument("--inputs", type=Path, required=True)
    compare.add_argument("--output-root", type=Path, required=True)
    compare.set_defaults(func=no_edit_compare)
    return result


def main() -> None:
    if len(sys.argv) == 1 and os.environ.get("M3BENCH_PRIVATE_ARGV"):
        sys.argv.extend(json.loads(os.environ["M3BENCH_PRIVATE_ARGV"]))
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
