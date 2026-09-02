#!/usr/bin/env python3
"""Validate and freeze the manifest-driven formal runner without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editor_paperspec_formal import load_records


METHOD_SOURCES = (
    "m3bench_repro/editors/llava_runtime.py",
    "m3bench_repro/editors/methods.py",
    "m3bench_repro/editors/routed_layers.py",
    "m3bench_repro/editors/routing.py",
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_frozen(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite runner lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    os.chmod(path, 0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method-source-commit", required=True)
    parser.add_argument("--records-path", default="inputs/frozen/FORMAL_EDITOR_RECORDS_189.jsonl")
    parser.add_argument("--expected-record-count", type=int, default=189)
    parser.add_argument("--prefixes", default="1,50,100,189")
    parser.add_argument("--final-prefix", type=int, default=189)
    parser.add_argument("--sequence-label", default="M3BENCH_AMENDED_EXCLUSION_ONLY_189")
    args = parser.parse_args()

    records_path = args.run_root / args.records_path
    records = load_records(args.run_root, args.records_path, args.expected_record_count)
    prefixes = tuple(int(value) for value in args.prefixes.split(","))
    catalog = read_jsonl(args.run_root / "inputs/frozen/FORMAL_PROBE_CATALOG.jsonl")
    record_ids = {record.record_id for record in records}
    head = subprocess.check_output(
        ["git", "-C", str(args.worktree), "rev-parse", "HEAD"], text=True
    ).strip()

    current_hashes = {path: sha256(args.worktree / path) for path in METHOD_SOURCES}
    locked_hashes = {
        path: sha_bytes(
            subprocess.check_output(
                ["git", "-C", str(args.worktree), "show", f"{args.method_source_commit}:{path}"]
            )
        )
        for path in METHOD_SOURCES
    }
    checks = {
        "records_exact": len(records) == args.expected_record_count,
        "record_ids_unique": len(record_ids) == args.expected_record_count,
        "positions_dense": [record.formal_sequence_position for record in records]
        == list(range(1, args.expected_record_count + 1)),
        "prefix_contract": prefixes == (1, 50, 100, args.expected_record_count),
        "final_prefix_exact": args.final_prefix == args.expected_record_count,
        "catalog_edit_ids_bound": all(row["edit_id"] in record_ids for row in catalog),
        "every_record_has_t0": all(
            sum(row["edit_id"] == record_id and row["task"] == "T0" for row in catalog) == 1
            for record_id in record_ids
        ),
        "method_sources_unchanged": current_hashes == locked_hashes,
    }
    preflight = {
        "schema_version": "m3bench-formal-runner-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "record_count": len(records),
        "catalog_count": len(catalog),
        "sequence_label": args.sequence_label,
        "prefixes": list(prefixes),
    }
    lock = {
        "schema_version": "m3bench-formal-runner-lock-v1",
        "scoring_code_commit": head,
        "method_source_commit": args.method_source_commit,
        "method_source_sha256": current_hashes,
        "runner_sha256": sha256(args.worktree / "scripts/editor_paperspec_formal.py"),
        "records_sha256": sha256(records_path),
        "sequence_label": args.sequence_label,
        "expected_record_count": args.expected_record_count,
        "prefixes": list(prefixes),
        "final_prefix": args.final_prefix,
    }
    output = args.run_root / "runtime/amended189"
    write_frozen(output / "FORMAL_RUNNER_LOCK.json", lock)
    write_frozen(output / "FORMAL_RUNNER_PREFLIGHT.json", preflight)
    if preflight["status"] != "PASS":
        raise SystemExit(1)
    marker = output / "M3BENCH_AMENDED189_RUNNER_PREFLIGHT_PASS"
    marker.write_text("PASS\n", encoding="utf-8")
    os.chmod(marker, 0o444)
    print(json.dumps({"status": "PASS", "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
