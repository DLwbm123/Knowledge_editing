#!/usr/bin/env python3
"""Orchestrate no-leakage post-hoc validation for a completed LiveEdit-Med run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.posthoc_validation import (CHECKPOINT_STEPS, PROTOCOL,
    canonical_json_hash, freeze_validation_panel, immutable_tree_manifest, select_checkpoint,
    verify_checkpoint_set)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def parse_gpus(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError("--physical-gpus must contain unique GPU indices")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", default="0,1,2")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    run_dir, source_path, out = args.run_dir.resolve(), args.source_records.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads(source_path.read_text())

    # This irreversible precommitment is written before checkpoint verification
    # loads any tensors.  Workers receive only the resulting manifest.
    panel = freeze_validation_panel(source)
    write_json(out / "validation_panel_manifest.json", panel)
    write_json(out / "recovery_manifest.json", {"protocol": PROTOCOL,
        "input_training_run": str(run_dir), "training_status":
        "LIVEEDIT_SOURCE_TRAINING_CONVERGED__BEHAVIORAL_VALIDATION_PENDING",
        "panel_hash": panel["panel_hash"], "record953_accessed": False,
        "original_training_directory_write_permitted": False})

    training_before = immutable_tree_manifest(run_dir / "training")
    write_json(out / "original_training_tree_before.json", training_before)
    checkpoints = verify_checkpoint_set(run_dir)
    write_json(out / "checkpoint_hash_audit.json", checkpoints)
    gpus = parse_gpus(args.physical_gpus)
    pending = []
    for index, step in enumerate(CHECKPOINT_STEPS):
        result = out / "checkpoints" / f"checkpoint_{step:04d}.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        log = out / "logs" / f"checkpoint_{step:04d}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        command = [args.python, str(ROOT / "scripts/liveedit_med/evaluate_posthoc_validation_checkpoint.py"),
                   "--checkpoint", str(run_dir / "training" / f"checkpoint_{step:04d}"),
                   "--source-records", str(source_path), "--panel-manifest", str(out / "validation_panel_manifest.json"),
                   "--out", str(result), "--physical-gpu", str(gpus[index % len(gpus)])]
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
        handle = log.open("x")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        pending.append((step, process, handle, command, log))
        # Limit active workers to the declared GPU count.
        if len(pending) == len(gpus) or index == len(CHECKPOINT_STEPS) - 1:
            failures = []
            for item in pending:
                item[1].wait(); item[2].close()
                if item[1].returncode != 0:
                    failures.append({"step": item[0], "returncode": item[1].returncode,
                                     "command": item[3], "log": str(item[4])})
            if failures:
                write_json(out / "worker_failures.json", failures)
                raise RuntimeError("LIVEEDIT_MED_POSTHOC_VALIDATION_WORKER_FAILURE")
            pending = []

    rows = [json.loads((out / "checkpoints" / f"checkpoint_{step:04d}.json").read_text())
            for step in CHECKPOINT_STEPS]
    compact = [{key: row[key] for key in ("step", "routed_native_success_count",
        "routed_generality_success_count", "locality_exact_preservation_count",
        "routing_false_positive_count", "target_contamination_count", "forced_native_success_count",
        "forced_generality_success_count", "validation_source_loss")} for row in rows]
    selection = {**select_checkpoint(compact), "protocol": PROTOCOL, "panel_hash": panel["panel_hash"],
                 "checkpoint_set_hash": checkpoints["set_hash"], "lexicographic_rows": compact,
                 "record953_used_for_selection": False}
    selection["selection_hash"] = canonical_json_hash(selection)
    write_json(out / "checkpoint_selection.json", selection)
    training_after = immutable_tree_manifest(run_dir / "training")
    if training_before != training_after:
        write_json(out / "original_training_tree_after_mismatch.json", training_after)
        raise RuntimeError("LIVEEDIT_MED_ORIGINAL_TRAINING_DIRECTORY_MUTATED")
    write_json(out / "original_training_immutability.json", {"passed": True,
        "tree_hash_before": training_before["tree_hash"], "tree_hash_after": training_after["tree_hash"],
        "file_count": len(training_before["files"])})
    print(json.dumps({"protocol": PROTOCOL, "selection": selection}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
