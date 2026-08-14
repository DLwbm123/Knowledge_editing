#!/usr/bin/env python3
"""Resume a post-hoc validation run with a dynamic GPU 2/3 work queue."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.posthoc_validation import (CHECKPOINT_STEPS, PROTOCOL,
    canonical_json_hash, immutable_tree_manifest, select_checkpoint)


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False


def parse_active(value: str) -> dict[int, tuple[int, int]]:
    active = {}
    for item in value.split(","):
        gpu, step, pid = map(int, item.split(":"))
        if gpu not in (2, 3) or gpu in active:
            raise ValueError("Active workers must uniquely use physical GPUs 2 and 3")
        active[gpu] = (step, pid)
    return active


def launch(gpu: int, step: int, args) -> int:
    result = args.out_dir / "checkpoints" / f"checkpoint_{step:04d}.json"
    log = args.out_dir / "logs" / f"checkpoint_{step:04d}.log"
    if result.exists() or log.exists():
        raise RuntimeError(f"LIVEEDIT_MED_DYNAMIC_OUTPUT_COLLISION:{step}")
    command = [args.python, str(ROOT / "scripts/liveedit_med/evaluate_posthoc_validation_checkpoint.py"),
        "--checkpoint", str(args.run_dir / "training" / f"checkpoint_{step:04d}"),
        "--source-records", str(args.source_records), "--panel-manifest",
        str(args.out_dir / "validation_panel_manifest.json"), "--out", str(result),
        "--physical-gpu", str(gpu)]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    handle = log.open("x")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--active", required=True, help="gpu:step:pid entries")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve(); args.source_records = args.source_records.resolve()
    args.out_dir = args.out_dir.resolve()
    panel = json.loads((args.out_dir / "validation_panel_manifest.json").read_text())
    if panel.get("protocol") != PROTOCOL or panel.get("record953_excluded") is not True:
        raise RuntimeError("LIVEEDIT_MED_INVALID_RESUME_PANEL")
    before = json.loads((args.out_dir / "original_training_tree_before.json").read_text())
    completed = {step for step in CHECKPOINT_STEPS if
                 (args.out_dir / "checkpoints" / f"checkpoint_{step:04d}.json").exists()}
    active = parse_active(args.active)
    if completed & {step for step, _pid in active.values()}:
        raise RuntimeError("LIVEEDIT_MED_ACTIVE_ALREADY_COMPLETED")
    queue = [step for step in CHECKPOINT_STEPS if step not in completed and
             step not in {row[0] for row in active.values()}]
    events = [{"event": "DYNAMIC_RESUME", "completed": sorted(completed),
               "active": {str(gpu): {"step": step, "pid": pid} for gpu, (step, pid) in active.items()},
               "queue": queue}]
    while active or queue:
        for gpu, (step, pid) in list(active.items()):
            result = args.out_dir / "checkpoints" / f"checkpoint_{step:04d}.json"
            if result.exists():
                events.append({"event": "COMPLETED", "gpu": gpu, "step": step, "pid": pid})
                del active[gpu]; completed.add(step)
            elif not alive(pid):
                events.append({"event": "FAILED", "gpu": gpu, "step": step, "pid": pid})
                write_json(args.out_dir / "dynamic_scheduler_failure.json", {"events": events})
                raise RuntimeError(f"LIVEEDIT_MED_DYNAMIC_WORKER_FAILURE:{step}")
        for gpu in (2, 3):
            if gpu not in active and queue:
                step = queue.pop(0); pid = launch(gpu, step, args); active[gpu] = (step, pid)
                events.append({"event": "LAUNCHED", "gpu": gpu, "step": step, "pid": pid})
        time.sleep(5)

    rows = [json.loads((args.out_dir / "checkpoints" / f"checkpoint_{step:04d}.json").read_text())
            for step in CHECKPOINT_STEPS]
    compact = [{key: row[key] for key in ("step", "routed_native_success_count",
        "routed_generality_success_count", "locality_exact_preservation_count",
        "routing_false_positive_count", "target_contamination_count", "forced_native_success_count",
        "forced_generality_success_count", "validation_source_loss")} for row in rows]
    checkpoint_audit = json.loads((args.out_dir / "checkpoint_hash_audit.json").read_text())
    selection = {**select_checkpoint(compact), "protocol": PROTOCOL, "panel_hash": panel["panel_hash"],
                 "checkpoint_set_hash": checkpoint_audit["set_hash"], "lexicographic_rows": compact,
                 "record953_used_for_selection": False, "dynamic_gpu_queue": [2, 3]}
    selection["selection_hash"] = canonical_json_hash(selection)
    write_json(args.out_dir / "checkpoint_selection.json", selection)
    after = immutable_tree_manifest(args.run_dir / "training")
    if before != after:
        write_json(args.out_dir / "original_training_tree_after_mismatch.json", after)
        raise RuntimeError("LIVEEDIT_MED_ORIGINAL_TRAINING_DIRECTORY_MUTATED")
    write_json(args.out_dir / "original_training_immutability.json", {"passed": True,
        "tree_hash_before": before["tree_hash"], "tree_hash_after": after["tree_hash"],
        "file_count": len(before["files"])})
    write_json(args.out_dir / "dynamic_scheduler_events.json", {"events": events})
    print(json.dumps({"protocol": PROTOCOL, "selection": selection}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
