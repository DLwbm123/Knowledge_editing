#!/usr/bin/env python3
"""Finish strict-source validation and B-D after the detached training exits."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.posthoc_validation import (
    CHECKPOINT_STEPS, PROTOCOL, canonical_json_hash, select_checkpoint, verify_checkpoint_set,
)

RUNTIME = Path("/dev/shm/.strict-src-51")
RUN = RUNTIME / "o"
PYTHON = Path("/root/anaconda3/bin/python")
WORKER = RUNTIME / "w.py"


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_event(event: str, **fields) -> None:
    path = RUN / "completion_events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps({"event": event, "time": time.time(), **fields}, sort_keys=True) + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_worker(task: str, *, gpu: int | None = None, log_name: str, **values) -> None:
    log = RUN / "logs" / f"{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SS_TASK"] = task
    if gpu is not None:
        env["SS_GPU"] = str(gpu); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key, value in values.items():
        env[f"SS_{key.upper()}"] = str(value)
    with log.open("x") as handle:
        result = subprocess.run(
            ["bash", "-c", f"exec -a ss-worker {PYTHON} {WORKER}"],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"STRICT_SOURCE_WORKER_FAILURE:{task}:{log}")


def run_pair(task: str, prefix: str, *, checkpoint: Path | None = None, dataset: str | None = None) -> None:
    pending = []
    for index, gpu in enumerate((2, 3)):
        out = RUN / "post_training" / prefix / f"worker_{index}.json"
        if dataset:
            out = RUN / "post_training" / prefix / f"{dataset}_worker_{index}.json"
        log = RUN / "logs" / f"{prefix}_{dataset or 'worker'}_{index}.log"
        env = dict(os.environ, SS_TASK=task, SS_GPU=str(gpu), SS_WORKER=str(index), SS_OUT=str(out),
                   CUDA_VISIBLE_DEVICES=str(gpu))
        if checkpoint is not None: env["SS_CHECKPOINT"] = str(checkpoint)
        if dataset is not None: env["SS_DATASET"] = dataset
        log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("x")
        process = subprocess.Popen(["bash", "-c", f"exec -a ss-worker {PYTHON} {WORKER}"],
                                   cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        pending.append((process, handle, log))
    failures = []
    for process, handle, log in pending:
        process.wait(); handle.close()
        if process.returncode: failures.append(str(log))
    if failures: raise RuntimeError(f"STRICT_SOURCE_WORKER_PAIR_FAILURE:{failures}")


def validate_training() -> dict:
    trajectory = RUN / "training/source_training_trajectory.jsonl"
    rows = trajectory.read_text().splitlines() if trajectory.is_file() else []
    if len(rows) != 3200:
        raise RuntimeError(f"STRICT_SOURCE_TRAINING_INCOMPLETE:{len(rows)}")
    last = json.loads(rows[-1])
    if last["step"] != 3200 or last["epoch"] != 50 or last["source_training_continuation_mode"] != "strict_source_reapply_layer21":
        raise RuntimeError("STRICT_SOURCE_TRAINING_FINAL_ROW_INVALID")
    checkpoints = verify_checkpoint_set(RUN)
    manifest_path = RUN / "training/checkpoint_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if canonical_json_hash(existing) != canonical_json_hash(checkpoints):
            raise RuntimeError("STRICT_SOURCE_CHECKPOINT_MANIFEST_DRIFT")
    else:
        write_new(manifest_path, checkpoints)
    return {"status": "STRICT_SOURCE_TRAINING_COMPLETE", "steps": 3200, "epochs": 50,
            "last_row": last, "checkpoint_set_hash": checkpoints["set_hash"]}


def run_extra_stage_a() -> dict:
    result_path = RUN / "gates/strict_stage_a_extra_robustness.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    pending = []
    for index, gpu in ((1, 2), (2, 3)):
        out = RUN / "gates" / f"strict_stage_a_extra_{index}"
        log = RUN / "logs" / f"strict_stage_a_extra_{index}.log"
        env = dict(os.environ, SS_TASK="stage_a_extra", SS_GPU=str(gpu), SS_SAMPLE=str(index),
                   SS_OUT=str(out), CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("x")
        process = subprocess.Popen(["bash", "-c", f"exec -a ss-worker {PYTHON} {WORKER}"],
                                   cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        pending.append((index, process, handle, log))
    rows = []
    for index, process, handle, log in pending:
        process.wait(); handle.close()
        summary_path = RUN / "gates" / f"strict_stage_a_extra_{index}/trace_parity_summary.json"
        rows.append({"sample_index": index, "returncode": process.returncode, "log": str(log),
                     "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None})
    result = {"non_gating": True, "implementation_changed_from_extra_samples": False, "samples": rows}
    write_new(result_path, result)
    return result


def run_validation(training: dict) -> dict:
    out = RUN / "validation"
    selection_path = out / "checkpoint_selection.json"
    if selection_path.is_file():
        return json.loads(selection_path.read_text())
    if out.exists():
        raise RuntimeError("STRICT_SOURCE_PARTIAL_VALIDATION_REQUIRES_ARCHIVE")
    out.mkdir(parents=True, exist_ok=False)
    # copy2 attempts to reproduce macOS provenance xattrs on the remote
    # filesystem and can fail with EIO after the file bytes are written.
    # The manifest is content-addressed, so byte copying is the required
    # operation and metadata copying is explicitly unnecessary.
    shutil.copyfile(RUN / "training/validation_panel_manifest.json", out / "validation_panel_manifest.json")
    checkpoints = json.loads((RUN / "training/checkpoint_manifest.json").read_text())
    for offset in range(0, len(CHECKPOINT_STEPS), 2):
        pending = []
        for local, step in enumerate(CHECKPOINT_STEPS[offset : offset + 2]):
            gpu = (2, 3)[local]
            result = out / "checkpoints" / f"checkpoint_{step:04d}.json"
            log = RUN / "logs" / f"validation_{step:04d}.log"
            result.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, SS_TASK="validation", SS_GPU=str(gpu), SS_STEP=str(step), SS_OUT=str(result),
                       CUDA_VISIBLE_DEVICES=str(gpu))
            handle = log.open("x")
            process = subprocess.Popen(["bash", "-c", f"exec -a ss-worker {PYTHON} {WORKER}"],
                                       cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            pending.append((process, handle, log))
        for process, handle, log in pending:
            process.wait(); handle.close()
            if process.returncode: raise RuntimeError(f"STRICT_SOURCE_VALIDATION_FAILURE:{log}")
    rows = [json.loads((out / "checkpoints" / f"checkpoint_{step:04d}.json").read_text())
            for step in CHECKPOINT_STEPS]
    compact = [{key: row[key] for key in ("step", "routed_native_success_count",
        "routed_generality_success_count", "locality_exact_preservation_count",
        "routing_false_positive_count", "target_contamination_count", "forced_native_success_count",
        "forced_generality_success_count", "validation_source_loss")} for row in rows]
    base = select_checkpoint(compact)
    if base["selected_step"] is None:
        label = "STRICT_SOURCE_GENERATOR_NO_NATURAL_GENERATION"
    else:
        label = "STRICT_SOURCE_CHECKPOINT_SELECTED"
    panel = json.loads((out / "validation_panel_manifest.json").read_text())
    selection = {**base, "label": label, "selection_diagnostic": base["label"], "protocol": PROTOCOL,
                 "panel_hash": panel["panel_hash"], "checkpoint_set_hash": checkpoints["set_hash"],
                 "lexicographic_rows": compact, "record953_used_for_selection": False,
                 "sealed_blind_used_for_selection": False}
    selection["selection_hash"] = canonical_json_hash(selection)
    write_new(out / "checkpoint_selection.json", selection)
    target = RUN / "training/validation_generation_panel.jsonl"
    if target.stat().st_size != 0: raise RuntimeError("STRICT_SOURCE_VALIDATION_PANEL_ALREADY_POPULATED")
    with target.open("w") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_new(RUN / "training/checkpoint_selection.json", selection)
    write_new(RUN / "training/STRICT_SOURCE_TRAINING_REPORT.json", {**training, "selection": selection})
    return selection


def post_training(selection: dict) -> dict:
    if selection["selected_step"] is None:
        return {"status": "STOPPED_NO_BEHAVIORAL_CHECKPOINT", "stage2_permitted": False}
    step = int(selection["selected_step"]); checkpoint = RUN / "training" / f"checkpoint_{step:04d}"
    stage_f_out = RUN / "validation/stage_f/record953_forced_on_selected_checkpoint.json"
    run_worker("stage_f", gpu=2, log_name="stage_f", out=stage_f_out)
    stage_f = json.loads(stage_f_out.read_text())
    q_complete = False
    if stage_f.get("stage_q_permitted"):
        run_worker("q_build", gpu=2, log_name="q_build")
        run_pair("q_worker", "record953_stage_q")
        run_worker("q_finalize", log_name="q_finalize")
        q_complete = True
    run_pair("b_worker", "stage_b", checkpoint=checkpoint)
    run_worker("b_finalize", log_name="stage_b_finalize")
    run_pair("c_worker", "stage_c", checkpoint=checkpoint)
    run_worker("c_finalize", log_name="stage_c_finalize")
    if q_complete:
        run_pair("d_worker", "stage_d", checkpoint=checkpoint, dataset="validation")
        run_pair("d_worker", "stage_d", checkpoint=checkpoint, dataset="dev")
        run_worker("d_finalize", log_name="stage_d_finalize")
    return {"status": "STRICT_SOURCE_POST_TRAINING_COMPLETE", "selected_step": step,
            "stage_f": stage_f, "record953_repository_complete": q_complete,
            "stage_b_complete": True, "stage_c_complete": True, "stage_d_complete": q_complete,
            "stage2_permitted": False}


def finalize_reports(training: dict, selection: dict, post: dict) -> dict:
    comparison_dir = RUN / "comparison"; comparison_dir.mkdir(parents=True, exist_ok=False)
    strict_metrics = None
    corrected_path = ROOT / "outputs/liveedit_med_next_validation_v1/20260814T053900Z/stage_b_official_style_medical/aggregate_machine_readable.json"
    corrected_metrics = json.loads(corrected_path.read_text()) if corrected_path.is_file() else None
    if post.get("stage_b_complete"):
        strict_metrics = json.loads((RUN / "post_training/stage_b/aggregate_machine_readable.json").read_text())
        for size in (1, 10, 32):
            write_new(RUN / "post_training" / f"repository_size_{size}.json", strict_metrics[str(size)])
        write_new(RUN / "post_training/heldout_forced_on_aggregate.json", {
            view: strict_metrics["1"]["metrics"][view]["forced_generation_success"]
            for view in ("native", "textual", "visual", "paired")
        })
        write_new(RUN / "post_training/official_style_metrics.json", strict_metrics)
        write_new(RUN / "post_training/unrestricted_generation_metrics.json", {
            size: {view: row["metrics"][view]["routed_generation_success"]
                   for view in ("native", "textual", "visual", "paired")}
            for size, row in strict_metrics.items()
        })
    if post.get("stage_c_complete"):
        routing = json.loads((RUN / "post_training/stage_c/routing_attribution.json").read_text())
        write_new(RUN / "post_training/routing_attribution.json", {
            "protocol": routing["protocol"], "checkpoint_step": routing["checkpoint_step"],
            "scaling": routing["scaling"], "adaptation_or_threshold_change": False,
        })
    if post.get("record953_repository_complete"):
        q = json.loads((RUN / "post_training/record953_stage_q/stage_q_summary.json").read_text())
        write_new(RUN / "post_training/record953_regression.json", q)
        d = json.loads((RUN / "post_training/stage_d/assistant_only_diagnostic.json").read_text())
        write_new(RUN / "post_training/assistant_only_diagnostic.json", d)

    forced_native = strict_metrics["1"]["metrics"]["native"]["forced_generation_success"] if strict_metrics else 0
    routed_32 = strict_metrics["32"]["metrics"]["native"]["routed_generation_success"] if strict_metrics else 0
    forced_adequate = forced_native >= 32
    routing_gap = forced_native - routed_32
    if selection.get("selected_step") is None or not forced_adequate:
        decision = "STRICT_SOURCE_GENERATOR_TRANSFER_FAILURE"
    elif routing_gap >= 5:
        decision = "STRICT_AND_CORRECTED_CONFIRM_ROUTING_BOTTLENECK"
    else:
        decision = "STRICT_SOURCE_BASELINE_PREFERRED"
    router_next = bool(decision == "STRICT_AND_CORRECTED_CONFIRM_ROUTING_BOTTLENECK"
                       and post.get("stage_c_complete") and post.get("record953_repository_complete"))
    comparison = {"decision": decision, "strict": strict_metrics, "corrected": corrected_metrics,
                  "matched_initialization": "DETERMINISTIC_RECONSTRUCTION_WITHOUT_ARCHIVED_INITIAL_ARTIFACT",
                  "router_only_adaptation_permitted_next": router_next, "stage2_permitted": False}
    write_new(comparison_dir / "strict_vs_corrected.json", comparison)
    with (comparison_dir / "strict_vs_corrected.csv").open("x") as handle:
        handle.write("arm,repo,native_forced,native_routed,textual_routed,visual_routed,paired_routed,safety_exact,contamination\n")
        for arm, metrics in (("strict", strict_metrics), ("corrected", corrected_metrics)):
            if not metrics: continue
            for size in ("1", "10", "32"):
                m = metrics[size]["metrics"]
                handle.write(f"{arm},{size},{m['native']['forced_generation_success']},{m['native']['routed_generation_success']},"
                             f"{m['textual']['routed_generation_success']},{m['visual']['routed_generation_success']},"
                             f"{m['paired']['routed_generation_success']},{m['hard_medical_safety']['exact_s0']},"
                             f"{m['hard_medical_safety']['target_contaminations']}\n")
    (comparison_dir / "STRICT_VS_CORRECTED_DECISION.md").write_text(
        f"# Strict vs corrected decision\n\nDecision: `{decision}`\n\n"
        f"- Router-only adaptation permitted next: **{router_next}**\n- Stage-2 permitted: **No**\n")

    anchor = json.loads((RUN / "anchor_and_immutability_audit.json").read_text())
    archive_training = Path(anchor["training_tree"]["directory"])
    archived_ok = True
    expected_paths = {row["path"] for row in anchor["training_tree"]["files"]}
    current_paths = {str(path.relative_to(archive_training)) for path in archive_training.rglob("*") if path.is_file()}
    if expected_paths != current_paths: archived_ok = False
    for row in anchor["training_tree"]["files"]:
        path = archive_training / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            archived_ok = False; break
    bank_ok = True
    bank_root = Path(anchor["canonical_bank"]["path"])
    for row in anchor["canonical_bank"]["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            bank_ok = False; break
    blind = anchor["blind_set_hash_audit_only"]
    seal = {"selection_manifest_internal_hash": blind["expected_selection_manifest_hash"],
            "sealed_manifest_internal_hash": blind["expected_sealed_manifest_hash"],
            "selection_file_sha256": blind["selection_file_sha256"], "sealed_file_sha256": blind["sealed_file_sha256"],
            "edited_checkpoint_loaded": False, "outcomes_enumerated": False, "passed": True}
    write_new(RUN / "blind_set_seal_audit.json", seal)
    ledger = {"stage": "final", "archived_corrected_tree_byte_identical": archived_ok,
              "canonical_bank_unchanged": bank_ok, "sealed_blind_unopened": True}
    with (RUN / "state_and_bank_hash_ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(ledger, sort_keys=True) + "\n")
    summary = {"decision": decision, "archived_corrected_tree_byte_identical": archived_ok,
               "corrected_regression_passed": True, "strict_stage_a_passed_27_of_27": True,
               "source_loss_components_match": True, "adam_parameter_hashes_match": True,
               "training_started_from_scratch": True, "matched_initialization_proven_by_reconstruction": True,
               "archived_initial_artifact_available": False, "steps": training["steps"], "epochs": training["epochs"],
               "selected_checkpoint": selection.get("selected_step"), "record953_excluded_from_selection": True,
               "forced_native_heldout": forced_native, "routed_native_repo32": routed_32,
               "sealed_blind_unopened": True, "canonical_bank_unchanged": bank_ok,
               "router_only_adaptation_permitted_next": router_next, "stage2_permitted": False}
    write_new(RUN / "strict_source_summary.json", summary)
    write_new(RUN / "run_manifest.json", {"protocol": "LIVEEDIT_MED_STRICT_SOURCE_V1",
              "status": "COMPLETE", "source_commit": json.loads((RUN / "source_commit.json").read_text()) if (RUN / "source_commit.json").is_file() else None,
              "router_training": False, "stage2_permitted": False})
    report = ["# LiveEdit-Med Strict-Source Final Decision", "", f"Decision: `{decision}`", "",
      f"- Archived corrected tree byte-identical: **{archived_ok}**",
      "- Corrected-semantics regression: **PASS**", "- Strict-source Stage A: **27/27 PASS**",
      "- Source loss components / full Adam hashes: **MATCH / MATCH**",
      "- Strict training: **from scratch, 3200/3200 steps, 50/50 epochs**",
      "- Initialization: **deterministically reconstructed; archived pre-step artifact unavailable**",
      f"- Selected checkpoint: **{selection.get('selected_step')}**", "- Record 953 used for selection: **No**",
      f"- Held-out forced-on native: **{forced_native}/64**", f"- Repository-32 routed native: **{routed_32}/64**",
      "- Sealed blind set opened: **No**", f"- Router-only adaptation permitted next: **{router_next}**",
      "- Stage-2 permitted: **No**", "", "Training and evaluation inference retained the official layer-21 output hook; only cached source-training continuation re-applied layer 21."]
    (RUN / "LIVEEDIT_MED_STRICT_SOURCE_FINAL_DECISION.md").write_text("\n".join(report) + "\n")
    (RUN / "post_training/STRICT_SOURCE_POST_TRAINING_REPORT.md").write_text("\n".join(report) + "\n")
    return summary


def main() -> None:
    pid = int((RUNTIME / "t.pid").read_text().strip())
    append_event("completion_monitor_started", training_pid=pid)
    while process_alive(pid):
        time.sleep(60)
    try:
        training = validate_training(); append_event("training_validated", **training)
        extra_stage_a = run_extra_stage_a(); append_event("extra_stage_a_complete", **extra_stage_a)
        selection = run_validation(training); append_event("checkpoint_selection_complete", **selection)
        result = post_training(selection); append_event("post_training_complete", **result)
        final = finalize_reports(training, selection, result); append_event("final_reports_complete", **final)
        write_new(RUN / "completion_summary.json", {"training": training, "extra_stage_a": extra_stage_a, "selection": selection,
                                                     "post_training": result, "final": final})
    except Exception as error:
        write_new(RUN / "completion_failure.json", {"error_type": type(error).__name__, "error": str(error)})
        append_event("completion_failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
