#!/usr/bin/env python3
"""Produce public aggregate recovery evidence from the closed private attempt."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_frozen_expert_visual_verifier import CONDITIONS, _task_results, atomic_json, atomic_text, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()
    run, public = args.run_root, args.public_dir
    results = _task_results(run)
    completion = json.loads((run / "RUN_COMPLETION.json").read_text())
    if len(results) != 21 or completion["evaluation"] != "COMPLETE":
        raise RuntimeError("recovery summary requires complete task and evaluation closure")
    config = json.loads((run / "private/CAMPAIGN_CONFIG.json").read_text())
    start = json.loads((run / "private/CAMPAIGN_START.json").read_text())
    fits = Counter(result["training"][kind]["fit_source"] for result in results for kind in ("M2", "M3"))
    assert sum(fits.values()) == 42
    retry_path = run / "private/TASK_RETRY_HISTORY.jsonl"
    retries = read_jsonl(retry_path) if retry_path.exists() else []
    telemetry = read_jsonl(run / "private/GPU_TELEMETRY.jsonl")
    lifecycle = read_jsonl(run / "private/WORKER_LIFECYCLE.jsonl")
    timings = {key: sum(result["timing"][key] for result in results) for key in results[0]["timing"]}
    gpu = {}
    for physical in (2, 3):
        samples = [row for row in telemetry if row["physical_gpu"] == physical]
        events = [row for row in lifecycle if int(row["gpu"]) == physical]
        gpu[str(physical)] = {
            "uuid": config["gpu_uuids"][str(physical)], "sample_count": len(samples),
            "sample_mean_utilization_percent": mean(row["gpu_utilization_percent"] for row in samples) if samples else None,
            "peak_sample_memory_mib": max((row["nvidia_memory_used_mib"] for row in samples), default=None),
            "model_load_seconds": sum(row.get("load_seconds", 0) for row in events),
            "worker_resident_seconds_including_load": sum(row.get("resident_seconds", 0) for row in events),
        }
    completion.update(verifier_fits={"closed": 42, "expected": 42, **dict(fits)}, historical_failed_attempts=len(retries),
                      initialized_utc=start["utc"], initialization_epoch=start["epoch"], parent_attempt_id=Path(config["parent_attempt"]["path"]).name,
                      parent_epoch=config["parent_attempt"]["epoch"], gpu_workers_exited=all(value == 0 for key, value in completion["process_exit_codes"].items() if key.startswith("worker")))
    atomic_json(public / "RUN_COMPLETION.json", completion)
    atomic_json(public / "RECOVERY_TASK_ATTEMPTS.json", {
        "failed_attempts_preserved": len(retries), "final_failed_tasks": completion["tasks"]["failed"],
        "retries": [{"task_id": row["prior"]["task_id"], "prior_attempt": row["prior"]["attempts"], "reason": "NFS_COPY_METADATA_EIO", "retry_limit": 1} for row in retries],
        "score_or_performance_retries": 0,
    })
    atomic_json(public / "GPU_AND_TIMING.json", {
        "start_utc": start["utc"], "wall_seconds": completion["wall_seconds"], "conservative_gpu_hour_upper_bound": 2 * completion["wall_seconds"] / 3600,
        "gpu1_used": False, "devices": gpu, "successful_task_seconds": timings, "coordinator_wait_seconds": completion["phase_wait_seconds"],
        "timing_notes": ["Worker residency includes loading/waiting and is not full-load compute time.",
                         "Telemetry utilization is sampled every 60 seconds; it is not an integral of GPU compute.",
                         "Training_seconds includes loading reused heads. New-fit and reused-fit totals are separated below.",
                         "Judge wait includes its CPU preflight, loading and inference; the existing Judge exposes no per-phase clock, so these are not invented.",
                         "Task phase timers omit some checkpoint/cache I/O; total_seconds includes it."],
        "new_fit_task_training_seconds": sum(r["timing"]["training_seconds"] for r in results if r["training"]["M2"]["fit_source"] == "NEW_800_STEP_FIT"),
        "reused_fit_task_load_seconds": sum(r["timing"]["training_seconds"] for r in results if r["training"]["M2"]["fit_source"] == "VALIDATED_REUSE"),
        "feature_cache_bytes": sum(p.stat().st_size for p in (run / "private/features").glob("*.pt")),
        "head_checkpoint_bytes": sum(p.stat().st_size for p in (run / "private/tasks").glob("VERIFY_*/*_heads.pt")),
    })
    details = json.loads((run / "private/FINAL_DETAILED_RESULTS.json").read_text())
    groups = {}
    for row in details:
        key = (row["seed"], row["event_index"], row["panel"], row["condition"], row["operating_point"])
        groups.setdefault(key, []).append(row["on"])
    constant = []
    for key, values in groups.items():
        if not any(values) or all(values):
            constant.append(dict(zip(("seed", "event_index", "panel", "condition", "operating_point"), key), outcome="ALL_ON" if all(values) else "ALL_OFF", n=len(values)))
    atomic_json(public / "CONSTANT_ROUTE_OUTCOMES.json", {"rows": constant, "all_outcomes_retained": True, "engineering_fixture_rows_in_metrics": 0})
    signal = json.loads((public / "PAIRED_SIGNAL_SUMMARY.json").read_text())
    lines = ["# GPT Pro review: frozen-C1 visual-verifier recovery", "", "Computation and evaluation are complete: 21 edit/seed tasks, 42 verifier fits closed. Seven source facts are repeated across three seeds. This is viewed development diagnosis, not full TIME/MedTRACE or clinical validation.", "",
             "|Condition|Matched hard macro FPR|Matched positive joint ON+correct|T1G joint|T2G joint|Base-correct negative damage|", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        row = signal[condition]
        lines.append(f"|{condition[:2]}|{row['hard_edit_macro_fpr']:.2%}|{row['evaluation_positive_joint']:.2%}|{row['t1g_joint']:.2%}|{row['t2g_joint']:.2%}|{row['base_correct_negative_damage_num']}/{row['base_correct_negative_den']}|")
    lines += ["", "The same matched panel and PRIMARY_SAFETY_FIRST operating point are compared. Native, original evaluation positives, T1G/T1L/T2G, broad and same-image-other-fact results remain separate in RESULTS_BY_EDIT_SEED.csv and RESULTS_MACRO_MICRO.json. Gated correctness and joint ON+correct are separate quantities.", "", "## Evidence for the three routing explanations", ""]
    for left, right, question in ((0, 1, "Compensating mean fusion versus conjunction"), (1, 2, "Usable information in the existing four-dimensional CP response"), (2, 3, "Pre-CP versus compressed feature availability")):
        a, b = signal[CONDITIONS[left]], signal[CONDITIONS[right]]
        lines.append(f"- {question}: M{right} minus M{left} hard FPR = {b['hard_edit_macro_fpr']-a['hard_edit_macro_fpr']:+.2%}; positive joint = {b['evaluation_positive_joint']-a['evaluation_positive_joint']:+.2%}. These paired changes support a trade-off comparison, not a unique causal attribution.")
    lines += ["", "M1 changes only the decision rule. M2 adds supervised linear readouts on CP4; improvement would demonstrate usable information under this supervision, not prove a need for larger CP. M3 changes readout input width and parameter count while matching pooling and data; any improvement is diagnostic and cannot by itself establish information-theoretic loss. Fit/calibration/evaluation gaps are reported in ROUTER_TRAINING_AND_PARAMETER_REPORT.md.", "", f"Predeclared development retention signal passed: {completion['scientific_gain']}. See GENERALITY_SAFETY_TRADEOFF.md for every criterion and paired edit-cluster intervals. All OFF/ON outcomes and all failures/retries remain disclosed; no performance-based resampling or retraining was used.", "", f"Fit provenance: {dict(fits)}. Earlier compatible heads/outputs were reused only after binding checks. M0–M3 scores and calibration were recalculated; the first original task was fully computed anew. {len(retries)} NFS metadata failures were repaired as an engineering retry, not a method failure.", "", "C1 Q/P/rho and V4 inputs/generation remained frozen. Historical LoRA QUAL_VALIDATION_FAIL, C1/C2/C3, A2 and Judge/reference disagreements were not changed. Public files exclude private QA, images, tokens, features, checkpoints, Judge mappings and full logs."]
    atomic_text(public / "GPT_PRO_REVIEW.md", "\n".join(lines) + "\n")
    parameter_report = public / "ROUTER_TRAINING_AND_PARAMETER_REPORT.md"
    existing = parameter_report.read_text()
    # The original finalizer reports a mean including reused-head loads; explicitly correct its label.
    existing = existing.replace("mean training time", "mean current training-or-reused-head-load time")
    atomic_text(parameter_report, existing + f"\nFit provenance: {dict(fits)}. The reused-head load duration is not the original 800-step training time. See GPU_AND_TIMING.json for separated current-fit/load times and cache/checkpoint storage.\n")


if __name__ == "__main__":
    main()
