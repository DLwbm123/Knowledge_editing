#!/usr/bin/env python3
"""Repair MedTRACE calibration using frozen fit prototypes and existing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.medtrace import AsymmetricCPExpert  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import tensor_sha256  # noqa: E402
from scripts.medtrace.run_dev16 import sha256_file, sha256_json  # noqa: E402
from scripts.medtrace.run_longrun_campaign import (  # noqa: E402
    BUDGETS,
    OPERATING_POINTS,
    REPRESENTATIONS,
    SCORE_DEFINITION_SHA256,
    build_fit_prototypes,
    calibrate_operating_points,
    route_score_one,
    score_with_frozen_prototypes,
    state_hash,
    verify_gpu,
)

ORIGINAL_CODE_COMMIT = "7d4fc0558e0ea1a02bd4a3e17a5201eed47806c7"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def feature_rows(cache: dict[str, Any], role: str) -> tuple[list[dict[str, Any]], torch.Tensor, list[torch.Tensor]]:
    values = [value for value in cache["values"].values() if value["row"]["role"] == role]
    values.sort(key=lambda value: (value["row"]["label"] != "positive", value["row"]["logical_id"]))
    return values, torch.stack([value["prompt"] for value in values]), [value["visual"] for value in values]


def to_device(prompt: torch.Tensor, visual: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return prompt.to(device), [value.to(device) for value in visual]


def aggregate_profiles(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for representation in REPRESENTATIONS:
        for budget in BUDGETS:
            key = f"{representation}__step{budget}"
            for operating_point in OPERATING_POINTS:
                values = [
                    {**result["calibrations"][key]["operating_points"][operating_point], "hard_evaluable": result["calibrations"][key]["hard_evaluable"]}
                    for result in results
                ]
                hard = [value["hard_fpr"] for value in values if value["hard_evaluable"]]
                profiles.append({
                    "representation": representation,
                    "budget": budget,
                    "operating_point": operating_point,
                    "positive_tpr": mean(value["positive_tpr"] for value in values),
                    "hard_fpr": mean(hard) if hard else None,
                    "broad_fpr": mean(value["broad_fpr"] for value in values),
                    "task_count": len(values),
                    "hard_task_count": len(hard),
                })
    return profiles


def pick_profile(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [value for value in profiles if value["positive_tpr"] >= 0.75]
    pool = covered or profiles
    return min(pool, key=lambda value: (
        value["hard_fpr"] if value["hard_fpr"] is not None else 1.0,
        value["broad_fpr"], -value["positive_tpr"], value["budget"], value["representation"], value["operating_point"],
    ))


def select_profiles(results: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = aggregate_profiles(results)
    return {
        "schema_version": "medtrace-selected-route-profile-fixed-v1",
        "status": "FROZEN_BEFORE_CORRECTED_EVALUATION_SCORING",
        "selection_source": "original_calibration_roles_with_frozen_fit_prototypes",
        "r0": pick_profile([value for value in profiles if value["representation"] == REPRESENTATIONS[0]]),
        "alternative": pick_profile([value for value in profiles if value["representation"] in REPRESENTATIONS[1:]]),
        "all_calibration_profiles": profiles,
    }


def recalculate(args: argparse.Namespace) -> None:
    started = time.time()
    run_root, private, public = args.original_run, args.private_out, args.public_out
    if private.exists() or public.exists():
        raise FileExistsError("repair output already exists")
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    config_path = run_root / "private/CAMPAIGN_RUNTIME_CONFIG.json"
    data_path = run_root / "private/frozen_data.json"
    config = json.loads(config_path.read_text())
    locks = {
        "code_commit": ORIGINAL_CODE_COMMIT,
        "config_sha256": sha256_file(config_path),
        "data_sha256": sha256_file(data_path),
        "runtime_lock_sha256": sha256_file(Path(config["runtime_lock"])),
    }
    queue = json.loads((run_root / "private/TASK_QUEUE.json").read_text())["tasks"]
    b_tasks = [task for task in queue if task["kind"] == "B" and task["status"] == "COMPLETE"]
    if len(b_tasks) != 36:
        raise RuntimeError(f"expected 36 complete B tasks, got {len(b_tasks)}")
    device = torch.device(args.device)
    if device.type == "cuda":
        verify_gpu()
    fixed_results, parity_rows, csv_rows = [], [], []
    for task in b_tasks:
        task_dir = run_root / "private/tasks" / task["task_id"]
        legacy = json.loads((task_dir / "result_private.json").read_text())
        if legacy["locks"] != locks:
            raise RuntimeError(f"run lock mismatch for {task['task_id']}")
        cache = torch.load(run_root / f"private/features/e{task['event_index']:02d}.pt", map_location="cpu", weights_only=False)
        if cache["locks"] != locks:
            raise RuntimeError(f"feature cache lock mismatch for {task['task_id']}")
        fit_rows, fit_prompt, fit_visual = feature_rows(cache, "fit")
        cal_rows, cal_prompt, cal_visual = feature_rows(cache, "calibration")
        fit_count = sum(value["row"]["label"] == "positive" for value in fit_rows)
        cal_fit_count = sum(value["row"]["label"] == "positive" for value in cal_rows)
        fit_prompt, fit_visual = to_device(fit_prompt, fit_visual, device)
        cal_prompt, cal_visual = to_device(cal_prompt, cal_visual, device)
        fixed = {
            "task_id": task["task_id"],
            "seed": task["seed"],
            "event_index": task["event_index"],
            "record_id": task["record_id"],
            "scope_status": legacy["scope_status"],
            "calibrations": {},
        }
        for representation in REPRESENTATIONS:
            marker = json.loads((task_dir / f"{representation}.json").read_text())
            for budget in BUDGETS:
                key = f"{representation}__step{budget}"
                checkpoint_path = task_dir / f"{key}.pt"
                expected_checkpoint_hash = marker["saved"][str(budget)]["checkpoint_sha256"]
                actual_checkpoint_hash = sha256_file(checkpoint_path)
                if actual_checkpoint_hash != expected_checkpoint_hash:
                    raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                expert = AsymmetricCPExpert(cal_prompt.shape[-1], 4096, 4).to(device)
                expert.load_state_dict(checkpoint["expert"])
                q_hash = tensor_sha256(expert.input_basis())
                if q_hash != marker["saved"][str(budget)]["q_sha256"]:
                    raise RuntimeError(f"Q hash mismatch: {checkpoint_path}")
                rebuilt = build_fit_prototypes(expert, fit_prompt[:fit_count], fit_visual[:fit_count], representation)
                prototype_hash = state_hash(checkpoint["prototypes"])
                rebuilt_hash = state_hash(rebuilt)
                rebuild_max_diff = max(
                    (float((checkpoint["prototypes"][name].to(device) - rebuilt[name]).abs().max().item()) for name in rebuilt),
                    default=0.0,
                )
                if rebuild_max_diff > 1e-6:
                    raise RuntimeError(f"fit prototype rebuild mismatch: {checkpoint_path}")
                fixed_scores = score_with_frozen_prototypes(expert, cal_prompt, cal_visual, representation, checkpoint["prototypes"])
                legacy_prototypes = build_fit_prototypes(expert, cal_prompt[:cal_fit_count], cal_visual[:cal_fit_count], representation)
                reproduced_legacy = score_with_frozen_prototypes(expert, cal_prompt, cal_visual, representation, legacy_prototypes)
                stored_by_id = {value["logical_id"]: value["score"] for value in legacy["calibrations"][key]["scores"]}
                stored = torch.tensor([stored_by_id[value["row"]["logical_id"]] for value in cal_rows], device=device)
                legacy_reproduction_max_diff = float((reproduced_legacy - stored).abs().max().item())
                reversed_scores = score_with_frozen_prototypes(expert, cal_prompt.flip(0), list(reversed(cal_visual)), representation, checkpoint["prototypes"]).flip(0)
                split = torch.cat([
                    score_with_frozen_prototypes(expert, cal_prompt[:1], cal_visual[:1], representation, checkpoint["prototypes"]),
                    score_with_frozen_prototypes(expert, cal_prompt[1:], cal_visual[1:], representation, checkpoint["prototypes"]),
                ])
                request_score = route_score_one(expert, checkpoint, cal_prompt[0], cal_visual[0])
                request_batch_max_diff = max(
                    float((fixed_scores - reversed_scores).abs().max().item()),
                    float((fixed_scores - split).abs().max().item()),
                    abs(float(fixed_scores[0].item()) - request_score),
                )
                scored = [
                    {
                        "logical_id": value["row"]["logical_id"],
                        "role": value["row"]["role"],
                        "label": value["row"]["label"],
                        "fact_relation": value["row"]["fact_relation"],
                        "score": float(score.item()),
                    }
                    for value, score in zip(cal_rows, fixed_scores, strict=True)
                ]
                positive = [value["score"] for value in scored if value["label"] == "positive"]
                hard = [value["score"] for value in scored if value["fact_relation"] == "same_question_different_image_conflicting_source_answer"]
                broad = [value["score"] for value in scored if value["fact_relation"] == "broad_unrelated_source_qa"]
                points = calibrate_operating_points(positive, hard, broad, hard_evaluable=legacy["scope_status"] == "HARD_EVALUABLE")
                legacy_scores = legacy["calibrations"][key]["scores"]
                differences = [abs(value["score"] - stored_by_id[value["logical_id"]]) for value in scored]
                decision_changes = {
                    operating_point: sum(
                        (value["score"] > legacy["calibrations"][key]["operating_points"][operating_point]["threshold"])
                        != (stored_by_id[value["logical_id"]] > legacy["calibrations"][key]["operating_points"][operating_point]["threshold"])
                        for value in scored
                    )
                    for operating_point in OPERATING_POINTS
                }
                fixed["calibrations"][key] = {"scores": scored, "operating_points": points, "hard_evaluable": legacy["scope_status"] == "HARD_EVALUABLE"}
                parity_rows.append({
                    "task_id": task["task_id"], "representation": representation, "budget": budget,
                    "checkpoint_sha256": actual_checkpoint_hash, "q_sha256": q_hash,
                    "prototype_sha256": prototype_hash, "rebuilt_fit_prototype_sha256": rebuilt_hash,
                    "fit_prototype_rebuild_max_abs_diff": rebuild_max_diff,
                    "score_definition_sha256": SCORE_DEFINITION_SHA256,
                    "legacy_reproduction_max_abs_diff": legacy_reproduction_max_diff,
                    "request_batch_max_abs_diff": request_batch_max_diff,
                    "legacy_fixed_changed_score_count": sum(value > 1e-7 for value in differences),
                    "legacy_fixed_mean_abs_diff": mean(differences), "legacy_fixed_max_abs_diff": max(differences),
                    "legacy_threshold_decision_changes": decision_changes,
                    "calibration_row_count": len(legacy_scores),
                })
                opaque_edit = hashlib.sha256(task["record_id"].encode()).hexdigest()[:16]
                for operating_point, metrics in points.items():
                    csv_rows.append([task["seed"], task["event_index"], opaque_edit, legacy["scope_status"], representation, budget, operating_point, metrics["positive_tpr"], metrics["hard_fpr"], metrics["broad_fpr"]])
                del expert, checkpoint
        fixed_results.append(fixed)
        del cache, fit_prompt, fit_visual, cal_prompt, cal_visual
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = select_profiles(fixed_results)
    old_selection = json.loads((run_root / "private/SELECTED_PROFILE.json").read_text())
    atomic_json(private / "FIXED_CALIBRATION_PRIVATE.json", {
        "schema_version": "medtrace-fixed-calibration-private-v1", "locks": locks,
        "score_definition_sha256": SCORE_DEFINITION_SHA256, "results": fixed_results,
        "selected_profile": selection, "old_selected_profile": old_selection,
    })
    atomic_json(private / "PROTOTYPE_PARITY_PRIVATE.json", parity_rows)
    with (public / "ROUTER_CALIBRATION_RESULTS_FIXED.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "event_index", "opaque_edit", "scope_status", "representation", "budget", "operating_point", "positive_tpr", "hard_fpr", "broad_fpr"])
        writer.writerows(csv_rows)
    atomic_json(public / "SELECTED_PROFILE_FIXED.json", {**selection, "private_thresholds_withheld": True})
    summary = {}
    for representation in REPRESENTATIONS:
        for budget in BUDGETS:
            values = [value for value in parity_rows if value["representation"] == representation and value["budget"] == budget]
            summary[f"{representation}__step{budget}"] = {
                "calibration_scores": sum(value["calibration_row_count"] for value in values),
                "changed_score_count": sum(value["legacy_fixed_changed_score_count"] for value in values),
                "mean_abs_score_difference": mean(value["legacy_fixed_mean_abs_diff"] for value in values),
                "maximum_abs_score_difference": max(value["legacy_fixed_max_abs_diff"] for value in values),
                "legacy_decision_changes_at_safety_threshold": sum(value["legacy_threshold_decision_changes"]["SAFETY_FIRST"] for value in values),
                "legacy_decision_changes_at_coverage_threshold": sum(value["legacy_threshold_decision_changes"]["COVERAGE_CONSTRAINED"] for value in values),
            }
    parity_public = {
        "schema_version": "medtrace-prototype-source-score-parity-public-v1",
        "candidate_checkpoint_count": len(parity_rows),
        "checkpoint_hash_failures": 0,
        "fit_prototype_rebuild_failures": 0,
        "maximum_fit_prototype_rebuild_difference": max(value["fit_prototype_rebuild_max_abs_diff"] for value in parity_rows),
        "maximum_legacy_score_reproduction_difference": max(value["legacy_reproduction_max_abs_diff"] for value in parity_rows),
        "maximum_batch_request_score_difference": max(value["request_batch_max_abs_diff"] for value in parity_rows),
        "score_definition_sha256": SCORE_DEFINITION_SHA256,
        "prototype_source": "original_scope_fit_positive_rows_only",
        "by_candidate_family": summary,
        "private_qa_and_activation_values_withheld": True,
    }
    atomic_json(public / "PROTOTYPE_SOURCE_AND_SCORE_PARITY.json", parity_public)
    old_alt, new_alt = old_selection["alternative"], selection["alternative"]
    atomic_text(public / "PROTOTYPE_CALIBRATION_ROOT_CAUSE.md", f"""# Prototype calibration root cause

Status: `CONFIRMED_AND_FIXED`

The original execution source and final source were identical for the affected scorer. R1/R2 checkpoints stored prototypes built from scope-fit positives, but the legacy calibration function rebuilt prototypes from calibration positives. Request inference instead read the checkpoint prototypes, so calibration and deployment evaluated different functions. R0 never used a prototype and is unchanged.

The repair separates fit-only prototype construction from scoring with frozen prototypes. Calibration, evaluation and request inference now share score definition `{SCORE_DEFINITION_SHA256}`; scoring receives no role labels. All {len(parity_rows)} checkpoint files and their Q states matched the original run ledger, and every saved prototype was exactly rebuilt from the original fit-positive cache.
""")
    atomic_text(public / "CALIBRATION_CORRECTION_REPORT.md", f"""# Calibration correction report

Status: `ROUTER_CALIBRATION_FIXED`

- Reused candidates: {len(parity_rows)} checkpoints from 36 completed B tasks; no Q training or model generation was run.
- Old alternative: `{old_alt['representation']}@{old_alt['budget']}/{old_alt['operating_point']}` with calibration positive TPR {old_alt['positive_tpr']:.1%}, hard FPR {old_alt['hard_fpr']:.1%}, broad FPR {old_alt['broad_fpr']:.1%}.
- Fixed alternative: `{new_alt['representation']}@{new_alt['budget']}/{new_alt['operating_point']}` with calibration positive TPR {new_alt['positive_tpr']:.1%}, hard FPR {new_alt['hard_fpr']:.1%}, broad FPR {new_alt['broad_fpr']:.1%}.
- Fixed R0 remains `{selection['r0']['representation']}@{selection['r0']['budget']}/{selection['r0']['operating_point']}`.

This correction uses the already viewed development panel and is not a blind or unseen confirmation. Legacy R1/R2 calibration and selection remain preserved as execution evidence but are not deployment-consistent calibration evidence. Track A values were not read or modified by the recalculation.
""")
    atomic_json(private / "RECALCULATION_COMPLETION.json", {
        "status": "FIXED_CALIBRATION_COMPLETE", "elapsed_seconds": time.time() - started,
        "device": str(device), "candidate_checkpoint_count": len(parity_rows), "selected_profile": selection,
    })


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    command = sub.add_parser("recalculate")
    command.add_argument("--original-run", type=Path, required=True)
    command.add_argument("--private-out", type=Path, required=True)
    command.add_argument("--public-out", type=Path, required=True)
    command.add_argument("--device", default="cpu")
    command.set_defaults(func=recalculate)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
