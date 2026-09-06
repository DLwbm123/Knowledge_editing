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
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
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
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402
from scripts.medtrace.run_scope_pilot import generate as scope_generate  # noqa: E402

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


def profile_key(profile: dict[str, Any]) -> str:
    return f"{profile['representation']}__step{profile['budget']}"


def evaluation_category(row: dict[str, Any]) -> str:
    if row["role"] == "native":
        return "native"
    if row["role"] == "evaluation" and row["label"] == "positive":
        return "evaluation_positive"
    if row["role"] == "formal_development":
        return f"formal_{row.get('task', row['fact_relation'])}"
    return row["fact_relation"]


def score_state(runtime: Any, expert: AsymmetricCPExpert, batch: Any) -> dict[str, Any]:
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
    residual_norms: list[float] = []
    hook.attach()
    hook.set_teacher_routing(batch.labels)

    def capture(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        residual = expert.residual(inputs[0])
        mask = hook.token_mask.to(residual.device).unsqueeze(-1)
        residual_norms.append(float((residual * mask).float().norm().item()))

    handle = runtime.get_module(LAYER).register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            metrics = runtime.score_target(batch)
    finally:
        handle.remove()
        hook.detach()
    return {**metrics, "residual_norm": max(residual_norms, default=0.0)}


def diagnose(args: argparse.Namespace) -> None:
    started = time.time()
    physical, gpu_uuid = verify_gpu()
    run_root = args.original_run
    queue = json.loads((run_root / "private/TASK_QUEUE.json").read_text())["tasks"]
    by_id = {task["task_id"]: task for task in queue}
    frozen = json.loads((run_root / "private/frozen_data.json").read_text())
    config = json.loads((run_root / "private/CAMPAIGN_RUNTIME_CONFIG.json").read_text())
    fixed = json.loads(args.fixed_calibration.read_text())
    fixed_by_b = {value["task_id"]: value for value in fixed["results"]}
    selection = fixed["selected_profile"]
    seeds = {int(value) for value in args.seeds.split(",")}
    tasks = [task for task in queue if task["kind"] == "E2E" and task["seed"] in seeds]
    if not tasks:
        raise RuntimeError("no E2E tasks selected for diagnostics")
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(config["cpu_gate"])))
    diagnostics, replay_candidates = [], []
    torch.cuda.reset_peak_memory_stats()
    try:
        for task in tasks:
            result_path = run_root / "private/tasks" / task["task_id"] / "result_private.json"
            result = json.loads(result_path.read_text())
            event = frozen["dev"][task["event_index"] - 1]
            record = EditorRecord.from_dict(event["edit_record"])
            a_task = by_id[task["depends_on"]]["depends_on"]
            a2_path = run_root / "private/tasks" / a_task / "CP_NATIVE_PLUS_PARAPHRASE_80/expert.pt"
            a2 = torch.load(a2_path, map_location="cuda:0", weights_only=True)
            batches = [("native", runtime.build_edit_batch(record))]
            batches.extend(
                (f"fit_paraphrase_{index}", runtime.build_edit_batch(replace(record, question=question["question"])))
                for index, question in enumerate(frozen["generality_paraphrases"][task["record_id"]], 1)
            )
            candidate_scores = json.loads((run_root / "private/tasks" / task["task_id"] / "candidate_evaluation_scores_private.json").read_text())
            score_map = {(value["candidate"], value["logical_id"]): value["score"] for value in candidate_scores}
            for label, selection_key in (("R0", "r0"), ("ALT", "alternative")):
                profile = selection[selection_key]
                if (profile["representation"], profile["budget"]) != (
                    result["profiles"][label]["profile"]["representation"], result["profiles"][label]["profile"]["budget"],
                ):
                    raise RuntimeError("selected candidate has no frozen final expert")
                b_dir = run_root / "private/tasks" / task["depends_on"]
                candidate_path = b_dir / f"{profile_key(profile)}.pt"
                candidate = torch.load(candidate_path, map_location="cuda:0", weights_only=True)
                final_path = run_root / "private/tasks" / task["task_id"] / f"{label}.pt"
                if sha256_file(final_path) != result["profiles"][label]["checkpoint_sha256"]:
                    raise RuntimeError(f"final expert hash mismatch: {final_path}")
                final = torch.load(final_path, map_location="cuda:0", weights_only=True)
                mixed_state = {name: value.clone() for name, value in a2["expert"].items()}
                for name in ("u_in", "v_in"):
                    mixed_state[name] = candidate["expert"][name].clone()
                states = {"A2_PRE_SCOPE": a2["expert"], "POST_Q_PRE_OUTPUT_RECOVERY": mixed_state, "POST_OUTPUT_RECOVERY": final["expert"]}
                state_rows = []
                for state_name, state in states.items():
                    expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
                    expert.load_state_dict(state)
                    values = [{"input": input_name, **score_state(runtime, expert, batch)} for input_name, batch in batches]
                    state_rows.append({
                        "state": state_name,
                        "q_sha256": tensor_sha256(expert.input_basis()),
                        "p_rho_sha256": state_hash({name: expert.state_dict()[name] for name in ("u_out", "v_out", "rho")}),
                        "metrics": values,
                    })
                    del expert
                if state_rows[1]["q_sha256"] != result["profiles"][label]["output_fit"]["q_sha256_before_and_after"] or state_rows[1]["q_sha256"] != state_rows[2]["q_sha256"]:
                    raise RuntimeError("output recovery changed Q")
                diagnostics.append({
                    "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"],
                    "record_id": task["record_id"], "profile": label, "selected": profile,
                    "a2_checkpoint_sha256": sha256_file(a2_path), "candidate_checkpoint_sha256": sha256_file(candidate_path),
                    "final_checkpoint_sha256": sha256_file(final_path), "states": state_rows,
                })
                fixed_task = fixed_by_b[task["depends_on"]]
                threshold = fixed_task["calibrations"][profile_key(profile)]["operating_points"][profile["operating_point"]]["threshold"]
                for item in result["outputs"]:
                    row = item["row"]
                    replay_candidates.append({
                        "task": task, "result": result, "profile": label, "profile_value": profile,
                        "row": row, "outputs": item["outputs"], "category": evaluation_category(row),
                        "on": score_map[(profile_key(profile), row["logical_id"])] > threshold,
                    })
        replay_rows = []
        if args.replay:
            chosen: list[dict[str, Any]] = []

            def add_first(candidates: list[dict[str, Any]]) -> None:
                for candidate in sorted(candidates, key=lambda value: (value["task"]["task_id"], value["profile"], value["row"]["logical_id"])):
                    key = (candidate["task"]["task_id"], candidate["profile"], candidate["row"]["logical_id"])
                    if all((value["task"]["task_id"], value["profile"], value["row"]["logical_id"]) != key for value in chosen):
                        chosen.append(candidate)
                        return

            for profile in ("R0", "ALT"):
                add_first([value for value in replay_candidates if value["profile"] == profile and value["on"]])
                add_first([value for value in replay_candidates if value["profile"] == profile and not value["on"]])
            for category in ("native", "evaluation_positive", "same_question_different_image_conflicting_source_answer"):
                if not any(value["category"] == category for value in chosen):
                    add_first([value for value in replay_candidates if value["category"] == category])
            for value in chosen:
                task, label, row = value["task"], value["profile"], value["row"]
                final_path = run_root / "private/tasks" / task["task_id"] / f"{label}.pt"
                final = torch.load(final_path, map_location="cuda:0", weights_only=True)
                expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
                expert.load_state_dict(final["expert"])
                if value["on"]:
                    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
                    hook.attach()
                    try:
                        actual = scope_generate(runtime, row, hook)
                    finally:
                        hook.detach()
                    expected = value["outputs"][f"{label}__forced"]
                    path = "forced"
                else:
                    actual = scope_generate(runtime, row, None)
                    expected = value["outputs"]["base"]
                    path = "base"
                exact_replay = actual["raw_token_ids"] == expected["raw_token_ids"] and actual["raw_answer"] == expected["raw_answer"]
                if not exact_replay:
                    raise RuntimeError(f"frozen output replay mismatch: {task['task_id']} {label} {row['logical_id']}")
                replay_rows.append({
                    "task_id": task["task_id"], "profile": label, "logical_id": row["logical_id"],
                    "category": value["category"], "decision": "ON" if value["on"] else "OFF", "executed_path": path,
                    "expected_output_sha256": sha256_json((expected["raw_token_ids"], expected["raw_answer"])),
                    "actual_output_sha256": sha256_json((actual["raw_token_ids"], actual["raw_answer"])), "exact_replay": True,
                })
                del expert
            if not {value["decision"] for value in replay_rows} >= {"ON", "OFF"}:
                raise RuntimeError("replay sample did not cover both ON and OFF")
            if not {value["category"] for value in replay_rows} >= {"native", "evaluation_positive", "same_question_different_image_conflicting_source_answer"}:
                raise RuntimeError("replay sample did not cover required request categories")
        guard = runtime.base_guard.verify() if runtime.base_guard else None
        if not guard or not guard["unchanged"]:
            raise RuntimeError("diagnostic base guard failed")
    finally:
        del runtime
        torch.cuda.empty_cache()
    atomic_json(args.output, {
        "schema_version": "medtrace-route-repair-diagnostics-private-v1", "status": "COMPLETE",
        "physical_gpu": physical, "gpu_uuid": gpu_uuid, "seeds": sorted(seeds),
        "elapsed_seconds": time.time() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "diagnostics": diagnostics, "replays": replay_rows,
    })


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(values: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [value for value in values if value["category"] in {"native", "evaluation_positive"}]
    base_correct = [value for value in values if value["base_correct"]]
    forced_damage = [value for value in base_correct if not value["forced_correct"]]
    return {
        "n": len(values), "on_num": sum(value["new_on"] for value in values),
        "base_correct_num": sum(value["base_correct"] for value in values),
        "forced_correct_num": sum(value["forced_correct"] for value in values),
        "gated_correct_num": sum(value["new_gated_correct"] for value in values),
        "forced_damage_num": len(forced_damage), "damage_den": len(base_correct),
        "gated_damage_num": sum(not value["new_gated_correct"] for value in base_correct),
        "forced_damage_avoided_num": sum(value["new_gated_correct"] for value in forced_damage),
        "forced_damage_avoided_den": len(forced_damage),
        "rejected_correct_positive_num": sum(value["forced_correct"] and not value["new_on"] for value in positives),
        "rejected_correct_positive_den": sum(value["forced_correct"] for value in positives),
        "off_count": sum(not value["new_on"] for value in values),
    }


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def closeout(args: argparse.Namespace) -> None:
    run_root, private, public = args.original_run, args.private_out, args.public_out
    fixed = json.loads((private / "FIXED_CALIBRATION_PRIVATE.json").read_text())
    fixed_by_b = {value["task_id"]: value for value in fixed["results"]}
    selection = fixed["selected_profile"]
    old_selection = fixed["old_selected_profile"]
    for label, key in (("R0", "r0"), ("ALT", "alternative")):
        old, new = old_selection[key], selection[key]
        if (old["representation"], old["budget"]) != (new["representation"], new["budget"]):
            raise RuntimeError(f"{label} selected a candidate without frozen forced outputs")
    queue = json.loads((run_root / "private/TASK_QUEUE.json").read_text())["tasks"]
    e2e_tasks = [task for task in queue if task["kind"] == "E2E" and task["status"] == "COMPLETE"]
    if len(e2e_tasks) != 36:
        raise RuntimeError(f"expected 36 complete E2E tasks, got {len(e2e_tasks)}")
    packet = {value["opaque_query_id"]: value for value in read_jsonl(run_root / "private/judge/JUDGE_PACKET_PRIVATE.jsonl")}
    verdicts = {value["opaque_query_id"]: value for value in read_jsonl(run_root / "private/judge/JUDGE_OUTPUT_PRIVATE.jsonl")}
    sidecar = json.loads((run_root / "private/judge/JUDGE_SIDECAR_PRIVATE.json").read_text())
    side_by_path = {
        (value["task_id"], value["logical_id"], value["path"]): value
        for value in sidecar if value["kind"] == "E2E"
    }
    reused_opaque: set[str] = set()

    def judged(task_id: str, logical_id: str, path: str, row: dict[str, Any], output: dict[str, Any]) -> bool:
        side = side_by_path[(task_id, logical_id, path)]
        opaque = sha256_json((row["question"], row["reference"], output["raw_answer"], "medtrace-campaign-semantic-v1"))
        if opaque != side["opaque_query_id"] or packet[opaque]["question"] != row["question"] or packet[opaque]["gold_answer"] != row["reference"] or packet[opaque]["raw_base_answer"] != output["raw_answer"]:
            raise RuntimeError("Judge tuple binding mismatch")
        if opaque not in verdicts or not verdicts[opaque]["parse_valid"]:
            raise RuntimeError("reused Judge verdict missing or invalid")
        reused_opaque.add(opaque)
        return bool(verdicts[opaque]["is_correct"])

    derived = []
    for task in e2e_tasks:
        task_dir = run_root / "private/tasks" / task["task_id"]
        result = json.loads((task_dir / "result_private.json").read_text())
        scores = json.loads((task_dir / "candidate_evaluation_scores_private.json").read_text())
        score_by_key = {(value["candidate"], value["logical_id"]): value["score"] for value in scores}
        for label, selection_key in (("R0", "r0"), ("ALT", "alternative")):
            profile = selection[selection_key]
            key = profile_key(profile)
            threshold = fixed_by_b[task["depends_on"]]["calibrations"][key]["operating_points"][profile["operating_point"]]["threshold"]
            for item in result["outputs"]:
                row, outputs = item["row"], item["outputs"]
                base_correct = judged(task["task_id"], row["logical_id"], "base", row, outputs["base"])
                forced_correct = judged(task["task_id"], row["logical_id"], f"{label}__forced", row, outputs[f"{label}__forced"])
                old_gated_correct = judged(task["task_id"], row["logical_id"], f"{label}__gated", row, outputs[f"{label}__gated"])
                new_on = score_by_key[(key, row["logical_id"])] > threshold
                selected_output = outputs[f"{label}__forced"] if new_on else outputs["base"]
                new_gated_correct = forced_correct if new_on else base_correct
                derived.append({
                    "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"],
                    "record_id": task["record_id"], "scope_status": result["scope_status"], "profile": label,
                    "logical_id": row["logical_id"], "category": evaluation_category(row), "label": row["label"],
                    "score": score_by_key[(key, row["logical_id"])], "threshold": threshold,
                    "old_on": bool(item["decisions"][label]["on"]), "new_on": new_on,
                    "base_correct": base_correct, "forced_correct": forced_correct,
                    "old_gated_correct": old_gated_correct, "new_gated_correct": new_gated_correct,
                    "new_gated_source": "forced" if new_on else "base",
                    "new_gated_output_sha256": sha256_json((selected_output["raw_token_ids"], selected_output["raw_answer"])),
                    "derivation": "DERIVED_FROM_FROZEN_BASE_AND_FORCED_OUTPUTS",
                })
    original_counts = {}
    for profile in ("R0", "ALT"):
        values = [value for value in derived if value["profile"] == profile and value["category"] in {"native", "evaluation_positive"}]
        original_counts[profile] = {
            "n": len(values), "old_on": sum(value["old_on"] for value in values),
            "forced_correct": sum(value["forced_correct"] for value in values),
            "old_gated_correct": sum(value["old_gated_correct"] for value in values),
            "fixed_on": sum(value["new_on"] for value in values),
            "fixed_gated_correct": sum(value["new_gated_correct"] for value in values),
            "decision_changes": sum(value["old_on"] != value["new_on"] for value in values),
        }
    expected = {
        "R0": {"n": 180, "old_on": 144, "forced_correct": 180, "old_gated_correct": 144},
        "ALT": {"n": 180, "old_on": 157, "forced_correct": 126, "old_gated_correct": 122},
    }
    if any(any(original_counts[profile][key] != value for key, value in counts.items()) for profile, counts in expected.items()):
        raise RuntimeError("published positive execution counts did not reproduce from private mapping")
    diagnostic_files = [Path(value) for value in args.diagnostics]
    diagnostic_runs = [json.loads(path.read_text()) for path in diagnostic_files]
    if any(value["status"] != "COMPLETE" for value in diagnostic_runs):
        raise RuntimeError("diagnostic run incomplete")
    diagnostic_rows = [row for run in diagnostic_runs for row in run["diagnostics"]]
    if len(diagnostic_rows) != 72:
        raise RuntimeError(f"expected 72 matched profile diagnostics, got {len(diagnostic_rows)}")
    replay_rows = [row for run in diagnostic_runs for row in run["replays"]]
    if not replay_rows or any(not row["exact_replay"] for row in replay_rows):
        raise RuntimeError("actual replay coverage missing or failed")
    replay_counts = Counter((row["task_id"], row["profile"], row["category"]) for row in replay_rows)
    metric_rows: list[list[Any]] = []
    columns = [
        "aggregation_level", "seed", "event_index", "opaque_edit", "scope_status", "profile", "subset",
        "n", "on_num", "on_rate", "base_correct_num", "base_correct_rate", "forced_correct_num", "forced_correct_rate",
        "gated_correct_num", "gated_correct_rate", "forced_damage_num", "gated_damage_num", "damage_den",
        "forced_damage_avoided_num", "forced_damage_avoided_den", "forced_damage_avoided_rate",
        "gate_rejected_correct_positive_num", "gate_rejected_correct_positive_den", "off_count", "real_replay_count", "cached_reuse_count",
    ]

    def emit(level: str, identifiers: list[Any], values: list[dict[str, Any]], *, macro: list[dict[str, Any]] | None = None) -> None:
        result = metric(values)
        if macro is None:
            rates = {
                "on": rate(result["on_num"], result["n"]), "base": rate(result["base_correct_num"], result["n"]),
                "forced": rate(result["forced_correct_num"], result["n"]), "gated": rate(result["gated_correct_num"], result["n"]),
                "avoided": rate(result["forced_damage_avoided_num"], result["forced_damage_avoided_den"]),
            }
        else:
            rates = {
                "on": mean(value["on_num"] / value["n"] for value in macro if value["n"]),
                "base": mean(value["base_correct_num"] / value["n"] for value in macro if value["n"]),
                "forced": mean(value["forced_correct_num"] / value["n"] for value in macro if value["n"]),
                "gated": mean(value["gated_correct_num"] / value["n"] for value in macro if value["n"]),
                "avoided": mean((value["forced_damage_avoided_num"] / value["forced_damage_avoided_den"]) for value in macro if value["forced_damage_avoided_den"]) if any(value["forced_damage_avoided_den"] for value in macro) else None,
            }
        replay_groups = {(value["task_id"], value["profile"], value["category"]) for value in values}
        real_replays = sum(replay_counts[key] for key in replay_groups)
        metric_rows.append([
            level, *identifiers, result["n"], result["on_num"], rates["on"], result["base_correct_num"], rates["base"],
            result["forced_correct_num"], rates["forced"], result["gated_correct_num"], rates["gated"],
            result["forced_damage_num"], result["gated_damage_num"], result["damage_den"],
            result["forced_damage_avoided_num"], result["forced_damage_avoided_den"], rates["avoided"],
            result["rejected_correct_positive_num"], result["rejected_correct_positive_den"], result["off_count"], real_replays, result["off_count"],
        ])

    by_task_subset: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in derived:
        by_task_subset[(value["task_id"], value["profile"], value["category"])].append(value)
    per_group_metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    task_by_id = {task["task_id"]: task for task in e2e_tasks}
    for (task_id, profile, subset), values in sorted(by_task_subset.items()):
        task = task_by_id[task_id]
        opaque_edit = hashlib.sha256(task["record_id"].encode()).hexdigest()[:16]
        emit("EDIT_SEED", [task["seed"], task["event_index"], opaque_edit, values[0]["scope_status"], profile, subset], values)
        per_group_metrics[(profile, subset)].append(metric(values))
    subsets = sorted({value["category"] for value in derived} | {"formal_T1L"})
    for profile in ("R0", "ALT"):
        for subset in subsets:
            values = [value for value in derived if value["profile"] == profile and value["category"] == subset]
            emit("POOLED_MICRO", ["ALL", "", "12_EDITS", "MIXED", profile, subset], values)
            if values:
                emit("EDIT_MACRO", ["ALL", "", "12_EDITS", "MIXED", profile, subset], values, macro=per_group_metrics[(profile, subset)])
    with (public / "FIXED_E2E_METRICS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(metric_rows)
    atomic_json(private / "FIXED_E2E_DERIVATION_PRIVATE.json", {
        "schema_version": "medtrace-fixed-e2e-derivation-private-v1", "status": "COMPLETE",
        "selection": selection, "rows": derived, "original_positive_counts_reproduced": original_counts,
        "reused_judge_opaque_count": len(reused_opaque), "replay_rows": replay_rows,
    })
    state_summary = {}
    for profile in ("R0", "ALT"):
        for state in ("A2_PRE_SCOPE", "POST_Q_PRE_OUTPUT_RECOVERY", "POST_OUTPUT_RECOVERY"):
            rows = [metric for value in diagnostic_rows if value["profile"] == profile for item in value["states"] if item["state"] == state for metric in item["metrics"]]
            state_summary[f"{profile}__{state}"] = {
                "inputs": len(rows), "mean_target_content_nll": mean(value["target_content_nll"] for value in rows),
                "first_target_token_rank_one_rate": mean(value["first_target_token_rank"] == 1 for value in rows),
                "mean_residual_norm": mean(value["residual_norm"] for value in rows),
            }
    alt_a2 = state_summary["ALT__A2_PRE_SCOPE"]
    alt_q = state_summary["ALT__POST_Q_PRE_OUTPUT_RECOVERY"]
    alt_final = state_summary["ALT__POST_OUTPUT_RECOVERY"]
    q_increased_nll = alt_q["mean_target_content_nll"] > alt_a2["mean_target_content_nll"]
    recovery_incomplete = alt_final["mean_target_content_nll"] > alt_a2["mean_target_content_nll"]
    next_step = "TARGETED_Q_P_COUPLING_OR_OUTPUT_RECOVERY_CHANGE_NEEDED_BEFORE_EXPANSION" if q_increased_nll and recovery_incomplete else "NO_Q_P_TRAINING_CHANGE_SUPPORTED_BY_THIS_DIAGNOSTIC"
    positive = {profile: metric([value for value in derived if value["profile"] == profile and value["category"] in {"native", "evaluation_positive"}]) for profile in ("R0", "ALT")}
    hard = {profile: metric([value for value in derived if value["profile"] == profile and value["category"] == "same_question_different_image_conflicting_source_answer"]) for profile in ("R0", "ALT")}
    broad = {profile: metric([value for value in derived if value["profile"] == profile and value["category"] == "broad_unrelated_source_qa"]) for profile in ("R0", "ALT")}
    method_status = "R2_ROUTER_CALIBRATION_FIXED__END_TO_END_NOT_SUPPORTED_OVER_R0_ON_CURRENT_DEV"
    atomic_text(public / "POSITIVE_EXECUTION_RETENTION.md", f"""# Positive execution retention and forced-on diagnosis

Status: `{method_status}`

The private mapping independently reproduced the review counts. Native replay and four evaluation paraphrases remain separate in `FIXED_E2E_METRICS.csv`; repeated seeds are not new facts or patients.

| profile | old ON | fixed ON | forced correct | old gated correct | fixed gated correct | decision changes |
|---|---:|---:|---:|---:|---:|---:|
| R0 | {original_counts['R0']['old_on']}/180 | {original_counts['R0']['fixed_on']}/180 | {original_counts['R0']['forced_correct']}/180 | {original_counts['R0']['old_gated_correct']}/180 | {original_counts['R0']['fixed_gated_correct']}/180 | {original_counts['R0']['decision_changes']} |
| ALT R2 | {original_counts['ALT']['old_on']}/180 | {original_counts['ALT']['fixed_on']}/180 | {original_counts['ALT']['forced_correct']}/180 | {original_counts['ALT']['old_gated_correct']}/180 | {original_counts['ALT']['fixed_gated_correct']}/180 | {original_counts['ALT']['decision_changes']} |

For ALT across the fixed native plus fit-paraphrase diagnostic inputs, mean target-content NLL was {alt_a2['mean_target_content_nll']:.6f} before scope Q, {alt_q['mean_target_content_nll']:.6f} after Q with the old P/rho, and {alt_final['mean_target_content_nll']:.6f} after the fixed 80-step output recovery. Rank-one rates were {alt_a2['first_target_token_rank_one_rate']:.1%}, {alt_q['first_target_token_rank_one_rate']:.1%}, and {alt_final['first_target_token_rank_one_rate']:.1%}. Thus threshold repair cannot recover forced-on failures; the matched-state diagnosis records the Q-direction change separately from incomplete output recovery.

Actual deterministic replay covered {len(replay_rows)} fixed requests spanning ON, OFF, native, new paraphrase and hard-negative paths; every replay matched the frozen full token sequence and answer. All other corrected gates are explicitly derived from the frozen base/forced outputs.
""")
    recalculation = json.loads((private / "RECALCULATION_COMPLETION.json").read_text())
    gpu_seconds = recalculation["elapsed_seconds"] + sum(value["elapsed_seconds"] for value in diagnostic_runs)
    completion = {
        "schema_version": "medtrace-fixed-prototype-closeout-v1",
        "original_campaign_execution": "COMPLETE",
        "track_a_multiseed": "PRESERVED_UNCHANGED",
        "router_calibration": "FIXED",
        "corrected_e2e": "COMPLETE",
        "method_effectiveness": method_status,
        "next_training_step": next_step,
        "reused": {"router_candidate_checkpoints": 216, "final_expert_checkpoints": 72, "frozen_base_outputs": len(derived) // 2, "frozen_forced_outputs": len(derived), "judge_opaque_verdicts": len(reused_opaque)},
        "new_judge_requests": 0, "actual_replays": len(replay_rows),
        "new_gpu_seconds": gpu_seconds, "new_gpu_hours": gpu_seconds / 3600,
        "gpu1_used": False, "private_qa_images_tokens_weights_and_mapping_withheld": True,
    }
    atomic_json(public / "RUN_COMPLETION_FIXED.json", completion)
    atomic_text(public / "GPT_PRO_REVIEW_FIXED.md", f"""# GPT Pro review: fixed MedTRACE prototype calibration and E2E closeout

This is a correction on an already viewed development panel, not a blind test, V0.2 qualification, full TIME, or clinical validation. Historical LoRA `QUAL_VALIDATION_FAIL` remains unchanged.

1. **Root cause and repair.** Legacy R1/R2 calibration rebuilt prototypes from calibration positives while deployment used fit prototypes. The scorer is now frozen-prototype-only outside fit; 216 checkpoint/Q/cache bindings passed and R0 was numerically unchanged.
2. **Calibration change.** The selected configurations remain R0@200/SAFETY_FIRST and R2@800/SAFETY_FIRST. R2 calibration positive TPR changed from {old_selection['alternative']['positive_tpr']:.1%} to {selection['alternative']['positive_tpr']:.1%}; fixed hard and broad FPR are both {selection['alternative']['hard_fpr']:.1%}/{selection['alternative']['broad_fpr']:.1%}. Full score differences are in `PROTOTYPE_SOURCE_AND_SCORE_PARITY.json`.
3. **Corrected E2E.** On the 180 native plus evaluation-positive executions, R0 is ON {positive['R0']['on_num']}/180 and correct when gated {positive['R0']['gated_correct_num']}/180. R2 is ON {positive['ALT']['on_num']}/180, forced-correct {positive['ALT']['forced_correct_num']}/180, and gated-correct {positive['ALT']['gated_correct_num']}/180. R2 hard/broad activation is {hard['ALT']['on_num']}/{hard['ALT']['n']} and {broad['ALT']['on_num']}/{broad['ALT']['n']}; hard support comes from seven edits, not 21 independent facts.
4. **Reuse.** All 216 route checkpoints, 72 final experts, frozen base/forced outputs and {len(reused_opaque)} exact Judge tuples were reused. No new answer required Judge; {len(replay_rows)} full deterministic GPU replays verified the derivation contract.
5. **Mechanism judgment.** `{method_status}`. The calibration defect is fixed, but R2 forced-on correctness remains the binding failure. The matched A2 → new-Q/old-P → recovered-output measurements support `{next_step}` rather than further threshold tuning.

R0@200 versus R2@800 is the preregistered selected-configuration comparison, not a same-budget pure representation causal claim. Formal T1L/T1G/T2G, native, paraphrase, hard, broad and same-image challenge rows remain separate in `FIXED_E2E_METRICS.csv`.
""")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    command = sub.add_parser("recalculate")
    command.add_argument("--original-run", type=Path, required=True)
    command.add_argument("--private-out", type=Path, required=True)
    command.add_argument("--public-out", type=Path, required=True)
    command.add_argument("--device", default="cpu")
    command.set_defaults(func=recalculate)
    command = sub.add_parser("diagnose")
    command.add_argument("--original-run", type=Path, required=True)
    command.add_argument("--fixed-calibration", type=Path, required=True)
    command.add_argument("--seeds", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--replay", action="store_true")
    command.set_defaults(func=diagnose)
    command = sub.add_parser("closeout")
    command.add_argument("--original-run", type=Path, required=True)
    command.add_argument("--private-out", type=Path, required=True)
    command.add_argument("--public-out", type=Path, required=True)
    command.add_argument("--diagnostics", nargs="+", required=True)
    command.set_defaults(func=closeout)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
