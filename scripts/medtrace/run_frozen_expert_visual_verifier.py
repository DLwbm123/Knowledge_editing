#!/usr/bin/env python3
"""Run frozen-C1 MedTRACE mean/conjunction/CP4/pre-CP verifier diagnostics."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from methods.medtrace.frozen_verifier import (  # noqa: E402
    LinearApplicabilityVerifier,
    calibrate_decisions,
    train_verifier,
    verifier_features,
)
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
from scripts.medtrace.repair_route_calibration import evaluation_category, read_jsonl  # noqa: E402
from scripts.medtrace.run_dev16 import atomic_json, derive_seed, sha256_file, sha256_json  # noqa: E402
from scripts.medtrace.run_longrun_campaign import (  # noqa: E402
    GPU_UUIDS,
    TaskQueue,
    Telemetry,
    append_jsonl,
    atomic_text,
    normalize_rows,
    state_hash,
)
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402
from scripts.medtrace.run_scope_pilot import generate as scope_generate  # noqa: E402


CONDITIONS = (
    "M0_C1_R2_MEAN_CONTROL",
    "M1_C1_CP_CONJUNCTION_CONTROL",
    "M2_C1_CP_RESPONSE_VERIFIER",
    "M3_C1_PRE_CP_FEATURE_VERIFIER",
)
POINTS = ("PRIMARY_SAFETY_FIRST", "SECONDARY_COVERAGE90")
SEEDS = (20260906, 20260907, 20260908)
STOP_REQUESTED = False
HARD_RELATION = "same_question_different_image_conflicting_source_answer"
BROAD_RELATION = "broad_unrelated_source_qa"
MATCHED_PANEL = "MATCHED_QUESTION_IMAGE_PANEL_V1"


def verify_gpu() -> tuple[str, str]:
    physical = os.environ.get("M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES", "")
    expected = os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID", "")
    if physical not in {"2", "3"} or os.environ.get("CUDA_VISIBLE_DEVICES") != physical or expected != GPU_UUIDS.get(physical):
        raise RuntimeError("frozen-verifier GPU authorization is invalid")
    actual = subprocess.check_output(
        ["nvidia-smi", "-i", physical, "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip()
    if actual != expected:
        raise RuntimeError("frozen-verifier GPU UUID mismatch")
    return physical, actual


def _source_group(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("image_name") or row["image_path"]).encode()).hexdigest()[:16]


def matched_panel_rows(scope: dict[str, Any], event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Derive same-question native/hard-image pairs from already frozen roles."""
    if scope["status"] != "HARD_EVALUABLE":
        raise ValueError("matched panel requires HARD_EVALUABLE source support")
    primary, rows, exclusions = scope["primary"], [], []
    for role in ("fit", "calibration", "evaluation"):
        questions = list(scope["positives"][role])
        if role == "fit":
            questions = [{"family": "native", "question": event["edit_record"]["question"], "review_status": "SOURCE_NATIVE"}, *questions]
        deduplicated = []
        seen = set()
        for index, value in enumerate(questions, 1):
            question = str(value.get("question", "")).strip()
            approved = value.get("review_status", "APPROVED_EXISTING_FIT") in {
                "SOURCE_NATIVE", "APPROVED_EXISTING_FIT", "APPROVED_EQUIVALENT",
                "APPROVED_EQUIVALENT_SOURCE_QUESTION_ONLY",
            }
            if not question or not approved:
                exclusions.append({"role": role, "question_index": str(index), "reason": "equivalence_not_proven"})
                continue
            if question in seen:
                exclusions.append({"role": role, "question_index": str(index), "reason": "duplicate_question"})
                continue
            seen.add(question); deduplicated.append((index, value, question))
        hard = [value for value in scope["negative_roles"][role] if value["fact_relation"] == HARD_RELATION]
        if not hard:
            raise RuntimeError(f"{role} has no source-supported hard image")
        for question_index, question_meta, question in deduplicated:
            positive_id = f"matched-{role}-q{question_index}-native"
            common = {
                "role": role, "question": question, "question_family": str(question_meta.get("family") or f"q{question_index}"),
                "panel": MATCHED_PANEL, "shared_native_source": True,
            }
            rows.append({
                **common, "logical_id": positive_id, "label": "positive", "reference": primary["target"],
                "image_path": primary["image_path"], "source_group": hashlib.sha256(str(primary["image_name"]).encode()).hexdigest()[:16],
                "fact_relation": "matched_native_image", "positive_logical_id": positive_id,
            })
            for hard_index, negative in enumerate(hard, 1):
                rows.append({
                    **common, "logical_id": f"matched-{role}-q{question_index}-hard{hard_index}", "label": "negative",
                    "reference": negative["source_answer"], "image_path": negative["image_path"],
                    "source_group": _source_group(negative), "fact_relation": HARD_RELATION,
                    "positive_logical_id": positive_id,
                })
    if len({row["logical_id"] for row in rows}) != len(rows):
        raise RuntimeError("matched panel contains duplicate logical IDs")
    for negative in (row for row in rows if row["label"] == "negative"):
        positive = next(row for row in rows if row["logical_id"] == negative["positive_logical_id"])
        if negative["question"] != positive["question"] or negative["role"] != positive["role"]:
            raise RuntimeError("matched pair question or role drift")
    return rows, exclusions


def _cache_locks(old_cache: dict[str, Any], code_commit: str) -> dict[str, str]:
    return {**old_cache["locks"], "matched_panel": MATCHED_PANEL, "runner_commit": code_commit}


def matched_feature_cache(
    runtime: Any,
    event: dict[str, Any],
    scope: dict[str, Any],
    path: Path,
    locks: dict[str, str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached["locks"] != locks:
                raise RuntimeError("matched feature cache lock mismatch")
            return cached
        record = EditorRecord.from_dict(event["edit_record"])
        rows, exclusions = matched_panel_rows(scope, event)
        values, eqkeys = {}, []
        for row in rows:
            batch = runtime.build_question_batch(record, question=row["question"], image_path=Path(row["image_path"]))
            activation = runtime.extract_layer_input_features(batch, module_path=LAYER).cpu()
            prompt = activation[batch.key_token_index]
            visual = activation[batch.image_token_start:batch.image_token_end]
            if not len(visual):
                raise RuntimeError("matched panel realized an empty visual span")
            attention = batch.attention_mask if batch.attention_mask is not None else torch.ones(batch.inputs_embeds.shape[:2], dtype=torch.long)
            input_ids = batch.raw_input_ids[0].detach().cpu().tolist()
            eqkey = sha256_json({
                "image_tensor_sha256": batch.image_sha256,
                "target_free_prompt_tokens": input_ids,
                "attention_mask": attention[0].detach().cpu().tolist(),
                "assistant_boundary_index": batch.key_token_index,
                "image_token_span": [batch.image_token_start, batch.image_token_end],
                **locks,
            })
            values[row["logical_id"]] = {"row": row, "prompt": prompt, "visual": visual, "eqkey": eqkey, "prompt_tokens": input_ids}
            eqkeys.append(eqkey)
        if len(eqkeys) != len(set(eqkeys)):
            raise RuntimeError("matched panel EqKey collision crosses frozen examples")
        for negative in (value for value in values.values() if value["row"]["label"] == "negative"):
            positive = values[negative["row"]["positive_logical_id"]]
            if negative["prompt_tokens"] != positive["prompt_tokens"]:
                raise RuntimeError("matched image pair does not have bit-identical question tokens")
        cached = {
            "schema_version": "medtrace-matched-question-image-feature-cache-private-v1",
            "locks": locks, "values": values, "eqkeys": eqkeys, "exclusions": exclusions,
            "native_source_shared_across_roles": True,
        }
        temporary = path.with_suffix(".tmp")
        torch.save(cached, temporary); os.replace(temporary, path)
        return cached


def _feature_map(expert: AsymmetricCPExpert, cache: dict[str, Any]) -> dict[str, Any]:
    result = {}
    with torch.no_grad():
        for logical_id, value in cache["values"].items():
            result[logical_id] = verifier_features(expert, value["prompt"].to("cuda:0"), value["visual"].to("cuda:0"))
    return result


def _training_data(
    matched: dict[str, Any],
    original: dict[str, Any],
    features: dict[str, Any],
    original_features: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    selector_q = (lambda value: value.cp_prompt) if kind == "M2" else (lambda value: value.pre_cp_prompt)
    selector_v = (lambda value: value.cp_visual) if kind == "M2" else (lambda value: value.pre_cp_visual)
    matched_fit = sorted((value for value in matched["values"].values() if value["row"]["role"] == "fit"), key=lambda value: value["row"]["logical_id"])
    broad_fit = sorted((value for value in original["values"].values() if value["row"]["role"] == "fit" and value["row"]["fact_relation"] == BROAD_RELATION), key=lambda value: value["row"]["logical_id"])
    if not matched_fit or not broad_fit:
        raise RuntimeError("fit-only verifier supervision is incomplete")
    question_values = [*matched_fit, *broad_fit]
    matched_ids = {value["row"]["logical_id"] for value in matched_fit}
    question_features = torch.stack([
        selector_q(features[value["row"]["logical_id"]]) if value["row"]["logical_id"] in matched_ids else selector_q(original_features[value["row"]["logical_id"]])
        for value in question_values
    ])
    question_labels = torch.tensor([value["row"]["logical_id"] in matched_ids for value in question_values], dtype=torch.bool, device=question_features.device)
    question_groups = [value["row"].get("source_group") or hashlib.sha256(value["row"]["image_path"].encode()).hexdigest()[:16] for value in question_values]
    image_features = torch.stack([selector_v(features[value["row"]["logical_id"]]) for value in matched_fit])
    image_labels = torch.tensor([value["row"]["label"] == "positive" for value in matched_fit], dtype=torch.bool, device=image_features.device)
    image_groups = [value["row"]["source_group"] for value in matched_fit]
    indices = {value["row"]["logical_id"]: index for index, value in enumerate(matched_fit)}
    pairs = [(indices[value["row"]["positive_logical_id"]], indices[value["row"]["logical_id"]]) for value in matched_fit if value["row"]["label"] == "negative"]
    return {
        "question_features": question_features, "question_labels": question_labels, "question_groups": question_groups,
        "image_features": image_features, "image_labels": image_labels, "image_groups": image_groups, "matched_pairs": pairs,
    }


def _score_rows(
    expert: AsymmetricCPExpert,
    checkpoint: dict[str, Any],
    cache: dict[str, Any],
    features: dict[str, Any],
    verifiers: dict[str, LinearApplicabilityVerifier],
) -> dict[str, dict[str, list[float]]]:
    prompt_prototype = normalize_rows(checkpoint["prototypes"]["prompt_prototype"].to("cuda:0"))
    visual_prototype = normalize_rows(checkpoint["prototypes"]["visual_prototype"].to("cuda:0"))
    result = {}
    with torch.no_grad():
        for logical_id, value in cache["values"].items():
            feature = features[logical_id]
            prompt_score = float((feature.cp_prompt @ prompt_prototype).item())
            visual_score = float((feature.cp_visual @ visual_prototype).item())
            m2q, m2v = verifiers["M2"].logits(feature.cp_prompt[None], feature.cp_visual[None])
            m3q, m3v = verifiers["M3"].logits(feature.pre_cp_prompt[None], feature.pre_cp_visual[None])
            result[logical_id] = {
                CONDITIONS[0]: [0.5 * prompt_score + 0.5 * visual_score],
                CONDITIONS[1]: [prompt_score, visual_score],
                CONDITIONS[2]: [float(m2q.item()), float(m2v.item())],
                CONDITIONS[3]: [float(m3q.item()), float(m3v.item())],
            }
    return result


def _calibrate_panel(
    cache: dict[str, Any],
    scores: dict[str, dict[str, list[float]]],
    broad_cache: dict[str, Any],
    broad_scores: dict[str, dict[str, list[float]]],
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    positives = [value for value in cache["values"].values() if value["row"]["role"] == "calibration" and value["row"]["label"] == "positive"]
    hard = [value for value in cache["values"].values() if value["row"]["role"] == "calibration" and value["row"]["fact_relation"] == HARD_RELATION]
    broad = [value for value in broad_cache["values"].values() if value["row"]["role"] == "calibration" and value["row"]["fact_relation"] == BROAD_RELATION]
    if len(positives) != 4:
        raise RuntimeError(f"COVERAGE90 discreteness requires exactly four calibration positives, got {len(positives)}")
    result = {}
    for condition in conditions:
        result[condition] = calibrate_decisions(
            [tuple(scores[value["row"]["logical_id"]][condition]) for value in positives],
            [tuple(scores[value["row"]["logical_id"]][condition]) for value in hard],
            [tuple(broad_scores[value["row"]["logical_id"]][condition]) for value in broad],
        )
    return result


def _decision(score: Sequence[float], calibration: dict[str, Any]) -> bool:
    return all(value > threshold for value, threshold in zip(score, calibration["thresholds"], strict=True))


def _load_or_generate_base(
    runtime: Any,
    rows: list[dict[str, Any]],
    path: Path,
    binding: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["binding"] != binding:
                raise RuntimeError("matched base-output cache binding mismatch")
            return cached["outputs"]
        outputs = {row["logical_id"]: scope_generate(runtime, row, None) for row in rows}
        atomic_json(path, {"schema_version": "medtrace-matched-base-outputs-private-v1", "binding": binding, "outputs": outputs})
        return outputs


def _head_report(verifier: LinearApplicabilityVerifier, curve: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "medtrace-frozen-linear-verifier-private-v1",
        "state": {name: value.detach().cpu() for name, value in verifier.state_dict().items()},
        "curve": curve,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); os.replace(temporary, path)
    return {
        "parameter_count": sum(value.numel() for value in verifier.parameters()),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in verifier.parameters()),
        "checkpoint_bytes": path.stat().st_size,
        "curve": curve,
    }


def _old_calibration(condition_dir: Path) -> dict[str, Any]:
    value = json.loads((condition_dir / "calibration_step0800.json").read_text())
    return {
        "CONTINUITY_SAFETY_FIRST": {**value["operating_points"]["SAFETY_FIRST"], "thresholds": [value["operating_points"]["SAFETY_FIRST"]["threshold"]]},
        "CONTINUITY_COVERAGE_CONSTRAINED": {**value["operating_points"]["COVERAGE_CONSTRAINED"], "thresholds": [value["operating_points"]["COVERAGE_CONSTRAINED"]["threshold"]]},
        "stored_scores": {row["logical_id"]: row["score"] for row in value["scores"]},
        "binding": {
            "q_sha256": value["q_sha256"], "prototype_sha256": value["prototype_sha256"],
            "score_definition_sha256": value["score_definition_sha256"],
        },
    }


def _original_outputs(execution_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["row"]["logical_id"]: {"row": item["row"], "base": item["base"], "forced": item["forced"], "source": "REUSED_EXACT_C1_STEP800"}
        for item in execution_result["items"]
    }


def process_task(runtime: Any, args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    frozen = json.loads((args.old_run / "private/frozen_data.json").read_text())
    event = frozen["dev"][task["event_index"] - 1]
    scope = frozen["scopes"][task["record_id"]]
    if scope["status"] != "HARD_EVALUABLE":
        raise RuntimeError("main verifier queue contains a non-hard edit")
    old_cache = torch.load(args.old_run / f"private/features/e{task['event_index']:02d}.pt", map_location="cpu", weights_only=False)
    locks = _cache_locks(old_cache, args.expected_code_commit)
    feature_started = time.monotonic()
    matched_cache = matched_feature_cache(
        runtime, event, scope, args.run_root / f"private/features/e{task['event_index']:02d}.pt", locks,
    )
    feature_elapsed = time.monotonic() - feature_started
    task_dir = args.run_root / "private/tasks" / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    c1_dir = args.execution_run / "private/tasks" / f"P0_s{task['seed']}_e{task['event_index']:02d}" / "C1_R2_FIXED_Q_LONG_RECOVERY"
    checkpoint_path = c1_dir / "step0800.pt"
    e2e_path = c1_dir / "e2e_step0800.json"
    if not checkpoint_path.exists() or not e2e_path.exists():
        raise FileNotFoundError("frozen C1@800 artifact is incomplete")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)
    if checkpoint.get("condition") != "C1_R2_FIXED_Q_LONG_RECOVERY" or int(checkpoint.get("step", -1)) != 800:
        raise RuntimeError("executor is not the frozen C1@800 condition")
    expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"]); expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    expert_before = {name: value.detach().cpu().clone() for name, value in expert.state_dict().items()}
    expert_lock = {
        "checkpoint_sha256": sha256_file(checkpoint_path), "state_sha256": state_hash(expert_before),
        "q_sha256": tensor_sha256(expert.input_basis()),
        "p_rho_sha256": state_hash({name: expert_before[name] for name in ("u_out", "v_out", "rho")}),
        "layer": LAYER, "rank": 4, "seed": task["seed"], "event_index": task["event_index"],
    }
    original_features = _feature_map(expert, old_cache)
    matched_features = _feature_map(expert, matched_cache)
    verifiers, training_report = {}, {}
    training_started = time.monotonic()
    for kind in ("M2", "M3"):
        seed_everything(derive_seed(task["record_id"], base=task["seed"]) + (2 if kind == "M2" else 3))
        data = _training_data(matched_cache, old_cache, matched_features, original_features, kind)
        verifier, curve = train_verifier(**data)
        verifiers[kind] = verifier
        training_report[kind] = _head_report(verifier, curve, task_dir / f"{kind}_heads.pt") | {
            "question_examples": int(data["question_labels"].numel()),
            "question_positive": int(data["question_labels"].sum().item()),
            "image_examples": int(data["image_labels"].numel()),
            "image_positive": int(data["image_labels"].sum().item()),
            "matched_pair_count": len(data["matched_pairs"]),
            "input_width": int(data["question_features"].shape[-1]),
        }
    training_elapsed = time.monotonic() - training_started
    if any(not torch.equal(expert_before[name], value.detach().cpu()) for name, value in expert.state_dict().items()):
        raise RuntimeError("verifier training changed frozen C1 Q/P/rho")
    score_started = time.monotonic()
    original_scores = _score_rows(expert, checkpoint, old_cache, original_features, verifiers)
    matched_scores = _score_rows(expert, checkpoint, matched_cache, matched_features, verifiers)
    score_elapsed = time.monotonic() - score_started
    original_calibration = _calibrate_panel(old_cache, original_scores, old_cache, original_scores, CONDITIONS[1:])
    original_calibration[CONDITIONS[0]] = _old_calibration(c1_dir)
    for logical_id, stored in original_calibration[CONDITIONS[0]]["stored_scores"].items():
        if logical_id in original_scores and abs(stored - original_scores[logical_id][CONDITIONS[0]][0]) > 1e-5:
            raise RuntimeError("M0 continuity score differs from frozen C1 result")
    matched_calibration = _calibrate_panel(matched_cache, matched_scores, old_cache, original_scores)

    execution_result = json.loads(e2e_path.read_text())
    original_outputs = _original_outputs(execution_result)
    matched_evaluation = sorted(
        (value["row"] for value in matched_cache["values"].values() if value["row"]["role"] == "evaluation"),
        key=lambda row: row["logical_id"],
    )
    base_binding = {"event_index": task["event_index"], "matched_feature_locks": locks, "row_ids": [row["logical_id"] for row in matched_evaluation]}
    base_outputs = _load_or_generate_base(
        runtime, matched_evaluation, args.run_root / f"private/base_matched/e{task['event_index']:02d}.json", base_binding,
    )
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert); hook.attach()
    try:
        forced_outputs = {row["logical_id"]: scope_generate(runtime, row, hook) for row in matched_evaluation}
    finally:
        hook.detach()
    matched_outputs = {
        row["logical_id"]: {"row": row, "base": base_outputs[row["logical_id"]], "forced": forced_outputs[row["logical_id"]], "source": "NEW_FROZEN_TWO_PATH_OUTPUTS"}
        for row in matched_evaluation
    }

    replays = []
    if task.get("replay_designated"):
        for condition in CONDITIONS[1:]:
            point = matched_calibration[condition]["PRIMARY_SAFETY_FIRST"]
            candidates = [(row, _decision(matched_scores[row["logical_id"]][condition], point)) for row in matched_evaluation]
            for desired in (True, False):
                selected = next((row for row, on in candidates if on is desired), None)
                if selected is None:
                    raise RuntimeError(f"{condition} lacks a designated {'ON' if desired else 'OFF'} replay")
                if desired:
                    replay_hook = MedTraceLayerHook(runtime.get_module(LAYER), expert); replay_hook.attach()
                    try:
                        actual = scope_generate(runtime, selected, replay_hook)
                    finally:
                        replay_hook.detach()
                    expected = matched_outputs[selected["logical_id"]]["forced"]
                else:
                    actual = scope_generate(runtime, selected, None)
                    expected = matched_outputs[selected["logical_id"]]["base"]
                exact = actual["raw_token_ids"] == expected["raw_token_ids"] and actual["raw_answer"] == expected["raw_answer"]
                if not exact:
                    raise RuntimeError("derived gate path and actual replay differ")
                replays.append({"condition": condition, "decision": "ON" if desired else "OFF", "logical_id": selected["logical_id"], "exact_replay": True})
    guard = runtime.base_guard.verify() if runtime.base_guard else None
    if not guard or not guard["unchanged"]:
        raise RuntimeError("frozen-verifier base guard failed")
    result = {
        "schema_version": "medtrace-frozen-expert-visual-verifier-task-private-v1", "status": "COMPLETE",
        "task": task, "executor_lock": expert_lock, "matched_feature_locks": locks,
        "training": training_report,
        "calibration": {"original": original_calibration, "matched": matched_calibration},
        "scores": {"original": original_scores, "matched": matched_scores},
        "outputs": {"original": original_outputs, "matched": matched_outputs},
        "replays": replays, "base_guard": guard,
        "timing": {"feature_seconds": feature_elapsed, "training_seconds": training_elapsed, "score_seconds": score_elapsed, "total_seconds": time.monotonic() - started},
    }
    atomic_json(task_dir / "result_private.json", result)
    return result


def budget_exhausted(run_root: Path, queue: TaskQueue) -> bool:
    start = json.loads((run_root / "private/CAMPAIGN_START.json").read_text())["epoch"]
    wall = time.time() - start
    used = sum(row.get("elapsed_seconds", 0.0) for row in queue.snapshot()["tasks"] if row["status"] in {"COMPLETE", "FAILED"})
    return wall >= 12 * 3600 or used >= 24 * 3600 or (run_root / "STOP").exists() or STOP_REQUESTED


def worker(args: argparse.Namespace) -> None:
    physical, _ = verify_gpu()
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != args.expected_code_commit:
        raise RuntimeError("frozen-verifier worker code commit mismatch")
    queue = TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root)
    runtime_config = json.loads((args.old_run / "private/CAMPAIGN_RUNTIME_CONFIG.json").read_text())
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(runtime_config["cpu_gate"])))
    completed_here = 0
    try:
        with Telemetry(args.run_root / "private/GPU_TELEMETRY.jsonl", physical, f"gpu{physical}") as telemetry:
            while True:
                if budget_exhausted(args.run_root, queue):
                    queue.cancel_pending(); break
                task = queue.claim(f"gpu{physical}")
                if task is None:
                    if all(row["status"] in {"COMPLETE", "FAILED", "CANCELLED_BY_BUDGET"} for row in queue.snapshot()["tasks"]):
                        break
                    time.sleep(5); continue
                telemetry.task_id = task["task_id"]
                started = time.monotonic()
                try:
                    value = process_task(runtime, args, task)
                    queue.update(task["task_id"], "COMPLETE", result_status=value["status"], elapsed_seconds=time.monotonic() - started, finished_at=time.time())
                    completed_here += 1
                except torch.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    status = "PENDING" if task["attempts"] < 2 else "FAILED"
                    queue.update(task["task_id"], status, last_error=f"OOM: {error}", elapsed_seconds=time.monotonic() - started)
                except Exception as error:
                    queue.update(task["task_id"], "FAILED", last_error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
                    append_jsonl(args.run_root / "private/WORKER_ERRORS.jsonl", {"task_id": task["task_id"], "worker": f"gpu{physical}", "error": f"{type(error).__name__}: {error}", "at": time.time()})
                finally:
                    telemetry.task_id = None; torch.cuda.empty_cache()
                if args.max_tasks and completed_here >= args.max_tasks:
                    break
    finally:
        del runtime; torch.cuda.empty_cache()


def prepare(args: argparse.Namespace) -> None:
    if args.run_root.exists() or args.public_dir.exists():
        raise FileExistsError("frozen-verifier run or report directory already exists")
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", args.base_commit, actual]).returncode:
        raise RuntimeError("runner is not descended from the frozen starting commit")
    frozen = json.loads((args.old_run / "private/frozen_data.json").read_text())
    hard = [
        (index, event) for index, event in enumerate(frozen["dev"], 1)
        if frozen["scopes"][event["edit_record"]["record_id"]]["status"] == "HARD_EVALUABLE"
    ]
    if len(hard) != 7:
        raise RuntimeError("expected exactly seven frozen hard-evaluable edits")
    checkpoint_count = len(list(args.execution_run.glob("private/tasks/P0_s*_e*/C1_R2_FIXED_Q_LONG_RECOVERY/step0800.pt")))
    e2e_count = len(list(args.execution_run.glob("private/tasks/P0_s*_e*/C1_R2_FIXED_Q_LONG_RECOVERY/e2e_step0800.json")))
    if checkpoint_count != 36 or e2e_count != 36:
        raise RuntimeError("execution-preservation source run is incomplete")
    tasks = []
    first_event = hard[0][0]
    for seed_index, seed in enumerate(SEEDS):
        for cohort_index, (event_index, event) in enumerate(hard):
            record_id = event["edit_record"]["record_id"]
            tasks.append({
                "task_id": f"VERIFY_s{seed}_e{event_index:02d}", "kind": "VERIFY", "seed": seed,
                "event_index": event_index, "record_id": record_id,
                "priority": seed_index * 100 + cohort_index, "status": "PENDING", "attempts": 0,
                "replay_designated": seed == SEEDS[0] and event_index == first_event,
            })
    args.run_root.mkdir(parents=True); (args.run_root / "private").mkdir()
    atomic_json(args.run_root / "private/TASK_QUEUE.json", {"schema_version": "medtrace-frozen-verifier-queue-v1", "tasks": tasks})
    atomic_json(args.run_root / "private/CAMPAIGN_CONFIG.json", {
        "schema_version": "medtrace-frozen-verifier-config-v1", "base_commit": args.base_commit,
        "code_commit": actual, "old_run": str(args.old_run), "execution_run": str(args.execution_run),
        "conditions": list(CONDITIONS), "seeds": list(SEEDS), "hard_edits": 7, "tasks": 21,
        "wall_hours": 12, "gpu_hours": 24, "gpu_uuids": {key: GPU_UUIDS[key] for key in ("2", "3")},
        "gpu1_forbidden": True, "matched_panel": MATCHED_PANEL,
    })
    args.public_dir.mkdir(parents=True)
    atomic_json(args.public_dir / "EXPERIMENT_CONFIG.json", {
        "schema_version": "medtrace-frozen-verifier-public-config-v1", "base_commit": args.base_commit,
        "code_commit": actual, "executor": "C1_R2_FIXED_Q_LONG_RECOVERY@additional_step800",
        "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "cohort": {"hard_evaluable_edits": 7, "repeated_seed_runs": 3, "independent_fact_claim": False},
        "verifier": {"optimizer": "Adam", "lr": 0.01, "weight_decay": 0, "steps": 800, "gradient_clip": 1.0, "l2": 0.001},
        "operating_points": list(POINTS), "coverage90_discrete_requirement": "4/4 calibration positives",
        "devices": {"physical": [2, 3], "gpu1_forbidden": True}, "private_artifacts_withheld": True,
    })
    atomic_text(args.public_dir / "CAMPAIGN_PROTOCOL.md", (
        "# MedTRACE frozen-expert visual-verifier campaign\n\n"
        "Status: `READY_NOT_STARTED`\n\n"
        "The C1 additional-step-800 CP executor, Q, P, rho, layer, rank and generation configuration are frozen. "
        "M0/M1/M2/M3 change only ON/OFF routing. M2 and M3 are supervised diagnostic verifiers with additional parameters. "
        "The main cohort is seven hard-evaluable edits repeated under the three original seeds; seeds are not independent facts. "
        "The primary panel uses source-supported identical-question native/hard-image pairs. This is viewed development evidence, not blind or clinical validation.\n"
    ))
    atomic_text(args.public_dir / "TASK_LEDGER.jsonl", "".join(json.dumps({
        "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"],
        "opaque_edit": hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], "status": "PENDING",
    }, sort_keys=True) + "\n" for task in tasks))


def recover(args: argparse.Namespace) -> None:
    TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root).recover()


def _prior_judge_map(execution_run: Path) -> dict[tuple[str, str, str], bool]:
    sidecar_path = execution_run / "private/judge/JUDGE_SIDECAR_PRIVATE.json"
    output_path = execution_run / "private/judge/JUDGE_OUTPUT_PRIVATE.jsonl"
    sidecar = json.loads(sidecar_path.read_text())
    verdicts = {opaque: bool(value["reused_verdict"]) for opaque, value in sidecar["tuples"].items() if value["reused_verdict"] is not None}
    verdicts.update({row["opaque_query_id"]: bool(row["is_correct"]) for row in read_jsonl(output_path) if row.get("parse_valid")})
    return {
        (value["question"], value["reference"], value["raw"]): verdicts[opaque]
        for opaque, value in sidecar["tuples"].items() if opaque in verdicts
    }


def _task_results(run_root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((run_root / "private/tasks").glob("VERIFY_*/result_private.json"))]


def prepare_judge(args: argparse.Namespace) -> None:
    if args.packet.exists() or args.sidecar.exists():
        raise FileExistsError("frozen-verifier Judge packet already exists")
    results = _task_results(args.run_root)
    if len(results) != 21:
        raise RuntimeError(f"expected 21 complete verifier tasks, got {len(results)}")
    prior = _prior_judge_map(args.execution_run)
    tuples, packet, uses = {}, {}, []
    for result in results:
        for panel, outputs in result["outputs"].items():
            for logical_id, item in outputs.items():
                row = item["row"]
                for path in ("base", "forced"):
                    raw = item[path]["raw_answer"]
                    key = (row["question"], row["reference"], raw)
                    opaque = sha256_json((*key, "medtrace-frozen-verifier-semantic-v1"))
                    tuples[opaque] = {"question": key[0], "reference": key[1], "raw": key[2], "reused_verdict": prior.get(key)}
                    uses.append({"opaque_query_id": opaque, "task_id": result["task"]["task_id"], "panel": panel, "logical_id": logical_id, "path": path})
                    if key not in prior:
                        packet[opaque] = {"opaque_query_id": opaque, "question": key[0], "gold_answer": key[1], "raw_base_answer": key[2], "adjudication_pass": 1}
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.packet, "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet.values()))
    atomic_json(args.sidecar, {
        "schema_version": "medtrace-frozen-verifier-judge-sidecar-private-v1", "tuples": tuples, "uses": uses,
        "reused_count": sum(value["reused_verdict"] is not None for value in tuples.values()), "new_count": len(packet),
    })


def _verdicts(sidecar_path: Path, judge_output: Path) -> dict[str, bool]:
    sidecar = json.loads(sidecar_path.read_text())
    values = {opaque: bool(row["reused_verdict"]) for opaque, row in sidecar["tuples"].items() if row["reused_verdict"] is not None}
    for row in read_jsonl(judge_output):
        if not row.get("parse_valid"):
            raise RuntimeError("Judge output contains a parse failure")
        values[row["opaque_query_id"]] = bool(row["is_correct"])
    if set(values) != set(sidecar["tuples"]):
        raise RuntimeError("Judge closure does not cover every tuple")
    return values


def _semantic(verdicts: dict[str, bool], row: dict[str, Any], raw: str) -> bool:
    return verdicts[sha256_json((row["question"], row["reference"], raw, "medtrace-frozen-verifier-semantic-v1"))]


def _category(panel: str, row: dict[str, Any]) -> str:
    if panel == "matched":
        return "matched_evaluation_positive" if row["label"] == "positive" else "matched_hard_image"
    return evaluation_category(row)


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n, "on_num": sum(row["on"] for row in rows), "on_rate": sum(row["on"] for row in rows) / n if n else None,
        "base_exact_num": sum(row["base_exact"] for row in rows), "forced_exact_num": sum(row["forced_exact"] for row in rows),
        "gated_exact_num": sum(row["gated_exact"] for row in rows), "base_semantic_num": sum(row["base_semantic"] for row in rows),
        "forced_semantic_num": sum(row["forced_semantic"] for row in rows), "gated_semantic_num": sum(row["gated_semantic"] for row in rows),
        "joint_on_exact_rate": sum(row["on"] and row["forced_exact"] for row in rows) / n if n else None,
        "joint_on_semantic_rate": sum(row["on"] and row["forced_semantic"] for row in rows) / n if n else None,
        "base_correct_damage_num": sum(row["base_semantic"] and not row["gated_semantic"] for row in rows),
        "base_correct_den": sum(row["base_semantic"] for row in rows),
        "off_parity_num": sum((not row["on"]) and row["gated_raw_source"] == "base" for row in rows),
        "off_num": sum(not row["on"] for row in rows),
    }


def _hierarchical_by_edit(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    by_edit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_edit[row["event_index"]].append(row)
    result = {}
    for edit, edit_rows in by_edit.items():
        seed_values = []
        for seed in sorted({row["seed"] for row in edit_rows}):
            seed_rows = [row for row in edit_rows if row["seed"] == seed]
            groups = []
            for source_group in sorted({row["source_group"] for row in seed_rows}):
                values = [float(row[key]) for row in seed_rows if row["source_group"] == source_group]
                groups.append(mean(values))
            seed_values.append(mean(groups))
        result[edit] = mean(seed_values)
    return result


def _macro(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _hierarchical_by_edit(rows, key)
    return mean(values.values()) if values else None


def _paired_ci(candidate: dict[int, float], control: dict[int, float], draws: int = 2000) -> tuple[float | None, float | None]:
    edits = sorted(set(candidate) & set(control))
    if not edits: return None, None
    rng = random.Random(20260907); values = []
    for _ in range(draws):
        sample = [rng.choice(edits) for _ in edits]
        values.append(mean(candidate[edit] - control[edit] for edit in sample))
    values.sort(); return values[int(0.025 * draws)], values[int(0.975 * draws)]


def finalize(args: argparse.Namespace) -> None:
    results = _task_results(args.run_root)
    if len(results) != 21:
        raise RuntimeError(f"expected 21 task results, got {len(results)}")
    verdicts = _verdicts(args.sidecar, args.judge_output)
    detailed = []
    for result in results:
        task = result["task"]
        opaque_edit = hashlib.sha256(task["record_id"].encode()).hexdigest()[:16]
        for panel, outputs in result["outputs"].items():
            for logical_id, item in outputs.items():
                row = item["row"]
                base_raw, forced_raw = item["base"]["raw_answer"], item["forced"]["raw_answer"]
                base_exact = normalize_medical_answer(base_raw) == normalize_medical_answer(row["reference"])
                forced_exact = normalize_medical_answer(forced_raw) == normalize_medical_answer(row["reference"])
                base_semantic = _semantic(verdicts, row, base_raw)
                forced_semantic = _semantic(verdicts, row, forced_raw)
                for condition in CONDITIONS:
                    calibration = result["calibration"][panel][condition]
                    point_names = tuple(name for name in calibration if name.startswith("CONTINUITY_")) if panel == "original" and condition == CONDITIONS[0] else POINTS
                    for point in point_names:
                        on = _decision(result["scores"][panel][logical_id][condition], calibration[point])
                        detailed.append({
                            "seed": task["seed"], "event_index": task["event_index"], "opaque_edit": opaque_edit,
                            "panel": panel, "condition": condition, "operating_point": point,
                            "logical_id": logical_id, "category": _category(panel, row),
                            "source_group": row.get("source_group") or hashlib.sha256(row["image_path"].encode()).hexdigest()[:16],
                            "on": on, "base_exact": base_exact, "forced_exact": forced_exact,
                            "gated_exact": forced_exact if on else base_exact,
                            "base_semantic": base_semantic, "forced_semantic": forced_semantic,
                            "gated_semantic": forced_semantic if on else base_semantic,
                            "gated_raw_source": "forced" if on else "base",
                            "scores": result["scores"][panel][logical_id][condition],
                        })
    atomic_json(args.run_root / "private/FINAL_DETAILED_RESULTS.json", detailed)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        key = tuple(row[name] for name in ("seed", "event_index", "opaque_edit", "panel", "condition", "operating_point", "category"))
        grouped[key].append(row)
    columns = ["seed", "event_index", "opaque_edit", "panel", "condition", "operating_point", "category", *list(_metric([]))]
    with (args.public_dir / "RESULTS_BY_EDIT_SEED.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n"); writer.writeheader()
        for key, rows in sorted(grouped.items()):
            writer.writerow(dict(zip(columns[:7], key, strict=True)) | _metric(rows))

    matched_main = [row for row in detailed if row["panel"] == "matched" and row["operating_point"] == "PRIMARY_SAFETY_FIRST"]
    signal_rows = {}
    negative_categories = {"matched_hard_image", HARD_RELATION, BROAD_RELATION, "same_image_other_source_fact"}
    for condition in CONDITIONS:
        values = [row for row in matched_main if row["condition"] == condition]
        original_point = "CONTINUITY_SAFETY_FIRST" if condition == CONDITIONS[0] else "PRIMARY_SAFETY_FIRST"
        original = [row for row in detailed if row["panel"] == "original" and row["condition"] == condition and row["operating_point"] == original_point]
        hard = [row for row in values if row["category"] == "matched_hard_image"]
        positive = [row for row in values if row["category"] == "matched_evaluation_positive"]
        t1g = [row for row in original if row["category"] == "formal_T1G"]
        t2g = [row for row in original if row["category"] == "formal_T2G"]
        negative = [row for row in [*values, *original] if row["category"] in negative_categories and row["base_semantic"]]
        signal_rows[condition] = {
            "hard_edit_macro_fpr": _macro(hard, "on"), "hard_pooled": _metric(hard),
            "evaluation_positive_joint": _macro(positive, "gated_semantic"), "evaluation_positive_pooled": _metric(positive),
            "t1g_joint": _macro(t1g, "gated_semantic"), "t2g_joint": _macro(t2g, "gated_semantic"),
            "base_correct_negative_damage_rate": (sum(not row["gated_semantic"] for row in negative) / len(negative)) if negative else None,
            "base_correct_negative_damage_num": sum(not row["gated_semantic"] for row in negative), "base_correct_negative_den": len(negative),
            "hard_by_edit": _hierarchical_by_edit(hard, "on"),
            "positive_by_edit": _hierarchical_by_edit(positive, "gated_semantic"),
            "t1g_by_edit": _hierarchical_by_edit(t1g, "gated_semantic"), "t2g_by_edit": _hierarchical_by_edit(t2g, "gated_semantic"),
        }
    control = signal_rows[CONDITIONS[0]]
    passing = []
    for condition in CONDITIONS[1:]:
        value = signal_rows[condition]
        checks = {
            "hard_fpr_drop_at_least_0_10": value["hard_edit_macro_fpr"] <= control["hard_edit_macro_fpr"] - 0.10,
            "evaluation_positive_drop_at_most_0_02": value["evaluation_positive_joint"] >= control["evaluation_positive_joint"] - 0.02,
            "t1g_drop_at_most_0_02": value["t1g_joint"] >= control["t1g_joint"] - 0.02,
            "t2g_drop_at_most_0_02": value["t2g_joint"] >= control["t2g_joint"] - 0.02,
            "base_correct_damage_not_increased": value["base_correct_negative_damage_rate"] <= control["base_correct_negative_damage_rate"],
        }
        value["development_signal_checks"] = checks
        value["paired_hard_delta_95ci"] = _paired_ci(value["hard_by_edit"], control["hard_by_edit"])
        value["paired_positive_delta_95ci"] = _paired_ci(value["positive_by_edit"], control["positive_by_edit"])
        if all(checks.values()): passing.append(condition)
    selected = passing[0] if passing else None
    router_effect = "ROUTER_DEVELOPMENT_SIGNAL_MET" if selected else "ROUTER_DEVELOPMENT_SIGNAL_NOT_MET"

    # Fit/calibration/evaluation separability uses frozen calibration thresholds only.
    separability = []
    for result in results:
        original_cache = torch.load(args.old_run / f"private/features/e{result['task']['event_index']:02d}.pt", map_location="cpu", weights_only=False)
        for variant, condition in (("M2", CONDITIONS[2]), ("M3", CONDITIONS[3])):
            threshold_q, threshold_v = result["calibration"]["matched"][condition]["PRIMARY_SAFETY_FIRST"]["thresholds"]
            for role in ("fit", "calibration", "evaluation"):
                matched_cache = torch.load(args.run_root / f"private/features/e{result['task']['event_index']:02d}.pt", map_location="cpu", weights_only=False)
                matched_ids = [logical_id for logical_id, value in matched_cache["values"].items() if value["row"]["role"] == role]
                broad_ids = [logical_id for logical_id, value in original_cache["values"].items() if value["row"]["role"] == role and value["row"]["fact_relation"] == BROAD_RELATION]
                q_correct = sum(result["scores"]["matched"][logical_id][condition][0] > threshold_q for logical_id in matched_ids)
                q_correct += sum(result["scores"]["original"][logical_id][condition][0] <= threshold_q for logical_id in broad_ids)
                image_correct = sum((result["scores"]["matched"][logical_id][condition][1] > threshold_v) == (matched_cache["values"][logical_id]["row"]["label"] == "positive") for logical_id in matched_ids)
                separability.append({
                    "variant": variant, "seed": result["task"]["seed"], "event_index": result["task"]["event_index"], "role": role,
                    "question_accuracy": q_correct / (len(matched_ids) + len(broad_ids)), "image_accuracy": image_correct / len(matched_ids),
                })
    separability_summary = {}
    for variant in ("M2", "M3"):
        separability_summary[variant] = {
            role: {
                "question_accuracy": mean(row["question_accuracy"] for row in separability if row["variant"] == variant and row["role"] == role),
                "image_accuracy": mean(row["image_accuracy"] for row in separability if row["variant"] == variant and row["role"] == role),
            } for role in ("fit", "calibration", "evaluation")
        }
        separability_summary[variant]["image_fit_to_eval_gap"] = separability_summary[variant]["fit"]["image_accuracy"] - separability_summary[variant]["evaluation"]["image_accuracy"]

    provenance = []
    for event_index in sorted({result["task"]["event_index"] for result in results}):
        cache = torch.load(args.run_root / f"private/features/e{event_index:02d}.pt", map_location="cpu", weights_only=False)
        rows = [value["row"] for value in cache["values"].values()]
        provenance.append({
            "event_index": event_index, "opaque_edit": next(result["task"] for result in results if result["task"]["event_index"] == event_index)["record_id"],
            "role_counts": {role: {"positive": sum(row["role"] == role and row["label"] == "positive" for row in rows), "hard_negative": sum(row["role"] == role and row["label"] == "negative" for row in rows)} for role in ("fit", "calibration", "evaluation")},
            "excluded_pair_count": len(cache["exclusions"]), "question_token_identity_checked": True,
            "native_source_shared_across_roles": True,
        })
        provenance[-1]["opaque_edit"] = hashlib.sha256(provenance[-1]["opaque_edit"].encode()).hexdigest()[:16]
    atomic_json(args.public_dir / "MATCHED_PANEL_PROVENANCE_AND_COUNTS.json", {
        "schema_version": "medtrace-matched-panel-public-v1", "panel": MATCHED_PANEL, "edits": provenance,
        "hard_facts": 7, "seed_repetitions": 3, "seed_is_independent_fact": False,
        "source_questions_and_images_reused_without_gold_changes": True, "private_inputs_withheld": True,
    })
    atomic_json(args.public_dir / "C1_EXECUTOR_IDENTITY_LOCK.json", {
        "schema_version": "medtrace-c1-executor-lock-public-v1", "status": "EXECUTION_FROZEN_SHARED",
        "executor_count": len(results), "all_tasks_unchanged_after_verifier_training": True,
        "executors": [{"seed": result["task"]["seed"], "event_index": result["task"]["event_index"], **result["executor_lock"]} for result in results],
        "forced_output_shared_by_all_route_conditions": True,
    })
    calibration_public = []
    for result in results:
        for panel, conditions in result["calibration"].items():
            for condition, points in conditions.items():
                for point, value in points.items():
                    if point in {"stored_scores", "binding", "tie_break"}: continue
                    calibration_public.append({
                        "seed": result["task"]["seed"], "event_index": result["task"]["event_index"], "panel": panel,
                        "condition": condition, "operating_point": point,
                        **{key: value.get(key) for key in ("thresholds", "positive_tpr", "hard_fpr", "broad_fpr")},
                    })
    atomic_json(args.public_dir / "ROUTER_CALIBRATION.json", {
        "schema_version": "medtrace-frozen-verifier-calibration-public-v1", "selection_source": "calibration_only",
        "coverage90_discrete_requirement": "4/4", "rows": calibration_public, "raw_scores_withheld": True,
    })

    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{value:.1%}"

    report_lines = ["# Frozen-C1 router paired E2E report", "", "Primary panel: MATCHED_QUESTION_IMAGE_PANEL_V1; operating point: PRIMARY_SAFETY_FIRST.", ""]
    for condition in CONDITIONS:
        value = signal_rows[condition]
        report_lines.append(
            f"- `{condition}`: matched positive joint semantic {pct(value['evaluation_positive_joint'])}; "
            f"hard edit-macro FPR {pct(value['hard_edit_macro_fpr'])} "
            f"({value['hard_pooled']['on_num']}/{value['hard_pooled']['n']} pooled); "
            f"T1G joint {pct(value['t1g_joint'])}; T2G joint {pct(value['t2g_joint'])}; "
            f"base-correct negative damage {value['base_correct_negative_damage_num']}/{value['base_correct_negative_den']}."
        )
    report_lines.extend(["", f"Router status: `{router_effect}`.", f"Simplest preregistered passing condition: `{selected or 'NONE'}`.", "", "Seeds repeat the same seven facts; this is viewed DEV evidence, not blind or clinical validation."])
    atomic_text(args.public_dir / "ROUTER_PAIRED_E2E_REPORT.md", "\n".join(report_lines) + "\n")

    parameter_lines = ["# Router training and parameter report", "", "C1 Q/P/rho and the backbone were frozen for every task.", ""]
    for variant in ("M2", "M3"):
        task_reports = [result["training"][variant] for result in results]
        parameter_lines.append(
            f"- {variant}: {task_reports[0]['parameter_count']} parameters per edit/seed, "
            f"{task_reports[0]['parameter_bytes']} parameter bytes; mean training time "
            f"{mean(result['timing']['training_seconds'] / 2 for result in results):.3f}s per verifier."
        )
    parameter_lines.extend(["", f"M2/M3 separability: `{json.dumps(separability_summary, sort_keys=True)}`", "", "M2 and M3 are supervised diagnostic controls with extra parameters; neither is an intrinsic parameter-free router."])
    atomic_text(args.public_dir / "ROUTER_TRAINING_AND_PARAMETER_REPORT.md", "\n".join(parameter_lines) + "\n")

    trade_lines = ["# Generality and safety trade-off", "", "Pre-registered comparisons against M0 on the same matched panel and PRIMARY operating point:", ""]
    for condition in CONDITIONS[1:]:
        value = signal_rows[condition]
        trade_lines.append(f"- `{condition}`: " + "; ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in value["development_signal_checks"].items()) + f"; paired hard delta 95% CI={value['paired_hard_delta_95ci']}.")
    trade_lines.extend(["", f"Decision: `{router_effect}`; simplest passing condition: `{selected or 'NONE'}`."])
    atomic_text(args.public_dir / "GENERALITY_SAFETY_TRADEOFF.md", "\n".join(trade_lines) + "\n")

    # Collapse hard errors to image-group units and disclose cross-seed recurrence.
    hard_lines = ["# Fact-group error transitions", "", "Rows are matched evaluation hard-image groups; rewrite-level values are averaged before seed/edit aggregation.", "", "|edit|image group|seed|M0 on|M1 on|M2 on|M3 on|prompt score|visual score|M2 q/v|M3 q/v|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    hard_detail = [row for row in matched_main if row["category"] == "matched_hard_image"]
    keys = sorted({(row["event_index"], row["opaque_edit"], row["source_group"], row["seed"]) for row in hard_detail})
    repeat_counts: dict[tuple[int, str], int] = defaultdict(int)
    for event_index, opaque_edit, source_group, seed in keys:
        cells = {}
        for condition in CONDITIONS:
            values = [row for row in hard_detail if row["event_index"] == event_index and row["source_group"] == source_group and row["seed"] == seed and row["condition"] == condition]
            cells[condition] = mean(float(row["on"]) for row in values)
        m1_values = [row for row in hard_detail if row["event_index"] == event_index and row["source_group"] == source_group and row["seed"] == seed and row["condition"] == CONDITIONS[1]]
        m2_values = [row for row in hard_detail if row["event_index"] == event_index and row["source_group"] == source_group and row["seed"] == seed and row["condition"] == CONDITIONS[2]]
        m3_values = [row for row in hard_detail if row["event_index"] == event_index and row["source_group"] == source_group and row["seed"] == seed and row["condition"] == CONDITIONS[3]]
        if cells[CONDITIONS[0]] > 0: repeat_counts[(event_index, source_group)] += 1
        hard_lines.append(f"|{opaque_edit}|{source_group}|{seed}|{cells[CONDITIONS[0]]:.2f}|{cells[CONDITIONS[1]]:.2f}|{cells[CONDITIONS[2]]:.2f}|{cells[CONDITIONS[3]]:.2f}|{mean(row['scores'][0] for row in m1_values):.4f}|{mean(row['scores'][1] for row in m1_values):.4f}|{mean(row['scores'][0] for row in m2_values):.3f}/{mean(row['scores'][1] for row in m2_values):.3f}|{mean(row['scores'][0] for row in m3_values):.3f}/{mean(row['scores'][1] for row in m3_values):.3f}|")
    hard_lines.extend(["", "Cross-seed M0 recurrence counts:", ""] + [f"- edit {event_index}, image group {group}: {count}/3 seeds" for (event_index, group), count in sorted(repeat_counts.items())])
    atomic_text(args.public_dir / "FACT_GROUP_ERROR_TRANSITIONS.md", "\n".join(hard_lines) + "\n")

    sidecar = json.loads(args.sidecar.read_text())
    atomic_json(args.public_dir / "JUDGE_AND_RAW_CLOSURE.json", {
        "schema_version": "medtrace-frozen-verifier-judge-closure-public-v1", "status": "JUDGE_COMPLETE",
        "unique_tuples": len(sidecar["tuples"]), "reused_exact_tuples": sidecar["reused_count"], "new_judge_tuples": sidecar["new_count"],
        "all_parse_valid": True, "two_path_outputs": "DERIVED_FROM_FROZEN_TWO_PATH_OUTPUTS",
        "actual_replays": sum(len(result["replays"]) for result in results), "private_mapping_withheld": True,
    })
    queue = json.loads((args.run_root / "private/TASK_QUEUE.json").read_text())["tasks"]
    gpu_seconds = sum(task.get("elapsed_seconds", 0.0) for task in queue if task["status"] in {"COMPLETE", "FAILED"})
    completion = {
        "schema_version": "medtrace-frozen-verifier-completion-public-v1",
        "EXECUTION": "EXECUTION_FROZEN_SHARED", "DATA_SCOPE": "MATCHED_PANEL_COMPLETE" if len(results) == 21 else "PARTIAL",
        "ROUTER_EFFECT": router_effect, "PUBLICATION": "RETRY_REQUIRED", "selected_simplest_condition": selected,
        "task_complete": len(results), "task_expected": 21, "gpu_hours": gpu_seconds / 3600,
        "wall_limit_hours": 12, "gpu_limit_hours": 24, "gpu1_used": False,
    }
    atomic_json(args.public_dir / "GPU_USAGE_AND_COMPLETION.json", completion)
    review_lines = ["# GPT Pro review: frozen C1 expert and visual verifier", "", *report_lines[2:], "", *trade_lines[2:], "", "Review whether matched-question pairing removes text compensation, whether M2 retains usable CP4 information relative to M3, and whether any passing signal survives edit-cluster rather than rewrite-level aggregation. M2/M3 are additional supervised diagnostic heads, not intrinsic routing or clinical probabilities."]
    atomic_text(args.public_dir / "GPT_PRO_REVIEW.md", "\n".join(review_lines) + "\n")
    atomic_text(args.public_dir / "TASK_LEDGER.jsonl", "".join(json.dumps({
        "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"],
        "opaque_edit": hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], "status": task["status"],
    }, sort_keys=True) + "\n" for task in queue))


def signal_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__); sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("run-root", "old-run", "execution-run", "public-dir"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--base-commit", required=True); prepare_parser.set_defaults(func=prepare)
    recover_parser = sub.add_parser("recover"); recover_parser.add_argument("--run-root", type=Path, required=True); recover_parser.set_defaults(func=recover)
    worker_parser = sub.add_parser("worker")
    for name in ("run-root", "old-run", "execution-run"):
        worker_parser.add_argument(f"--{name}", type=Path, required=True)
    worker_parser.add_argument("--expected-code-commit", required=True); worker_parser.set_defaults(func=worker)
    worker_parser.add_argument("--max-tasks", type=int, default=0)
    judge = sub.add_parser("prepare-judge")
    for name in ("run-root", "execution-run", "packet", "sidecar"):
        judge.add_argument(f"--{name}", type=Path, required=True)
    judge.set_defaults(func=prepare_judge)
    final = sub.add_parser("finalize")
    for name in ("run-root", "old-run", "judge-output", "sidecar", "public-dir"):
        final.add_argument(f"--{name}", type=Path, required=True)
    final.set_defaults(func=finalize)
    return value


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_stop)
    options = parser().parse_args(); options.func(options)
