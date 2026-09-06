#!/usr/bin/env python3
"""Run the bounded MedTRACE C1/C2/C3 execution-preservation campaign."""

from __future__ import annotations

import argparse
import csv
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
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
from scripts.medtrace.repair_route_calibration import evaluation_category, read_jsonl, score_state  # noqa: E402
from scripts.medtrace.run_dev16 import atomic_json, derive_seed, sha256_file, sha256_json  # noqa: E402
from scripts.medtrace.run_longrun_campaign import (  # noqa: E402
    GPU_UUIDS,
    OPERATING_POINTS,
    REPRESENTATIONS,
    SCORE_DEFINITION_SHA256,
    TaskQueue,
    Telemetry,
    append_jsonl,
    atomic_text,
    build_fit_prototypes,
    calibrate_operating_points,
    ordered_features,
    route_score_one,
    score_with_frozen_prototypes,
    scope_rows,
    state_hash,
)
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402
from scripts.medtrace.run_scope_pilot import generate as scope_generate  # noqa: E402


CONDITIONS = (
    "C1_R2_FIXED_Q_LONG_RECOVERY",
    "C2_R2_SHARED_Q_JOINT",
    "C3_R2_SHARED_Q_EXECUTION_ANCHOR",
)
MILESTONES = (0, 80, 320, 800)
R2 = REPRESENTATIONS[2]
STOP_REQUESTED = False


def verify_gpu() -> tuple[str, str]:
    physical = os.environ.get("M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES", "")
    expected = os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID", "")
    if physical not in GPU_UUIDS or os.environ.get("CUDA_VISIBLE_DEVICES") != physical or expected != GPU_UUIDS[physical]:
        raise RuntimeError("execution-preservation GPU authorization is invalid")
    actual = subprocess.check_output(
        ["nvidia-smi", "-i", physical, "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip()
    if actual != expected:
        raise RuntimeError("execution-preservation GPU UUID mismatch")
    return physical, actual


def parameter_groups(expert: AsymmetricCPExpert, condition: str) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    output = [expert.u_out, expert.v_out, expert.rho]
    if condition == CONDITIONS[0]:
        for parameter in expert.parameters():
            parameter.requires_grad_(False)
        for parameter in output:
            parameter.requires_grad_(True)
        return [{"params": output, "lr": 1e-3}], output
    if condition not in CONDITIONS[1:]:
        raise ValueError(f"unknown condition: {condition}")
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    inputs = [expert.u_in, expert.v_in]
    return [{"params": output, "lr": 1e-3}, {"params": inputs, "lr": 1e-4}], [*output, *inputs]


def route_loss(
    expert: AsymmetricCPExpert,
    prompt: torch.Tensor,
    visual: list[torch.Tensor],
    fit_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prototypes = build_fit_prototypes(expert, prompt[:fit_count], visual[:fit_count], R2)
    scores = score_with_frozen_prototypes(expert, prompt, visual, R2, prototypes)
    positive, negative = scores[:fit_count], scores[fit_count:]
    logits = torch.cat((positive[:, None], negative.expand(len(positive), -1)), dim=1) / 0.1
    infonce = F.cross_entropy(logits, torch.zeros(len(positive), dtype=torch.long, device=logits.device))
    hinge = F.relu(0.1 + negative.max() - positive.min())
    q = expert.input_basis().float()
    orthogonality = (q.T @ q - torch.eye(expert.rank, device=q.device)).square().mean()
    return infonce + hinge + 0.01 * orthogonality, {
        "infonce": infonce,
        "hinge": hinge,
        "q_orthogonality": orthogonality,
    }


def _grad_snapshot(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(value) if value.grad is None else value.grad.detach().clone() for value in parameters]


def _gradient_norm(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.float().square().sum() for value in values]).sum().sqrt()


def _parameter_deltas(expert: AsymmetricCPExpert, start: dict[str, torch.Tensor]) -> dict[str, float]:
    current = expert.state_dict()
    groups = {"q": ("u_in", "v_in"), "p": ("u_out", "v_out"), "rho": ("rho",)}
    return {
        name: math.sqrt(sum(float((current[key].detach().cpu().float() - start[key].float()).square().sum()) for key in keys))
        for name, keys in groups.items()
    }


def _save_training_checkpoint(
    path: Path,
    *,
    expert: AsymmetricCPExpert,
    optimizer: torch.optim.Optimizer,
    condition: str,
    step: int,
    start_hash: str,
    prototypes: dict[str, torch.Tensor],
    curve: list[dict[str, Any]],
) -> str:
    payload = {
        "schema_version": "medtrace-execution-preservation-checkpoint-v1",
        "condition": condition,
        "step": step,
        "rank": expert.rank,
        "representation": R2,
        "start_state_sha256": start_hash,
        "expert": expert.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "python_rng_state": random.getstate(),
        "next_paraphrase_index": step % 4,
        "prototypes": {name: value.detach().cpu() for name, value in prototypes.items()},
        "q_sha256": tensor_sha256(expert.input_basis()),
        "prototype_sha256": state_hash(prototypes),
        "score_definition_sha256": SCORE_DEFINITION_SHA256,
        "curve": curve,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def _diagnostics(runtime: Any, expert: AsymmetricCPExpert, batches: list[Any], labels: list[str]) -> list[dict[str, Any]]:
    return [{"input": label, **score_state(runtime, expert, batch)} for label, batch in zip(labels, batches, strict=True)]


def load_or_build_anchor_cache(
    runtime: Any,
    teacher: AsymmetricCPExpert,
    batches: list[Any],
    labels: list[str],
    path: Path,
    binding: dict[str, str],
) -> dict[str, Any]:
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["binding"] != binding or cached["labels"] != labels or cached["fit_only"] is not True:
            raise RuntimeError("teacher anchor cache binding mismatch")
        return cached
    values, near_zero = [], []
    with torch.no_grad():
        for batch in batches:
            activation = runtime.extract_layer_input_features(batch, module_path=LAYER)
            mask = torch.zeros_like(batch.labels, dtype=torch.bool)
            mask[:, :-1] = batch.labels[:, 1:] != -100
            residual = teacher.residual(activation)[mask].detach().cpu()
            values.append(residual)
            near_zero.append(float(residual.float().square().sum()) < 1e-8)
    cached = {
        "schema_version": "medtrace-fit-only-teacher-residual-cache-v1",
        "binding": binding,
        "labels": labels,
        "fit_only": True,
        "reference_residuals": values,
        "reference_near_zero": near_zero,
        "stored": "active-predictor residuals only",
        "bytes": sum(value.numel() * value.element_size() for value in values),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(cached, temporary)
    os.replace(temporary, path)
    return cached


def validate_anchor_alignment(runtime: Any, teacher: AsymmetricCPExpert, batch: Any, reference: torch.Tensor) -> float:
    hook = MedTraceLayerHook(runtime.get_module(LAYER), teacher)
    hook.attach()
    try:
        hook.set_teacher_routing(batch.labels)
        hook.set_anchor_reference(reference)
        with torch.no_grad():
            runtime.compute_loss(batch)
        value = float(hook.pop_anchor_loss().item())
    finally:
        hook.detach()
    if value > 1e-6:
        raise RuntimeError(f"teacher anchor cache alignment failed: {value}")
    return value


def train_condition(
    runtime: Any,
    batches: list[Any],
    batch_labels: list[str],
    start_checkpoint: dict[str, Any],
    route_features: tuple[torch.Tensor, list[torch.Tensor], int],
    anchor_cache: dict[str, Any],
    condition: str,
    out: Path,
    *,
    seed: int,
    target_step: int,
) -> dict[str, Any]:
    layer = runtime.get_module(LAYER)
    if (layer.in_features, layer.out_features) != (14336, 4096):
        raise RuntimeError(f"unexpected MedTRACE layer dimensions: {layer.in_features}->{layer.out_features}")
    start_state = {name: value.detach().cpu().clone() for name, value in start_checkpoint["expert"].items()}
    start_hash = state_hash(start_state)
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    groups, optimized = parameter_groups(expert, condition)
    optimizer = torch.optim.AdamW(groups, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    resume = out / "resume.pt"
    curve: list[dict[str, Any]] = []
    if resume.exists():
        saved = torch.load(resume, map_location="cuda:0", weights_only=True)
        if saved["condition"] != condition or saved["start_state_sha256"] != start_hash:
            raise RuntimeError("training resume binding mismatch")
        expert.load_state_dict(saved["expert"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng_state"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
        random.setstate(saved["python_rng_state"])
        if saved["next_paraphrase_index"] != int(saved["step"]) % 4:
            raise RuntimeError("training resume microbatch cursor mismatch")
        curve = saved["curve"]
        completed_step = int(saved["step"])
    else:
        seed_everything(seed)
        expert.load_state_dict(start_state)
        completed_step = 0
    route_prompt, route_visual, fit_count = route_features
    guard_expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    guard_expert.load_state_dict(start_state)
    initial_q_hash = tensor_sha256(guard_expert.input_basis())
    del guard_expert
    prototypes = build_fit_prototypes(expert, route_prompt[:fit_count], route_visual[:fit_count], R2)
    initial_prototype_hash = state_hash(prototypes)
    if completed_step == 0 and not (out / "step0000.pt").exists():
        checkpoint_sha = _save_training_checkpoint(
            out / "step0000.pt", expert=expert, optimizer=optimizer, condition=condition, step=0,
            start_hash=start_hash, prototypes=prototypes, curve=curve,
        )
        atomic_json(out / "diagnostics_step0000.json", {"step": 0, "checkpoint_sha256": checkpoint_sha, "values": _diagnostics(runtime, expert, batches, batch_labels)})
    hook = MedTraceLayerHook(layer, expert)
    hook.attach()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    window = defaultdict(lambda: torch.zeros((), device="cuda:0"))
    window_steps = 0
    try:
        for step in range(completed_step + 1, target_step + 1):
            optimizer.zero_grad(set_to_none=True)
            selected = (0, 1 + (step - 1) % (len(batches) - 1))
            for index in selected:
                batch = batches[index]
                hook.set_teacher_routing(batch.labels)
                if condition == CONDITIONS[2]:
                    hook.set_anchor_reference(anchor_cache["reference_residuals"][index])
                task = runtime.compute_loss(batch)
                anchor = hook.pop_anchor_loss() if condition == CONDITIONS[2] else task.new_zeros(())
                (0.5 * task + 0.5 * anchor).backward()
                window["task_ce"] += task.detach()
                window["anchor"] += anchor.detach()
            log_step = step % 20 == 0 or step in MILESTONES or step == target_step
            task_grads = _grad_snapshot(optimized) if log_step else []
            if condition != CONDITIONS[0]:
                scope, parts = route_loss(expert, route_prompt, route_visual, fit_count)
                (0.1 * scope).backward()
                window["scope"] += scope.detach()
                for name, value in parts.items():
                    window[name] += value.detach()
            else:
                window["scope"] += torch.zeros((), device="cuda:0")
            if log_step:
                total_grads = _grad_snapshot(optimized)
                route_grads = [total - task for total, task in zip(total_grads, task_grads, strict=True)]
                task_grad_norm = _gradient_norm(task_grads)
                route_grad_norm = _gradient_norm(route_grads)
            grad_norm = torch.nn.utils.clip_grad_norm_(optimized, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite execution-preservation gradient")
            optimizer.step()
            if condition == CONDITIONS[0]:
                expert.normalize_output_factors_(verify_dense=step in MILESTONES or step == target_step)
            else:
                expert.normalize_factors_(verify_dense=step in MILESTONES or step == target_step)
            window_steps += 1
            if log_step:
                row = {
                    "step": step,
                    "task_ce": float((window["task_ce"] / (2 * window_steps)).item()),
                    "scope": float((window["scope"] / window_steps).item()),
                    "anchor": float((window["anchor"] / (2 * window_steps)).item()),
                    "infonce": float((window["infonce"] / max(window_steps, 1)).item()),
                    "hinge": float((window["hinge"] / max(window_steps, 1)).item()),
                    "q_orthogonality": float((window["q_orthogonality"] / max(window_steps, 1)).item()),
                    "task_gradient_norm": float(task_grad_norm.item()),
                    "route_gradient_norm": float(route_grad_norm.item()),
                    "clipped_total_gradient_norm": float(grad_norm.item()),
                    **{f"{name}_delta_l2": value for name, value in _parameter_deltas(expert, start_state).items()},
                }
                if not all(math.isfinite(value) for key, value in row.items() if key != "step"):
                    raise FloatingPointError("non-finite execution-preservation training trace")
                curve.append(row)
                window.clear()
                window_steps = 0
            if step in MILESTONES or step == target_step or STOP_REQUESTED:
                hook.detach()
                diagnostics = _diagnostics(runtime, expert, batches, batch_labels)
                hook.attach()
                prototypes = build_fit_prototypes(expert, route_prompt[:fit_count], route_visual[:fit_count], R2)
                checkpoint_path = out / f"step{step:04d}.pt"
                checkpoint_sha = _save_training_checkpoint(
                    checkpoint_path, expert=expert, optimizer=optimizer, condition=condition, step=step,
                    start_hash=start_hash, prototypes=prototypes, curve=curve,
                )
                _save_training_checkpoint(
                    resume, expert=expert, optimizer=optimizer, condition=condition, step=step,
                    start_hash=start_hash, prototypes=prototypes, curve=curve,
                )
                atomic_json(out / f"diagnostics_step{step:04d}.json", {"step": step, "checkpoint_sha256": checkpoint_sha, "values": diagnostics})
                if condition == CONDITIONS[0] and tensor_sha256(expert.input_basis()) != initial_q_hash:
                    raise RuntimeError("C1 changed frozen Q")
                if condition == CONDITIONS[0] and state_hash(prototypes) != initial_prototype_hash:
                    raise RuntimeError("C1 changed frozen fit prototypes")
                if STOP_REQUESTED:
                    raise InterruptedError("stop requested after resumable checkpoint")
    finally:
        hook.detach()
    result = {
        "schema_version": "medtrace-execution-preservation-training-v1",
        "status": "TRAINING_COMPLETE",
        "condition": condition,
        "start_state_sha256": start_hash,
        "completed_step": target_step,
        "optimizer_steps": target_step,
        "micro_forwards": target_step * 2,
        "curve": curve,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "initial_q_sha256": initial_q_hash,
        "final_q_sha256": tensor_sha256(expert.input_basis()),
        "reference_near_zero_count": sum(anchor_cache["reference_near_zero"]) if condition == CONDITIONS[2] else 0,
        "anchor_cache_bytes": anchor_cache["bytes"] if condition == CONDITIONS[2] else 0,
    }
    atomic_json(out / f"training_step{target_step:04d}.json", result)
    return result


def calibration_for_checkpoint(
    checkpoint: dict[str, Any],
    cal_rows: list[dict[str, Any]],
    cal_prompt: torch.Tensor,
    cal_visual: list[torch.Tensor],
    hard_evaluable: bool,
) -> dict[str, Any]:
    expert = AsymmetricCPExpert(cal_prompt.shape[-1], 4096, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"])
    scores = [
        {"logical_id": value["row"]["logical_id"], "label": value["row"]["label"], "fact_relation": value["row"]["fact_relation"], "score": route_score_one(expert, checkpoint, prompt, visual)}
        for value, prompt, visual in zip(cal_rows, cal_prompt, cal_visual, strict=True)
    ]
    positive = [row["score"] for row in scores if row["label"] == "positive"]
    hard = [row["score"] for row in scores if row["fact_relation"] == "same_question_different_image_conflicting_source_answer"]
    broad = [row["score"] for row in scores if row["fact_relation"] == "broad_unrelated_source_qa"]
    return {
        "scores": scores,
        "hard_evaluable": hard_evaluable,
        "operating_points": calibrate_operating_points(positive, hard, broad, hard_evaluable=hard_evaluable),
        "q_sha256": checkpoint["q_sha256"],
        "prototype_sha256": checkpoint["prototype_sha256"],
        "score_definition_sha256": checkpoint["score_definition_sha256"],
    }


def evaluate_checkpoint(
    runtime: Any,
    *,
    checkpoint_path: Path,
    calibration_path: Path,
    old_result: dict[str, Any],
    cache: dict[str, Any],
    out: Path,
    condition: str,
    step: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)
    calibration = json.loads(calibration_path.read_text())
    expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"])
    reloaded = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)["expert"])
    if any(not torch.equal(a, b) for a, b in zip(expert.state_dict().values(), reloaded.state_dict().values(), strict=True)):
        raise RuntimeError("checkpoint save/reload mismatch")
    old_by_id = {item["row"]["logical_id"]: item for item in old_result["outputs"]}
    values = [
        value for value in cache["values"].values()
        if value["row"]["role"] in {"fit", "evaluation", "challenge", "native", "formal_development"}
        and not (value["row"]["role"] == "fit" and value["row"]["label"] != "positive")
    ]
    values.sort(key=lambda value: value["row"]["logical_id"])
    items, off_validated = [], False
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
    hook.attach()
    try:
        for value in values:
            row = value["row"]
            if row["logical_id"] in old_by_id:
                base = old_by_id[row["logical_id"]]["outputs"]["base"]
                base_source = "REUSED_EXACT_OLD_E2E"
            else:
                base = scope_generate(runtime, row, None)
                base_source = "NEW_FIT_DIAGNOSTIC"
            forced = scope_generate(runtime, row, hook)
            score = route_score_one(expert, checkpoint, value["prompt"].to("cuda:0"), value["visual"].to("cuda:0"))
            decisions = {}
            for operating_point in OPERATING_POINTS:
                threshold = calibration["operating_points"][operating_point]["threshold"]
                on = score > threshold
                decisions[operating_point] = {"threshold": threshold, "score": score, "on": on, "gated_source": "DERIVED_FORCED" if on else "DERIVED_BASE"}
                if not on and not off_validated:
                    replay = scope_generate(runtime, row, None)
                    if replay["raw_token_ids"] != base["raw_token_ids"] or replay["raw_answer"] != base["raw_answer"]:
                        raise RuntimeError("real OFF replay did not match frozen base")
                    off_validated = True
            items.append({"row": row, "base": base, "base_source": base_source, "forced": forced, "decisions": decisions})
    finally:
        hook.detach()
    on_states = {point: [item["decisions"][point]["on"] for item in items] for point in OPERATING_POINTS}
    result = {
        "schema_version": "medtrace-execution-preservation-e2e-v1",
        "status": "GENERATION_COMPLETE",
        "condition": condition,
        "step": step,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "items": items,
        "replay": {
            "checkpoint_reload_exact": True,
            "off_path": "VALIDATED" if off_validated else "ALL_ON",
            "on_path": "VALIDATED_FORCED" if any(any(values) for values in on_states.values()) else "ALL_OFF",
        },
    }
    atomic_json(out, result)
    return result


def condition_order(seed: int, event_index: int) -> list[str]:
    offset = int(hashlib.sha256(f"{seed}:{event_index}".encode()).hexdigest(), 16) % len(CONDITIONS)
    return list(CONDITIONS[offset:] + CONDITIONS[:offset])


def _load_task_assets(runtime: Any, args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    old = args.old_run
    frozen = json.loads((old / "private/frozen_data.json").read_text())
    event = frozen["dev"][task["event_index"] - 1]
    record = EditorRecord.from_dict(event["edit_record"])
    scope = frozen["scopes"][task["record_id"]]
    a_dir = old / "private/tasks" / f"A_s{task['seed']}_e{task['event_index']:02d}" / "CP_NATIVE_PLUS_PARAPHRASE_80"
    e2e_dir = old / "private/tasks" / f"E2E_s{task['seed']}_e{task['event_index']:02d}"
    b_dir = old / "private/tasks" / f"B_s{task['seed']}_e{task['event_index']:02d}"
    a_result = json.loads((a_dir / "result.json").read_text())
    old_result = json.loads((e2e_dir / "result_private.json").read_text())
    b_result = json.loads((b_dir / "result_private.json").read_text())
    if sha256_file(a_dir / "expert.pt") != a_result["checkpoint_sha256"]:
        raise RuntimeError("A2 teacher checkpoint binding failure")
    if sha256_file(e2e_dir / "ALT.pt") != old_result["profiles"]["ALT"]["checkpoint_sha256"]:
        raise RuntimeError("R2-80 starting checkpoint binding failure")
    cache = torch.load(old / f"private/features/e{task['event_index']:02d}.pt", map_location="cpu", weights_only=False)
    if cache["locks"] != b_result["locks"] or cache["schema_version"] != "medtrace-route-feature-cache-private-v1":
        raise RuntimeError("target-free route cache binding failure")
    start = torch.load(e2e_dir / "ALT.pt", map_location="cuda:0", weights_only=True)
    teacher_state = torch.load(a_dir / "expert.pt", map_location="cuda:0", weights_only=True)["expert"]
    teacher = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    questions = [row["question"] for row in frozen["generality_paraphrases"][task["record_id"]]]
    batches = [runtime.build_edit_batch(record), *[runtime.build_edit_batch(replace(record, question=value)) for value in questions]]
    labels = ["native", *[f"fit_paraphrase_{index}" for index in range(1, len(questions) + 1)]]
    anchor_path = args.run_root / "private/teacher_anchors" / f"s{task['seed']}_e{task['event_index']:02d}.pt"
    anchor = load_or_build_anchor_cache(
        runtime, teacher, batches, labels, anchor_path,
        {"teacher_checkpoint_sha256": a_result["checkpoint_sha256"], "record_id": task["record_id"]},
    )
    validate_anchor_alignment(runtime, teacher, batches[0], anchor["reference_residuals"][0])
    fit_rows, fit_prompt, fit_visual = ordered_features(cache, roles=("fit",))
    fit_count = sum(value["row"]["label"] == "positive" for value in fit_rows)
    if fit_count != 4 or len(fit_rows) <= fit_count:
        raise RuntimeError("route fit cache coverage failure")
    cal_rows, cal_prompt, cal_visual = ordered_features(cache, roles=("calibration",))
    return {
        "frozen": frozen, "event": event, "record": record, "scope": scope, "cache": cache,
        "start": start, "teacher": teacher, "batches": batches, "batch_labels": labels, "anchor": anchor,
        "route": (fit_prompt, fit_visual, fit_count), "cal": (cal_rows, cal_prompt, cal_visual), "old_result": old_result,
    }


def _ensure_calibrations(assets: dict[str, Any], condition_dir: Path, scope_status: str, steps: tuple[int, ...]) -> None:
    cal_rows, cal_prompt, cal_visual = assets["cal"]
    for step in steps:
        path = condition_dir / f"calibration_step{step:04d}.json"
        if path.exists():
            continue
        checkpoint = torch.load(condition_dir / f"step{step:04d}.pt", map_location="cuda:0", weights_only=True)
        atomic_json(path, calibration_for_checkpoint(checkpoint, cal_rows, cal_prompt, cal_visual, scope_status == "HARD_EVALUABLE"))


def run_p0(runtime: Any, args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    assets = _load_task_assets(runtime, args, task)
    failures, completed = {}, []
    start_hash = state_hash(assets["start"]["expert"])
    triple_dir = args.run_root / "private/tasks" / task["task_id"]
    for condition in task["condition_order"]:
        condition_dir = triple_dir / condition
        try:
            train_condition(
                runtime, assets["batches"], assets["batch_labels"], assets["start"], assets["route"], assets["anchor"],
                condition, condition_dir, seed=derive_seed(task["record_id"], base=task["seed"]), target_step=800,
            )
            _ensure_calibrations(assets, condition_dir, assets["scope"]["status"], MILESTONES)
            evaluate_checkpoint(
                runtime, checkpoint_path=condition_dir / "step0800.pt", calibration_path=condition_dir / "calibration_step0800.json",
                old_result=assets["old_result"], cache=assets["cache"], out=condition_dir / "e2e_step0800.json",
                condition=condition, step=800,
            )
            completed.append(condition)
        except Exception as error:
            failures[condition] = f"{type(error).__name__}: {error}"
            append_jsonl(args.run_root / "private/CONDITION_ERRORS.jsonl", {"task_id": task["task_id"], "condition": condition, "error": failures[condition], "at": time.time()})
            torch.cuda.empty_cache()
    result = {
        "status": "COMPLETE" if len(completed) == len(CONDITIONS) else "PARTIAL_CONDITION_FAILURE",
        "start_state_sha256": start_hash,
        "completed_conditions": completed,
        "failures": failures,
    }
    atomic_json(triple_dir / "result.json", result)
    return result


def run_p1(runtime: Any, args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    assets = _load_task_assets(runtime, args, task)
    condition_dir = args.run_root / "private/tasks" / task["depends_on"] / task["condition"]
    completed = []
    for step in (80, 320):
        checkpoint = condition_dir / f"step{step:04d}.pt"
        if not checkpoint.exists():
            continue
        _ensure_calibrations(assets, condition_dir, assets["scope"]["status"], (step,))
        evaluate_checkpoint(
            runtime, checkpoint_path=checkpoint, calibration_path=condition_dir / f"calibration_step{step:04d}.json",
            old_result=assets["old_result"], cache=assets["cache"], out=condition_dir / f"e2e_step{step:04d}.json",
            condition=task["condition"], step=step,
        )
        completed.append(step)
    result = {"status": "COMPLETE" if completed == [80, 320] else "SKIPPED_UPSTREAM_FAILURE", "completed_steps": completed}
    atomic_json(args.run_root / "private/tasks" / task["task_id"] / "result.json", result)
    return result


def run_p2(runtime: Any, args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    assets = _load_task_assets(runtime, args, task)
    failures, completed = {}, []
    triple_dir = args.run_root / "private/tasks" / task["p0_task_id"]
    for condition in task["condition_order"]:
        condition_dir = triple_dir / condition
        if not (condition_dir / "step0800.pt").exists():
            failures[condition] = "missing step800 checkpoint"
            continue
        try:
            train_condition(
                runtime, assets["batches"], assets["batch_labels"], assets["start"], assets["route"], assets["anchor"],
                condition, condition_dir, seed=derive_seed(task["record_id"], base=task["seed"]), target_step=1600,
            )
            _ensure_calibrations(assets, condition_dir, assets["scope"]["status"], (1600,))
            evaluate_checkpoint(
                runtime, checkpoint_path=condition_dir / "step1600.pt", calibration_path=condition_dir / "calibration_step1600.json",
                old_result=assets["old_result"], cache=assets["cache"], out=condition_dir / "e2e_step1600.json",
                condition=condition, step=1600,
            )
            completed.append(condition)
        except Exception as error:
            failures[condition] = f"{type(error).__name__}: {error}"
            append_jsonl(args.run_root / "private/CONDITION_ERRORS.jsonl", {"task_id": task["task_id"], "condition": condition, "error": failures[condition], "at": time.time()})
    result = {"status": "COMPLETE" if len(completed) == len(CONDITIONS) else "PARTIAL_CONDITION_FAILURE", "completed_conditions": completed, "failures": failures}
    atomic_json(args.run_root / "private/tasks" / task["task_id"] / "result.json", result)
    return result


def budget_exhausted(run_root: Path, queue: TaskQueue) -> bool:
    start = json.loads((run_root / "private/CAMPAIGN_START.json").read_text())
    wall = time.time() - start["first_gpu_epoch"]
    used = sum(row.get("elapsed_seconds", 0.0) for row in queue.snapshot()["tasks"] if row["status"] in {"COMPLETE", "FAILED"})
    return wall >= 20 * 3600 or used >= 44 * 3600 or (run_root / "STOP").exists() or STOP_REQUESTED


def worker(args: argparse.Namespace) -> None:
    physical, _ = verify_gpu()
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("execution-preservation worker code commit mismatch")
    queue = TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=args.cpu_gate))
    worker_id = f"gpu{physical}"
    try:
        with Telemetry(args.run_root / "private/GPU_TELEMETRY.jsonl", physical, worker_id) as telemetry:
            while True:
                if budget_exhausted(args.run_root, queue):
                    queue.cancel_pending()
                    break
                task = queue.claim(worker_id)
                if task is None:
                    snapshot = queue.snapshot()["tasks"]
                    if all(row["status"] in {"COMPLETE", "FAILED", "CANCELLED_BY_BUDGET"} for row in snapshot):
                        break
                    time.sleep(5)
                    continue
                telemetry.task_id = task["task_id"]
                started = time.monotonic()
                try:
                    if task["kind"] == "P0":
                        value = run_p0(runtime, args, task)
                    elif task["kind"] == "P1":
                        value = run_p1(runtime, args, task)
                    elif task["kind"] == "P2":
                        value = run_p2(runtime, args, task)
                    else:
                        raise RuntimeError(f"unknown task kind: {task['kind']}")
                    queue.update(task["task_id"], "COMPLETE", result_status=value["status"], elapsed_seconds=time.monotonic() - started, finished_at=time.time())
                except torch.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    if task["attempts"] < 2:
                        queue.update(task["task_id"], "PENDING", last_error=f"OOM retry: {error}", elapsed_seconds=time.monotonic() - started)
                    else:
                        queue.update(task["task_id"], "FAILED", last_error=f"OOM after retry: {error}", elapsed_seconds=time.monotonic() - started)
                except Exception as error:
                    queue.update(task["task_id"], "FAILED", last_error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
                    append_jsonl(args.run_root / "private/WORKER_ERRORS.jsonl", {"task_id": task["task_id"], "worker": worker_id, "error": f"{type(error).__name__}: {error}", "at": time.time()})
                finally:
                    telemetry.task_id = None
                    torch.cuda.empty_cache()
    finally:
        del runtime
        torch.cuda.empty_cache()


def prepare(args: argparse.Namespace) -> None:
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    ancestry = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", args.base_commit, actual]).returncode
    if ancestry:
        raise RuntimeError("campaign code is not descended from the required start commit")
    frozen = json.loads((args.old_run / "private/frozen_data.json").read_text())
    fixed = json.loads((args.fixed_run / "private/RECALCULATION_COMPLETION.json").read_text())
    if fixed.get("status") != "FIXED_CALIBRATION_COMPLETE" or fixed.get("candidate_checkpoint_count") != 216:
        raise RuntimeError("corrected prototype/calibration binding is incomplete")
    executable = [
        (index, event, frozen["scopes"][event["edit_record"]["record_id"]])
        for index, event in enumerate(frozen["dev"], 1)
        if frozen["scopes"][event["edit_record"]["record_id"]]["status"] in {"HARD_EVALUABLE", "BROAD_ONLY"}
    ]
    if len(executable) != 12 or sum(scope["status"] == "HARD_EVALUABLE" for _, _, scope in executable) != 7:
        raise RuntimeError("frozen executable cohort is not 7 HARD plus 5 BROAD")
    tasks, ledger = [], []
    for seed_index, seed in enumerate((20260906, 20260907, 20260908)):
        for cohort_index, (event_index, event, scope) in enumerate(executable):
            record_id = event["edit_record"]["record_id"]
            p0 = f"P0_s{seed}_e{event_index:02d}"
            order = condition_order(seed, event_index)
            tasks.append({
                "task_id": p0, "kind": "P0", "seed": seed, "event_index": event_index, "record_id": record_id,
                "scope_status": scope["status"], "condition_order": order, "priority": seed_index * 100 + cohort_index,
                "status": "PENDING", "attempts": 0,
            })
            for condition_index, condition in enumerate(CONDITIONS):
                p1 = f"P1_s{seed}_e{event_index:02d}_{condition[:2]}"
                tasks.append({
                    "task_id": p1, "kind": "P1", "seed": seed, "event_index": event_index, "record_id": record_id,
                    "scope_status": scope["status"], "condition": condition, "depends_on": p0,
                    "priority": 1000 + (0 if scope["status"] == "HARD_EVALUABLE" else 500) + seed_index * 100 + cohort_index * 3 + condition_index,
                    "status": "PENDING", "attempts": 0,
                })
            for condition in CONDITIONS:
                ledger.append({
                    "phase": "P0", "seed": seed, "event_index": event_index,
                    "opaque_edit": hashlib.sha256(record_id.encode()).hexdigest()[:16], "scope_status": scope["status"],
                    "condition": condition, "additional_steps": 800, "status": "PENDING",
                })
    args.run_root.mkdir(parents=True)
    (args.run_root / "private").mkdir()
    atomic_json(args.run_root / "private/TASK_QUEUE.json", {"schema_version": "medtrace-execution-preservation-queue-v1", "tasks": tasks})
    atomic_json(args.run_root / "private/CAMPAIGN_CONFIG.json", {
        "schema_version": "medtrace-execution-preservation-config-v1",
        "base_commit": args.base_commit, "code_commit": actual, "old_run": str(args.old_run), "fixed_run": str(args.fixed_run),
        "conditions": list(CONDITIONS), "seeds": [20260906, 20260907, 20260908], "executable_edits": 12,
        "hard_edits": 7, "broad_only_edits": 5, "fit_only_excluded": 4,
        "additional_steps": 800, "checkpoints": list(MILESTONES), "wall_hours": 24, "gpu_hours": 48,
        "gpu_uuids": {key: GPU_UUIDS[key] for key in ("2", "3")}, "gpu1_forbidden": True,
    })
    args.public_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.public_dir / "CAMPAIGN_CONFIG.json", {
        "schema_version": "medtrace-execution-preservation-public-config-v1", "base_commit": args.base_commit,
        "code_commit": actual, "conditions": list(CONDITIONS), "seeds": [20260906, 20260907, 20260908],
        "cohort": {"hard_evaluable": 7, "broad_only": 5, "fit_only_excluded": 4},
        "main_trajectories": 108, "optimizer_steps_each": 800, "micro_forwards_each_step": 2,
        "devices": {"physical": [2, 3], "gpu1_forbidden": True}, "private_artifacts_withheld": True,
    })
    atomic_text(args.public_dir / "CAMPAIGN_PROTOCOL.md", """# MedTRACE execution-preservation campaign\n\nStatus: `READY_NOT_STARTED`\n\nThis development campaign compares three preregistered continuations from the same corrected R2@800 plus 80-step output-recovery expert: fixed-Q long recovery (C1), shared-Q joint task/scope training (C2), and the same joint training with a fit/native A2 residual anchor (C3). It covers 12 scope-executable edits times three seeds, for 108 real-model training trajectories.\n\nThe primary checkpoint is additional step 800. Step 80/320 are budget diagnostics; step 1600 is permitted only by the frozen fit-only trigger and remaining-budget rule. Calibration, evaluation and generation keep fit/calibration/evaluation roles separate. This is a viewed DEV study, not blind qualification, full TIME, clinical validation, or a change to the historical LoRA qualification result.\n""")
    atomic_text(args.public_dir / "TASK_LEDGER.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger))


def recover(args: argparse.Namespace) -> None:
    TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root).recover()


def smoke(args: argparse.Namespace) -> None:
    verify_gpu()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=args.cpu_gate))
    frozen = json.loads((args.old_run / "private/frozen_data.json").read_text())
    event_index = next(index for index, event in enumerate(frozen["dev"], 1) if frozen["scopes"][event["edit_record"]["record_id"]]["status"] == "HARD_EVALUABLE")
    record_id = frozen["dev"][event_index - 1]["edit_record"]["record_id"]
    task = {"task_id": "SMOKE", "seed": 20260906, "event_index": event_index, "record_id": record_id}
    smoke_args = argparse.Namespace(old_run=args.old_run, run_root=args.out)
    try:
        assets = _load_task_assets(runtime, smoke_args, task)
        start_hashes = []
        for condition in CONDITIONS:
            result = train_condition(
                runtime, assets["batches"], assets["batch_labels"], assets["start"], assets["route"], assets["anchor"],
                condition, args.out / condition, seed=derive_seed(record_id, base=20260906), target_step=1,
            )
            start_hashes.append(result["start_state_sha256"])
        if len(set(start_hashes)) != 1:
            raise RuntimeError("smoke conditions did not share one starting state")
        atomic_json(args.out / "SMOKE_RESULT.json", {"status": "PASS", "conditions": list(CONDITIONS), "start_state_sha256": start_hashes[0]})
    finally:
        del runtime
        torch.cuda.empty_cache()


def schedule_p2(args: argparse.Namespace) -> None:
    queue_path = args.run_root / "private/TASK_QUEUE.json"
    data = json.loads(queue_path.read_text())
    if any(task["kind"] == "P2" for task in data["tasks"]):
        return
    p0 = [task for task in data["tasks"] if task["kind"] == "P0"]
    p1 = [task for task in data["tasks"] if task["kind"] == "P1"]
    elapsed = time.time() - json.loads((args.run_root / "private/CAMPAIGN_START.json").read_text())["first_gpu_epoch"]
    all_complete = all(task["status"] == "COMPLETE" for task in [*p0, *p1])
    ce_trigger = False
    coverage_trigger = False
    for task in p0:
        triple = args.run_root / "private/tasks" / task["task_id"]
        for condition in CONDITIONS:
            training = triple / condition / "training_step0800.json"
            e2e = triple / condition / "e2e_step0800.json"
            if not training.exists() or not e2e.exists():
                continue
            rows = json.loads(training.read_text())["curve"]
            recent = [row["task_ce"] for row in rows if row["step"] > 600]
            if len(recent) >= 2 and recent[-1] < 0.95 * recent[0]:
                ce_trigger = True
            items = [item for item in json.loads(e2e.read_text())["items"] if item["row"]["role"] in {"native", "fit"}]
            exact = [normalize_medical_answer(item["forced"]["raw_answer"]) == normalize_medical_answer(item["row"]["reference"]) for item in items]
            if exact and sum(exact) / len(exact) < 0.9:
                coverage_trigger = True
    trigger = all_complete and elapsed < 16 * 3600 and (ce_trigger or coverage_trigger)
    if trigger:
        for index, source in enumerate(p0):
            data["tasks"].append({
                "task_id": source["task_id"].replace("P0_", "P2_"), "kind": "P2", "seed": source["seed"],
                "event_index": source["event_index"], "record_id": source["record_id"], "scope_status": source["scope_status"],
                "condition_order": source["condition_order"], "p0_task_id": source["task_id"], "priority": 3000 + index,
                "status": "PENDING", "attempts": 0,
            })
    atomic_json(queue_path, data)
    atomic_json(args.run_root / "private/P2_DECISION.json", {
        "all_p0_p1_complete": all_complete, "elapsed_hours": elapsed / 3600, "fit_ce_trigger": ce_trigger,
        "fit_exact_coverage_trigger": coverage_trigger, "scheduled": trigger,
    })


def _old_judge_map(old_run: Path) -> dict[tuple[str, str, str], bool]:
    judge = old_run / "private/judge"
    packets = {row["opaque_query_id"]: row for row in read_jsonl(judge / "JUDGE_PACKET_PRIVATE.jsonl")}
    outputs = {row["opaque_query_id"]: row for row in read_jsonl(judge / "JUDGE_OUTPUT_PRIVATE.jsonl")}
    return {
        (row["question"], row["gold_answer"], row["raw_base_answer"]): bool(outputs[opaque]["is_correct"])
        for opaque, row in packets.items() if opaque in outputs and outputs[opaque].get("parse_valid")
    }


def _evaluation_files(run_root: Path) -> list[Path]:
    return sorted((run_root / "private/tasks").glob("P0_*/C*/e2e_step*.json"))


def prepare_judge(args: argparse.Namespace) -> None:
    old = _old_judge_map(args.old_run)
    packet, tuple_rows, sidecar = {}, {}, []
    for path in _evaluation_files(args.run_root):
        result = json.loads(path.read_text())
        task_id = path.parents[1].name
        for item in result["items"]:
            row = item["row"]
            for output_path in ("base", "forced"):
                raw = item[output_path]["raw_answer"]
                key = (row["question"], row["reference"], raw)
                opaque = sha256_json((*key, "medtrace-execution-preserving-semantic-v1"))
                tuple_rows[opaque] = {"question": key[0], "reference": key[1], "raw": key[2], "reused_verdict": old.get(key)}
                sidecar.append({"opaque_query_id": opaque, "task_id": task_id, "condition": result["condition"], "step": result["step"], "logical_id": row["logical_id"], "path": output_path})
                if key not in old:
                    packet[opaque] = {"opaque_query_id": opaque, "question": key[0], "gold_answer": key[1], "raw_base_answer": key[2], "adjudication_pass": 1}
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.packet, "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet.values()))
    atomic_json(args.sidecar, {"tuples": tuple_rows, "uses": sidecar, "reused_count": sum(row["reused_verdict"] is not None for row in tuple_rows.values()), "new_count": len(packet)})


def _verdicts(sidecar_path: Path, judge_output: Path) -> dict[str, bool]:
    sidecar = json.loads(sidecar_path.read_text())
    values = {opaque: bool(row["reused_verdict"]) for opaque, row in sidecar["tuples"].items() if row["reused_verdict"] is not None}
    if judge_output.exists():
        for row in read_jsonl(judge_output):
            if not row.get("parse_valid"):
                raise RuntimeError("Judge output contains a parse failure")
            values[row["opaque_query_id"]] = bool(row["is_correct"])
    if set(sidecar["tuples"]) != set(values):
        raise RuntimeError("Judge closure does not cover every exact tuple")
    return values


def _correct(verdicts: dict[str, bool], row: dict[str, Any], raw: str) -> bool:
    return verdicts[sha256_json((row["question"], row["reference"], raw, "medtrace-execution-preserving-semantic-v1"))]


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "on_num": sum(row["on"] for row in rows),
        "on_rate": sum(row["on"] for row in rows) / n if n else None,
        "base_correct_num": sum(row["base_correct"] for row in rows),
        "forced_correct_num": sum(row["forced_correct"] for row in rows),
        "gated_correct_num": sum(row["gated_correct"] for row in rows),
        "base_correct_rate": sum(row["base_correct"] for row in rows) / n if n else None,
        "forced_correct_rate": sum(row["forced_correct"] for row in rows) / n if n else None,
        "gated_correct_rate": sum(row["gated_correct"] for row in rows) / n if n else None,
        "joint_on_and_correct_rate": sum(row["on"] and row["forced_correct"] for row in rows) / n if n else None,
        "base_correct_to_gated_wrong_num": sum(row["base_correct"] and not row["gated_correct"] for row in rows),
        "gate_rejected_correct_positive_num": sum((not row["on"]) and row["forced_correct"] for row in rows),
    }


def _edit_macro(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    by_edit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_edit[row["event_index"]].append(row)
    metrics = [_metric(values) for values in by_edit.values()]
    keys = ("on_rate", "base_correct_rate", "forced_correct_rate", "gated_correct_rate", "joint_on_and_correct_rate")
    return {key: mean(value[key] for value in metrics if value[key] is not None) if metrics else None for key in keys}


def _cluster_bootstrap(rows: list[dict[str, Any]], key: str, *, draws: int = 2000) -> tuple[float | None, float | None]:
    by_edit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_edit[row["event_index"]].append(row)
    if not by_edit:
        return None, None
    values = list(by_edit.values())
    rng = random.Random(20260906)
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(values) for _ in values]
        samples.append(mean(_metric(cluster)[key] for cluster in chosen))
    samples.sort()
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def finalize(args: argparse.Namespace) -> None:
    verdicts = _verdicts(args.sidecar, args.judge_output)
    detailed = []
    for path in _evaluation_files(args.run_root):
        result = json.loads(path.read_text())
        task_id = path.parents[1].name
        queue_task = next(row for row in json.loads((args.run_root / "private/TASK_QUEUE.json").read_text())["tasks"] if row["task_id"] == task_id)
        opaque_edit = hashlib.sha256(queue_task["record_id"].encode()).hexdigest()[:16]
        for item in result["items"]:
            row = item["row"]
            base_correct = _correct(verdicts, row, item["base"]["raw_answer"])
            forced_correct = _correct(verdicts, row, item["forced"]["raw_answer"])
            for point in OPERATING_POINTS:
                on = item["decisions"][point]["on"]
                detailed.append({
                    "seed": queue_task["seed"], "event_index": queue_task["event_index"], "opaque_edit": opaque_edit,
                    "scope_status": queue_task["scope_status"], "condition": result["condition"], "step": result["step"],
                    "operating_point": point, "subset": evaluation_category(row) if row["role"] != "fit" else "fit_positive",
                    "on": on, "base_correct": base_correct, "forced_correct": forced_correct,
                    "gated_correct": forced_correct if on else base_correct,
                })
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[(row["seed"], row["event_index"], row["opaque_edit"], row["scope_status"], row["condition"], row["step"], row["operating_point"], row["subset"])].append(row)
    columns = ["seed", "event_index", "opaque_edit", "scope_status", "condition", "step", "operating_point", "subset", *list(_metric([]))]
    csv_path = args.public_dir / "PAIRED_E2E_BY_EDIT_SEED.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for key, rows in sorted(grouped.items()):
            writer.writerow(dict(zip(columns[:8], key, strict=True)) | _metric(rows))
    curves = []
    for path in sorted((args.run_root / "private/tasks").glob("P0_*/C*/training_step*.json")):
        value = json.loads(path.read_text())
        if value["completed_step"] not in {800, 1600}:
            continue
        task_id = path.parents[1].name
        task = next(row for row in json.loads((args.run_root / "private/TASK_QUEUE.json").read_text())["tasks"] if row["task_id"] == task_id)
        for row in value["curve"]:
            curves.append({"seed": task["seed"], "event_index": task["event_index"], "opaque_edit": hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], "condition": value["condition"], **row})
    curve_columns = sorted({key for row in curves for key in row}, key=lambda key: (key not in {"seed", "event_index", "opaque_edit", "condition", "step"}, key))
    with (args.public_dir / "EXECUTION_PRESERVATION_TRAINING_CURVES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(curves)
    calibrations = []
    for path in sorted((args.run_root / "private/tasks").glob("P0_*/C*/calibration_step*.json")):
        value = json.loads(path.read_text()); condition = path.parent.name; task_id = path.parents[1].name
        task = next(row for row in json.loads((args.run_root / "private/TASK_QUEUE.json").read_text())["tasks"] if row["task_id"] == task_id)
        step = int(path.stem[-4:])
        for point, metrics in value["operating_points"].items():
            calibrations.append({"seed": task["seed"], "event_index": task["event_index"], "opaque_edit": hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], "scope_status": task["scope_status"], "condition": condition, "step": step, "operating_point": point, **metrics})
    if calibrations:
        with (args.public_dir / "CHECKPOINT_CALIBRATION.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibrations[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(calibrations)
    main = [row for row in detailed if row["step"] == 800 and row["operating_point"] == "SAFETY_FIRST"]
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in main if row["condition"] == condition]
        positive_rows = [row for row in rows if row["subset"] in {"native", "evaluation_positive"}]
        hard_rows = [row for row in rows if row["subset"] == "same_question_different_image_conflicting_source_answer"]
        broad_rows = [row for row in rows if row["subset"] == "broad_unrelated_source_qa"]
        summary[condition] = {
            "positive": _metric(positive_rows) | {"edit_macro": _edit_macro(positive_rows), "gated_edit_bootstrap_95ci": _cluster_bootstrap(positive_rows, "gated_correct_rate")},
            "hard": _metric(hard_rows) | {"edit_macro": _edit_macro(hard_rows), "on_edit_bootstrap_95ci": _cluster_bootstrap(hard_rows, "on_rate")},
            "broad": _metric(broad_rows) | {"edit_macro": _edit_macro(broad_rows)},
            "all": _metric(rows),
        }
    queue = json.loads((args.run_root / "private/TASK_QUEUE.json").read_text())["tasks"]
    p0_complete = sum(task["kind"] == "P0" and task.get("result_status") == "COMPLETE" for task in queue)
    trajectory_complete = sum((args.run_root / "private/tasks" / task["task_id"] / condition / "e2e_step0800.json").exists() for task in queue if task["kind"] == "P0" for condition in CONDITIONS)
    p1_complete = sum(task["kind"] == "P1" and task.get("result_status") == "COMPLETE" for task in queue)
    p2 = json.loads((args.run_root / "private/P2_DECISION.json").read_text()) if (args.run_root / "private/P2_DECISION.json").exists() else {"scheduled": False}
    gpu_seconds = sum(task.get("elapsed_seconds", 0.0) for task in queue if task["status"] in {"COMPLETE", "FAILED"})
    judge_sidecar = json.loads(args.sidecar.read_text())
    atomic_json(args.public_dir / "JUDGE_EXECUTION_AND_CLOSURE.json", {
        "status": "JUDGE_COMPLETE", "unique_tuples": len(judge_sidecar["tuples"]), "reused_exact_tuples": judge_sidecar["reused_count"],
        "new_judge_tuples": judge_sidecar["new_count"], "all_parse_valid": True, "private_mapping_withheld": True,
    })
    c1_positive = summary[CONDITIONS[0]]["positive"]["gated_correct_rate"]
    c3_positive = summary[CONDITIONS[2]]["positive"]["gated_correct_rate"]
    c1_hard = summary[CONDITIONS[0]]["hard"]["on_rate"]
    c3_hard = summary[CONDITIONS[2]]["hard"]["on_rate"]
    if c1_positive is None or c3_positive is None:
        retention = "EXECUTION_RETENTION_INCONCLUSIVE"
    else:
        retention = "EXECUTION_RETENTION_IMPROVED" if c3_positive > c1_positive else "EXECUTION_RETENTION_NOT_IMPROVED"
    if None in {c1_positive, c3_positive, c1_hard, c3_hard}:
        pareto = "ROUTING_EXECUTION_PARETO_INCONCLUSIVE"
    else:
        pareto = "ROUTING_EXECUTION_PARETO_IMPROVED" if c3_positive > c1_positive and c3_hard <= c1_hard else "ROUTING_EXECUTION_PARETO_NOT_IMPROVED"
    atomic_json(args.public_dir / "RUN_COMPLETION.json", {
        "training": "TRAINING_COMPLETE" if trajectory_complete == 108 else "PARTIAL_BUDGET",
        "generation": "GENERATION_COMPLETE" if trajectory_complete == 108 else "PARTIAL",
        "judge": "JUDGE_COMPLETE", "execution_retention": retention, "routing_execution_pareto": pareto,
        "publication": "RETRY_REQUIRED", "p0_matched_triples_complete": p0_complete, "p0_trajectories_complete": trajectory_complete,
        "p1_tasks_complete": p1_complete, "p2": p2, "gpu_hours": gpu_seconds / 3600, "gpu1_used": False,
    })
    lines = ["# Paired MedTRACE execution-preservation E2E summary", "", "Primary comparison: additional step 800, SAFETY_FIRST, viewed development panel.", ""]
    pct = lambda value: "NA" if value is None else f"{value:.1%}"
    for condition, values in summary.items():
        positive, hard, broad = values["positive"], values["hard"], values["broad"]
        lines.append(f"- `{condition}`: positive forced {positive['forced_correct_num']}/{positive['n']} (micro {pct(positive['forced_correct_rate'])}, edit macro {pct(positive['edit_macro']['forced_correct_rate'])}); gated {positive['gated_correct_num']}/{positive['n']} (micro {pct(positive['gated_correct_rate'])}, edit macro {pct(positive['edit_macro']['gated_correct_rate'])}); joint ON+correct micro {pct(positive['joint_on_and_correct_rate'])}, edit macro {pct(positive['edit_macro']['joint_on_and_correct_rate'])}; hard FPR {hard['on_num']}/{hard['n']} (micro {pct(hard['on_rate'])}, edit macro {pct(hard['edit_macro']['on_rate'])}); broad FPR {broad['on_num']}/{broad['n']} (micro {pct(broad['on_rate'])}, edit macro {pct(broad['edit_macro']['on_rate'])}).")
    lines.extend(["", f"Status: `{retention}`; `{pareto}`.", "", "Seeds repeat the same 12 facts; hard support is seven edits. These are DEV results, not blind or clinical validation."])
    atomic_text(args.public_dir / "PAIRED_E2E_SUMMARY.md", "\n".join(lines) + "\n")
    atomic_text(args.public_dir / "GPU_THROUGHPUT_AND_BUDGET_REPORT.md", f"# GPU throughput and budget\n\n- Accounted worker GPU time: {gpu_seconds / 3600:.3f} GPU-hours.\n- P0 matched triples complete: {p0_complete}/36; step800 trajectories with generation: {trajectory_complete}/108.\n- P1 task coverage: {p1_complete}/108.\n- P2 scheduled: {p2.get('scheduled', False)}.\n- GPU1 used: false.\n")
    atomic_text(args.public_dir / "GPT_PRO_REVIEW.md", "\n".join([
        "# GPT Pro review: MedTRACE execution-preservation long run", "", *lines[2:], "",
        "Review whether C1 isolates recovery budget, C2 changes the routing/execution tradeoff, and C3 adds positive execution retention without hiding ALL_OFF behavior. Check edit-level denominators and the separate formal T1L/T1G/T2G rows in the CSV. Do not interpret this viewed panel as blind qualification.",
    ]) + "\n")
    ledger = []
    for task in queue:
        if task["kind"] == "P0":
            for condition in CONDITIONS:
                ledger.append({
                    "phase": "P0", "seed": task["seed"], "event_index": task["event_index"],
                    "opaque_edit": hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], "condition": condition,
                    "status": "COMPLETE" if (args.run_root / "private/tasks" / task["task_id"] / condition / "e2e_step0800.json").exists() else task["status"],
                })
    atomic_text(args.public_dir / "TASK_LEDGER.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger))


def signal_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("run-root", "old-run", "fixed-run", "public-dir"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--base-commit", required=True)
    prepare_parser.set_defaults(func=prepare)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--run-root", type=Path, required=True)
    recover_parser.set_defaults(func=recover)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--old-run", type=Path, required=True)
    smoke_parser.add_argument("--cpu-gate", type=Path, required=True)
    smoke_parser.add_argument("--out", type=Path, required=True)
    smoke_parser.set_defaults(func=smoke)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--old-run", type=Path, required=True)
    worker_parser.add_argument("--cpu-gate", type=Path, required=True)
    worker_parser.add_argument("--expected-code-commit", required=True)
    worker_parser.set_defaults(func=worker)
    p2 = sub.add_parser("schedule-p2")
    p2.add_argument("--run-root", type=Path, required=True)
    p2.set_defaults(func=schedule_p2)
    judge = sub.add_parser("prepare-judge")
    judge.add_argument("--run-root", type=Path, required=True)
    judge.add_argument("--old-run", type=Path, required=True)
    judge.add_argument("--packet", type=Path, required=True)
    judge.add_argument("--sidecar", type=Path, required=True)
    judge.set_defaults(func=prepare_judge)
    final = sub.add_parser("finalize")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--judge-output", type=Path, required=True)
    final.add_argument("--sidecar", type=Path, required=True)
    final.add_argument("--public-dir", type=Path, required=True)
    final.set_defaults(func=finalize)
    return value


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_stop)
    options = parser().parse_args()
    options.func(options)
