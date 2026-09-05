#!/usr/bin/env python3
"""Run B0/B1/B2 scope ablation with matched output fitting and hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook, calibrate_threshold  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402
from scripts.medtrace.run_scope_pilot import atomic_json, atomic_text, extract, generate, score, sha256_file, sha256_json, train_input  # noqa: E402

CONDITIONS = ("ORIGINAL_Q_MATCHED_OUTPUT_FIT", "BROAD_Q_MATCHED_OUTPUT_FIT", "HARD_MIXED_Q_MATCHED_OUTPUT_FIT")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def positive_rows(scope: dict[str, Any], role: str) -> list[dict[str, Any]]:
    primary = scope["primary"]
    return [{
        "logical_id": f"positive-{role}-{index}", "role": role, "label": "positive",
        "question": row["question"], "reference": primary["target"], "image_path": primary["image_path"],
        "source_group": primary["image_name"], "fact_relation": "reviewed_same_fact_text_augmentation",
    } for index, row in enumerate(scope["positives"][role], 1)]


def negative_rows(scope: dict[str, Any], key: str, role: str) -> list[dict[str, Any]]:
    return [{
        "logical_id": f"negative-{role}-{row['source_qid']}", "role": role, "label": "negative",
        "question": row["question"], "reference": row["source_answer"], "image_path": row["image_path"],
        "source_group": row["image_name"], "source_qid": row["source_qid"], "fact_relation": row["fact_relation"],
    } for row in scope["negative_roles"][key]]


def validate_eqkeys(rows: list[dict[str, Any]], eq_rows: list[dict[str, Any]]) -> None:
    if len(eq_rows) != len(rows) or len({row["eqkey"] for row in eq_rows}) != len(eq_rows):
        raise RuntimeError("scope EqKey coverage or uniqueness failure")
    by_id = {row["logical_id"]: row for row in eq_rows}
    if set(by_id) != {row["logical_id"] for row in rows}:
        raise RuntimeError("scope EqKey logical-ID coverage failure")
    for row in rows:
        eq = by_id[row["logical_id"]]
        if eq["role"] != row["role"] or eq["label"] != row["label"]:
            raise RuntimeError("scope EqKey role or label conflict")


def fixed_output_fit(runtime: Any, expert: AsymmetricCPExpert, record: EditorRecord, questions: list[str], q_hash: str, threshold: float) -> dict[str, Any]:
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    for parameter in (expert.u_out, expert.v_out, expert.rho):
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([expert.u_out, expert.v_out, expert.rho], lr=1e-3, weight_decay=0)
    native = runtime.build_edit_batch(record)
    paraphrases = [runtime.build_edit_batch(replace(record, question=question)) for question in questions]
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
    hook.attach()
    trajectory, sequence_tokens, target_tokens = [], 0, 0
    started = time.monotonic()
    try:
        for step in range(1, 81):
            optimizer.zero_grad(set_to_none=True)
            batches = (native, paraphrases[(step - 1) % len(paraphrases)])
            losses = []
            for batch in batches:
                hook.set_teacher_routing(batch.labels)
                loss = runtime.compute_loss(batch)
                (0.5 * loss).backward()
                losses.append(float(loss.item()))
                sequence_tokens += int(batch.inputs_embeds.shape[1])
                target_tokens += int((batch.labels != -100).sum().item())
            grad_norm = float(torch.nn.utils.clip_grad_norm_([expert.u_out, expert.v_out, expert.rho], 1.0).item())
            if not math.isfinite(grad_norm) or any(not math.isfinite(value) for value in losses):
                raise FloatingPointError("non-finite matched output fit")
            optimizer.step()
            expert.normalize_output_factors_()
            if tensor_sha256(expert.input_basis()) != q_hash:
                raise RuntimeError("matched output fit changed calibrated Q")
            if step in (40, 80):
                trajectory.append({"step": step, "micro_losses": losses, "gradient_norm": grad_norm})
    finally:
        hook.detach()
    if tensor_sha256(expert.input_basis()) != q_hash or threshold != threshold:
        raise RuntimeError("Q or threshold changed during output fit")
    return {
        "steps": 80,
        "micro_forwards": 160,
        "sequence_token_proxy": sequence_tokens,
        "supervised_target_tokens": target_tokens,
        "trajectory": trajectory,
        "elapsed_seconds": time.monotonic() - started,
        "q_sha256_before_and_after": q_hash,
        "threshold_sha256_before_and_after": sha256_json(threshold),
    }


def summarize_lifecycle(output: dict[str, Any]) -> dict[str, Any]:
    trace = output["request_lifecycle"]
    return {
        "hook_executed": bool(trace),
        "active_predictor_steps": len(trace),
        "max_active_residual_norm": max((row.get("active_residual_norm", 0.0) for row in trace), default=0.0),
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("hard-scope runner code commit mismatch")
    frozen = json.loads(args.data.read_text())
    scope = frozen["scope"]
    dev = read_jsonl(args.dev_inputs)
    event = next(row for row in dev if row["edit_record"]["record_id"] == scope["primary"]["record_id"])
    record = EditorRecord.from_dict(event["edit_record"])
    runtime = load_real_runtime(args)
    layer = runtime.get_module(LAYER)
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    if int(checkpoint["rank"]) != 4 or int(checkpoint["step"]) != 140:
        raise RuntimeError("scope checkpoint is not the frozen rank-4 step-140 artifact")
    args.out.mkdir(parents=True)
    locks = {
        "runtime_lock_sha256": sha256_file(args.runtime_lock),
        "tokenizer_config_sha256": sha256_file(Path(os.environ["M3BENCH_MODEL_PATH"]) / "tokenizer_config.json"),
        "processor_config_sha256": sha256_file(Path(os.environ["M3BENCH_VISION_PATH"]) / "preprocessor_config.json"),
    }
    rows_by_role = {
        "fit_positive": positive_rows(scope, "fit"),
        "calibration_positive": positive_rows(scope, "calibration"),
        "evaluation_positive": positive_rows(scope, "evaluation"),
        "broad_fit": negative_rows(scope, "broad_fit_control", "fit"),
        "mixed_fit": negative_rows(scope, "mixed_fit", "fit"),
        "calibration_negative": negative_rows(scope, "calibration", "calibration"),
        "evaluation_negative": negative_rows(scope, "evaluation", "evaluation"),
        "same_image_challenge": negative_rows(scope, "same_image_challenge", "challenge"),
    }
    unique_rows: dict[str, dict[str, Any]] = {}
    for rows in rows_by_role.values():
        for row in rows:
            previous = unique_rows.setdefault(row["logical_id"], row)
            if previous != row:
                raise RuntimeError("logical scope row has conflicting payloads")
    tensors, eq_rows = {}, []
    for row in unique_rows.values():
        activation, eq = extract(runtime, record, row, locks)
        tensors[row["logical_id"]] = activation
        eq_rows.append(eq)
    validate_eqkeys(list(unique_rows.values()), eq_rows)

    def matrix(key: str) -> torch.Tensor:
        return torch.stack([tensors[row["logical_id"]] for row in rows_by_role[key]]).to("cuda:0")

    condition_experts, condition_meta = {}, {}
    initial_hashes = []
    seed_everything(20260906)
    for condition in CONDITIONS:
        expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
        expert.load_state_dict(checkpoint["expert"])
        initial_hashes.append(sha256_json({name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in expert.state_dict().items()}))
        torch.cuda.reset_peak_memory_stats()
        if condition == "BROAD_Q_MATCHED_OUTPUT_FIT":
            input_trajectory = train_input(expert, matrix("fit_positive"), matrix("broad_fit"))
        elif condition == "HARD_MIXED_Q_MATCHED_OUTPUT_FIT":
            input_trajectory = train_input(expert, matrix("fit_positive"), matrix("mixed_fit"))
        else:
            input_trajectory = []
        cal_positive, cal_negative = score(expert, matrix("calibration_positive")), score(expert, matrix("calibration_negative"))
        calibration = calibrate_threshold(cal_positive.tolist(), cal_negative.tolist(), target_fpr=0.0)
        q_hash = tensor_sha256(expert.input_basis())
        output_fit = fixed_output_fit(runtime, expert, record, [row["question"] for row in frozen["generality_paraphrases"][record.record_id]], q_hash, calibration.threshold)
        condition_experts[condition] = expert
        condition_meta[condition] = {
            "input_steps": 0 if condition == CONDITIONS[0] else 200,
            "input_trajectory": input_trajectory,
            "calibration": calibration.__dict__,
            "q_sha256": q_hash,
            "output_fit": output_fit,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
    if len(set(initial_hashes)) != 1:
        raise RuntimeError("scope conditions did not share the same initial checkpoint")

    evaluation_rows = rows_by_role["evaluation_positive"] + rows_by_role["evaluation_negative"] + rows_by_role["same_image_challenge"]
    evaluations, packet, sidecar = [], [], []
    for row in evaluation_rows:
        activation = tensors[row["logical_id"]].to("cuda:0")
        base = generate(runtime, row, None)
        outputs = {"base": base}
        decisions = {}
        for condition in CONDITIONS:
            expert = condition_experts[condition]
            meta = condition_meta[condition]
            value = float(score(expert, activation[None])[0].item())
            on = value > meta["calibration"]["threshold"]
            hook = MedTraceLayerHook(layer, expert)
            hook.attach()
            try:
                forced = generate(runtime, row, hook)
                gated = generate(runtime, row, hook) if on else dict(base)
            finally:
                hook.detach()
            outputs[f"{condition}__forced"] = forced
            outputs[f"{condition}__gated"] = gated
            decisions[condition] = {
                "score": value,
                "on": on,
                "forced": summarize_lifecycle(forced),
                "gated": summarize_lifecycle(gated),
                "gated_token_exact_base": gated["raw_token_ids"] == base["raw_token_ids"] and gated["raw_answer"] == base["raw_answer"],
            }
        evaluations.append({
            "logical_id": row["logical_id"], "role": row["role"], "label": row["label"], "fact_relation": row["fact_relation"],
            "source_qid": row.get("source_qid"), "outputs": outputs, "decisions": decisions,
        })
        for path, output in outputs.items():
            opaque = sha256_json((row["logical_id"], path, actual_commit, sha256_file(args.checkpoint)))
            packet.append({"opaque_query_id": opaque, "question": row["question"], "gold_answer": row["reference"], "raw_base_answer": output["raw_answer"], "adjudication_pass": 1})
            sidecar.append({"opaque_query_id": opaque, "logical_id": row["logical_id"], "role": row["role"], "label": row["label"], "fact_relation": row["fact_relation"], "path": path})

    native = {"logical_id": "native", "question": record.question, "reference": record.target, "image_path": str(record.image_path)}
    native_batch = runtime.build_question_batch(record)
    native_activation = runtime.extract_layer_input_key(native_batch, module_path=LAYER, pooling="last_prompt").to("cuda:0")
    native_outputs, native_decisions = {}, {}
    for condition in CONDITIONS:
        expert = condition_experts[condition]
        meta = condition_meta[condition]
        value = float(score(expert, native_activation[None])[0].item())
        on = value > meta["calibration"]["threshold"]
        hook = MedTraceLayerHook(layer, expert)
        hook.attach()
        try:
            forced = generate(runtime, native, hook)
            gated = generate(runtime, native, hook) if on else generate(runtime, native, None)
        finally:
            hook.detach()
        native_outputs[f"{condition}__forced"] = forced
        native_outputs[f"{condition}__gated"] = gated
        native_decisions[condition] = {"score": value, "on": on, "forced": summarize_lifecycle(forced), "gated": summarize_lifecycle(gated)}
        for path, output in ((f"{condition}__forced", forced), (f"{condition}__gated", gated)):
            opaque = sha256_json(("native", path, actual_commit, sha256_file(args.checkpoint)))
            packet.append({"opaque_query_id": opaque, "question": native["question"], "gold_answer": native["reference"], "raw_base_answer": output["raw_answer"], "adjudication_pass": 1})
            sidecar.append({"opaque_query_id": opaque, "logical_id": "native", "role": "native", "label": "positive", "fact_relation": "native", "path": path})

    guard = runtime.base_guard.verify() if runtime.base_guard else None
    if not guard or not guard["unchanged"]:
        raise RuntimeError("hard-scope base guard failed")
    for condition, expert in condition_experts.items():
        path = args.out / f"{condition}.pt"
        torch.save({"rank": 4, "input_steps": condition_meta[condition]["input_steps"], "output_steps": 80, "condition": condition, "threshold": condition_meta[condition]["calibration"]["threshold"], "expert": expert.state_dict()}, path)
        restored = torch.load(path, map_location="cuda:0", weights_only=True)
        if restored["threshold"] != condition_meta[condition]["calibration"]["threshold"] or any(not torch.equal(left, right) for left, right in zip(expert.state_dict().values(), restored["expert"].values(), strict=True)):
            raise RuntimeError("hard-scope checkpoint reload mismatch")
    if len(packet) != 258 or len(sidecar) != 258:
        raise RuntimeError(f"hard-scope Judge coverage mismatch: {len(packet)}")
    atomic_json(args.out / "result_private.json", {
        "schema_version": "medtrace-hard-scope-ablation-private-v1",
        "status": "TRAINING_AND_GENERATION_COMPLETE__JUDGE_PENDING",
        "code_commit": actual_commit,
        "historical_checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "eqkey": {"count": len(eq_rows), "unique": len({row["eqkey"] for row in eq_rows}), "rows": eq_rows, **locks},
        "conditions": condition_meta,
        "evaluations": evaluations,
        "native": {"outputs": native_outputs, "decisions": native_decisions},
        "base_guard": guard,
    })
    atomic_text(args.out / "judge_packet_private.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet))
    atomic_json(args.out / "judge_sidecar_private.json", sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-gate", type=Path, required=True)
    parser.add_argument("--dev-inputs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
