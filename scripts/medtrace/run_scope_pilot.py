#!/usr/bin/env python3
"""Run the frozen single-expert scope pilot and finalize its judged aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook, calibrate_threshold  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402

METHOD = "MEDTRACE_NATIVE_TEXT_SCOPE_AUGMENTATION_PILOT"
SEED = 20260905


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def role_rows(roles: dict[str, Any], role: str) -> list[dict[str, Any]]:
    primary = roles["primary"]
    positives = [{
        "logical_id": f"positive-{role}-{index}", "label": "positive", "role": role,
        "question": row["question"], "reference": primary["target"], "image_path": primary["image_path"],
        "source_group": primary["image_name"], "fact_relation": "reviewed_same_fact_text_augmentation",
    } for index, row in enumerate(roles["positives"][role], 1)]
    negatives = [{
        "logical_id": f"negative-{role}-{index}", "label": "negative", "role": role,
        "question": row["question"], "reference": row["source_answer"], "image_path": row["image_path"],
        "source_group": row["image_name"], "fact_relation": row["fact_relation"],
    } for index, row in enumerate(roles["negatives"][role], 1)]
    return positives + negatives


def generate(runtime: Any, row: dict[str, Any], hook: MedTraceLayerHook | None, cap: int | None = None) -> dict[str, Any]:
    config = dict(runtime.generation_config)
    if cap is not None:
        config["max_new_tokens"] = cap
    batch = runtime.adapter.prepare_inputs(row["image_path"], row["question"], None)
    started = time.monotonic()
    if hook is None:
        result = runtime.adapter.generate_prepared_with_result(batch, config)
        lifecycle = ()
    else:
        with hook.generation_request():
            result = runtime.adapter.generate_prepared_with_result(batch, config)
        lifecycle = hook.last_generation_trace
    torch.cuda.synchronize()
    return {
        "raw_answer": result.decoded_text, "raw_token_ids": list(result.raw_token_ids),
        "generated_token_count": len(result.raw_token_ids),
        "reached_length_limit": len(result.raw_token_ids) >= int(config["max_new_tokens"]),
        "ended_with_eos": bool(result.raw_token_ids and result.raw_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id),
        "elapsed_seconds": time.monotonic() - started, "request_lifecycle": lifecycle,
    }


def extract(runtime: Any, record: EditorRecord, row: dict[str, Any], locks: dict[str, str]) -> tuple[torch.Tensor, dict[str, Any]]:
    batch = runtime.build_question_batch(record, question=row["question"], image_path=Path(row["image_path"]))
    activation = runtime.extract_layer_input_key(batch, module_path=LAYER, pooling="last_prompt").cpu()
    with Image.open(row["image_path"]) as image:
        image_size = list(image.size)
    attention = batch.attention_mask if batch.attention_mask is not None else torch.ones(batch.inputs_embeds.shape[:2], dtype=torch.long)
    eq_payload = {
        "image_tensor_sha256": batch.image_sha256,
        "image_tensor_shape": batch.image_tensor_shape,
        "image_sizes": image_size,
        "view_order": [0],
        "target_free_routing_input_ids": batch.raw_input_ids[0].detach().cpu().tolist(),
        "attention_mask": attention[0].detach().cpu().tolist(),
        "assistant_predictor_index": batch.key_token_index,
        **locks,
    }
    return activation, {
        "logical_id": row["logical_id"], "role": row["role"], "label": row["label"],
        "source_group_sha256": hashlib.sha256(row["source_group"].encode()).hexdigest(),
        "fact_relation": row["fact_relation"], "eqkey": sha256_json(eq_payload),
        "activation_sha256": tensor_sha256(activation),
    }


def score(expert: AsymmetricCPExpert, values: torch.Tensor) -> torch.Tensor:
    return expert.intrinsic_score(values)


def validate_eq_rows(rows: list[dict[str, Any]], expected: int = 72) -> None:
    if len(rows) != expected or len({row["eqkey"] for row in rows}) != expected:
        raise RuntimeError("EqKey coverage or uniqueness failure")
    for eqkey in {row["eqkey"] for row in rows}:
        related = [row for row in rows if row["eqkey"] == eqkey]
        if len({row["role"] for row in related}) > 1 or len({row["label"] for row in related}) > 1:
            raise RuntimeError("EqKey role or label conflict")


def train_input(expert: AsymmetricCPExpert, positive: torch.Tensor, negative: torch.Tensor) -> list[dict[str, float | int]]:
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    expert.u_in.requires_grad_(True)
    expert.v_in.requires_grad_(True)
    optimizer = torch.optim.AdamW([expert.u_in, expert.v_in], lr=1e-3, weight_decay=0)
    trajectory = []
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        positive_score, negative_score = score(expert, positive), score(expert, negative)
        logits = torch.cat((positive_score[:, None], negative_score.expand(len(positive_score), -1)), dim=1) / 0.1
        infonce = F.cross_entropy(logits, torch.zeros(len(positive_score), dtype=torch.long, device=logits.device))
        hinge = F.relu(0.1 + negative_score.max() - positive_score.min())
        q = expert.input_basis().float()
        orthogonality = (q.T @ q - torch.eye(expert.rank, device=q.device)).square().mean()
        loss = infonce + hinge + 0.01 * orthogonality
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([expert.u_in, expert.v_in], 1.0).item())
        if not math.isfinite(float(loss.item())) or not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite scope-input optimization")
        optimizer.step()
        expert.normalize_input_factors_()
        if step % 20 == 0:
            trajectory.append({"step": step, "loss": float(loss.item()), "infonce": float(infonce.item()), "hinge": float(hinge.item()), "orthogonality": float(orthogonality.item()), "gradient_norm": grad_norm})
    return trajectory


def train_output(runtime: Any, expert: AsymmetricCPExpert, record: EditorRecord, q_hash: str) -> tuple[list[dict[str, Any]], int]:
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    for parameter in (expert.u_out, expert.v_out, expert.rho):
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([expert.u_out, expert.v_out, expert.rho], lr=1e-3, weight_decay=0)
    batch = runtime.build_edit_batch(record)
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
    hook.attach()
    row = {"question": record.question, "image_path": str(record.image_path)}
    trajectory, selected = [], 200
    try:
        for step in range(201):
            if step:
                hook.set_teacher_routing(batch.labels)
                optimizer.zero_grad(set_to_none=True)
                loss = runtime.compute_loss(batch)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_([expert.u_out, expert.v_out, expert.rho], 1.0).item())
                if not math.isfinite(float(loss.item())) or not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite scope-output optimization")
                optimizer.step()
                expert.normalize_output_factors_()
                if tensor_sha256(expert.input_basis()) != q_hash:
                    raise RuntimeError("scope output recovery changed frozen Q")
            if step % 20:
                continue
            hook.set_teacher_routing(batch.labels)
            with torch.no_grad():
                target = runtime.score_target(batch)
            generated = generate(runtime, row, hook, 128)
            literal = normalize_medical_answer(generated["raw_answer"]) == normalize_medical_answer(record.target)
            trajectory.append({"step": step, "target": target, "literal_normalized_target_match": literal, "reached_length_limit": generated["reached_length_limit"]})
            if literal and target["first_target_token_rank"] == 1 and not generated["reached_length_limit"]:
                selected = step
                break
    finally:
        hook.detach()
    return trajectory, selected


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("scope runner code commit mismatch")
    roles = json.loads(args.roles.read_text())
    if roles["status"] != "EXPLORATORY_EVALUABLE__EQKEY_PENDING":
        raise RuntimeError("scope role freeze is not executable")
    dev = read_jsonl(args.dev_inputs)
    event = next(row for row in dev if row["edit_record"]["record_id"] == roles["primary"]["record_id"])
    record = EditorRecord.from_dict(event["edit_record"])
    runtime = load_real_runtime(args)
    layer = runtime.get_module(LAYER)
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    if int(checkpoint["rank"]) != 4 or int(checkpoint["step"]) != 140:
        raise RuntimeError("historical scope checkpoint is not rank-4 step-140")
    original = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    original.load_state_dict(checkpoint["expert"])
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"])
    seed_everything(SEED)
    args.out.mkdir(parents=True)
    locks = {
        "runtime_lock_sha256": sha256_file(args.runtime_lock),
        "tokenizer_config_sha256": sha256_file(Path(os.environ["M3BENCH_MODEL_PATH"]) / "tokenizer_config.json"),
        "processor_config_sha256": sha256_file(Path(os.environ["M3BENCH_VISION_PATH"]) / "preprocessor_config.json"),
    }
    tensors: dict[tuple[str, str], torch.Tensor] = {}
    eq_rows = []
    for role in ("fit", "calibration", "evaluation"):
        for row in role_rows(roles, role):
            activation, eq = extract(runtime, record, row, locks)
            tensors[(role, row["logical_id"])] = activation
            eq_rows.append(eq)
    validate_eq_rows(eq_rows)
    def matrix(role: str, label: str) -> torch.Tensor:
        ids = [row["logical_id"] for row in role_rows(roles, role) if row["label"] == label]
        return torch.stack([tensors[(role, item)] for item in ids]).to("cuda:0")
    original_cal_positive, original_cal_negative = score(original, matrix("calibration", "positive")), score(original, matrix("calibration", "negative"))
    control_calibration = calibrate_threshold(original_cal_positive.tolist(), original_cal_negative.tolist(), target_fpr=0.0)
    input_trajectory = train_input(expert, matrix("fit", "positive"), matrix("fit", "negative"))
    final_cal_positive, final_cal_negative = score(expert, matrix("calibration", "positive")), score(expert, matrix("calibration", "negative"))
    final_calibration = calibrate_threshold(final_cal_positive.tolist(), final_cal_negative.tolist(), target_fpr=0.0)
    q_hash = tensor_sha256(expert.input_basis())
    output_trajectory, output_step = train_output(runtime, expert, record, q_hash)
    if tensor_sha256(expert.input_basis()) != q_hash:
        raise RuntimeError("final Q changed during output recovery")
    eval_rows = role_rows(roles, "evaluation")
    evaluations, packet, sidecar = [], [], []
    for row in eval_rows:
        activation = tensors[("evaluation", row["logical_id"])].to("cuda:0")
        control_score = float(score(original, activation[None])[0].item())
        final_score = float(score(expert, activation[None])[0].item())
        control_on = control_score > control_calibration.threshold
        final_on = final_score > final_calibration.threshold
        base = generate(runtime, row, None)
        control_hook = MedTraceLayerHook(layer, original)
        if control_on:
            control_hook.attach()
            try:
                control = generate(runtime, row, control_hook)
            finally:
                control_hook.detach()
        else:
            control = generate(runtime, row, None)
        final_hook = MedTraceLayerHook(layer, expert)
        final_hook.attach()
        try:
            forced = generate(runtime, row, final_hook)
            gated = generate(runtime, row, final_hook) if final_on else None
        finally:
            final_hook.detach()
        if gated is None:
            gated = generate(runtime, row, None)
        outputs = {"base": base, "original_q_threshold_control": control, "final_forced_on": forced, "final_intrinsic_gated": gated}
        item = {**{k: row[k] for k in ("logical_id", "label", "fact_relation")}, "control_score": control_score, "control_on": control_on, "final_score": final_score, "final_on": final_on, "outputs": outputs}
        item["control_off_exact_base"] = control_on or (control["raw_token_ids"] == base["raw_token_ids"] and control["raw_answer"] == base["raw_answer"])
        item["final_off_exact_base"] = final_on or (gated["raw_token_ids"] == base["raw_token_ids"] and gated["raw_answer"] == base["raw_answer"])
        evaluations.append(item)
        for path, output in outputs.items():
            opaque = sha256_json((row["logical_id"], path, sha256_file(args.checkpoint), actual_commit))
            packet.append({"opaque_query_id": opaque, "question": row["question"], "gold_answer": row["reference"], "raw_base_answer": output["raw_answer"], "adjudication_pass": 1})
            sidecar.append({"opaque_query_id": opaque, "logical_id": row["logical_id"], "label": row["label"], "path": path})
    native_row = {"logical_id": "native", "question": record.question, "reference": record.target, "image_path": str(record.image_path), "source_group": roles["primary"]["image_name"], "fact_relation": "native"}
    native_batch = runtime.build_question_batch(record)
    native_activation = runtime.extract_layer_input_key(native_batch, module_path=LAYER, pooling="last_prompt").to("cuda:0")
    native_score = float(score(expert, native_activation[None])[0].item())
    native_on = native_score > final_calibration.threshold
    native_hook = MedTraceLayerHook(layer, expert)
    native_hook.attach()
    try:
        native_forced = generate(runtime, native_row, native_hook)
        native_gated = generate(runtime, native_row, native_hook) if native_on else None
    finally:
        native_hook.detach()
    if native_gated is None:
        native_gated = generate(runtime, native_row, None)
    guard = runtime.base_guard.verify() if runtime.base_guard else None
    if not guard or not guard["unchanged"] or not all(row["control_off_exact_base"] and row["final_off_exact_base"] for row in evaluations):
        raise RuntimeError("base guard or OFF parity failed")
    torch.save({"rank": 4, "step": output_step, "expert": expert.state_dict(), "threshold": final_calibration.threshold, "q_sha256": q_hash}, args.out / "final_scope_expert.pt")
    atomic_json(args.out / "result_private.json", {
        "schema_version": "medtrace-scope-pilot-private-v1", "status": "SCOPE_MODEL_RUN_COMPLETE__JUDGE_PENDING",
        "method": METHOD, "execution_mode": "EXPLORATORY_EVALUABLE", "code_commit": actual_commit,
        "historical_checkpoint_sha256": sha256_file(args.checkpoint), "roles_sha256": sha256_file(args.roles),
        "eqkey": {"count": len(eq_rows), "unique": len({row['eqkey'] for row in eq_rows}), "rows": eq_rows, **locks},
        "fit": {"steps": 200, "trajectory": input_trajectory},
        "control_calibration": control_calibration.__dict__, "final_calibration": final_calibration.__dict__,
        "output_recovery": {"selected_step": output_step, "trajectory": output_trajectory, "q_sha256_before_and_after": q_hash},
        "native": {"score": native_score, "on": native_on, "forced": native_forced, "gated": native_gated, "exact_forced": normalize_medical_answer(native_forced['raw_answer']) == normalize_medical_answer(record.target), "exact_gated": normalize_medical_answer(native_gated['raw_answer']) == normalize_medical_answer(record.target)},
        "evaluation": evaluations, "base_guard": guard,
    })
    atomic_text(args.out / "judge_packet_private.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet))
    atomic_json(args.out / "judge_sidecar_private.json", sidecar)


def finalize(args: argparse.Namespace) -> None:
    result = json.loads((args.private / "result_private.json").read_text())
    verdicts = {row["opaque_query_id"]: row for row in read_jsonl(args.judge_output)}
    sidecar = json.loads((args.private / "judge_sidecar_private.json").read_text())
    expected_ids = {row["opaque_query_id"] for row in sidecar}
    if len(sidecar) != 96 or set(verdicts) != expected_ids or any(not verdicts[query_id]["parse_valid"] for query_id in expected_ids):
        raise RuntimeError("scope Judge coverage is incomplete")
    judged = {(row["logical_id"], row["path"]): bool(verdicts[row["opaque_query_id"]]["is_correct"]) for row in sidecar}
    rows = result["evaluation"]
    def selected(label: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["label"] == label]
    positives, negatives = selected("positive"), selected("negative")
    base_correct_negatives = [row for row in negatives if judged[(row["logical_id"], "base")]]
    metrics = {
        "schema_version": "medtrace-scope-pilot-public-metrics-v1", "status": "SCOPE_PILOT_EVALUATION_COMPLETE",
        "execution_mode": "EXPLORATORY_EVALUABLE", "v02_scope_qualified": False,
        "role_counts": {"fit": {"positive": 4, "negative": 20}, "calibration": {"positive": 4, "negative": 20}, "evaluation": {"positive": 4, "negative": 20}},
        "control": {
            "threshold": result["control_calibration"]["threshold"],
            "calibration_tpr": result["control_calibration"]["true_positive_rate"], "calibration_fpr": result["control_calibration"]["false_positive_rate"],
            "evaluation_positive_activation": sum(row["control_on"] for row in positives) / len(positives),
            "evaluation_negative_fpr": sum(row["control_on"] for row in negatives) / len(negatives),
        },
        "final": {
            "threshold": result["final_calibration"]["threshold"],
            "calibration_tpr": result["final_calibration"]["true_positive_rate"], "calibration_fpr": result["final_calibration"]["false_positive_rate"],
            "evaluation_positive_activation": sum(row["final_on"] for row in positives) / len(positives),
            "evaluation_negative_fpr": sum(row["final_on"] for row in negatives) / len(negatives),
            "positive_base_correct": sum(judged[(row["logical_id"], "base")] for row in positives),
            "positive_forced_correct": sum(judged[(row["logical_id"], "final_forced_on")] for row in positives),
            "positive_gated_correct": sum(judged[(row["logical_id"], "final_intrinsic_gated")] for row in positives),
            "base_correct_negative_count": len(base_correct_negatives),
            "base_correct_negative_preserved": sum(judged[(row["logical_id"], "final_intrinsic_gated")] for row in base_correct_negatives),
            "all_negative_behavior_exact_base": sum(row["outputs"]["base"]["raw_token_ids"] == row["outputs"]["final_intrinsic_gated"]["raw_token_ids"] for row in negatives),
            "off_request_count": sum(not row["final_on"] for row in rows),
            "off_token_parity": sum((not row["final_on"]) and row["final_off_exact_base"] for row in rows),
            "activated_negative_damage": sum(row["final_on"] and judged[(row["logical_id"], "base")] and not judged[(row["logical_id"], "final_intrinsic_gated")] for row in negatives),
        },
        "native": {key: result["native"][key] for key in ("on", "exact_forced", "exact_gated")},
        "limitations": ["single image and single fact for all positives", "reviewed text augmentation only", "broad source negatives may dominate", "no patient-disjoint claim", "not V0.2 qualified"],
    }
    atomic_json(args.public / "SCOPE_METRICS.json", metrics)
    atomic_text(args.public / "SCOPE_PILOT_RESULT.md", f"""# MedTRACE scope pilot

Status: `SCOPE_PILOT_EVALUATION_COMPLETE`

Execution mode is `EXPLORATORY_EVALUABLE`; this is a single-image, single-fact text-augmentation pilot and is not V0.2 qualified.

- Roles: fit 4 positive/20 negative; calibration 4/20; evaluation 4/20.
- Original-Q control: threshold `{metrics['control']['threshold']:.8g}`, calibration TPR/FPR `{metrics['control']['calibration_tpr']:.3f}/{metrics['control']['calibration_fpr']:.3f}`, evaluation positive activation `{metrics['control']['evaluation_positive_activation']:.3f}`, negative FPR `{metrics['control']['evaluation_negative_fpr']:.3f}`.
- Final Q: threshold `{metrics['final']['threshold']:.8g}`, calibration TPR/FPR `{metrics['final']['calibration_tpr']:.3f}/{metrics['final']['calibration_fpr']:.3f}`, evaluation positive activation `{metrics['final']['evaluation_positive_activation']:.3f}`, negative FPR `{metrics['final']['evaluation_negative_fpr']:.3f}`.
- New positives semantic correctness: Base `{metrics['final']['positive_base_correct']}/4`, forced-on `{metrics['final']['positive_forced_correct']}/4`, gated `{metrics['final']['positive_gated_correct']}/4`.
- Base-correct negative preservation: `{metrics['final']['base_correct_negative_preserved']}/{metrics['final']['base_correct_negative_count']}`.
- All-negative exact behavior preservation: `{metrics['final']['all_negative_behavior_exact_base']}/20`.
- OFF token parity: `{metrics['final']['off_token_parity']}/{metrics['final']['off_request_count']}`.
- Activated-negative semantic damage: `{metrics['final']['activated_negative_damage']}/20`.
- Native gate ON/correct: `{metrics['native']['on']}/{metrics['native']['exact_gated']}`.

Full raw QA, answers, images, activations, EqKeys, checkpoint and Judge mapping remain private.
""")
    result["status"] = "SCOPE_PILOT_EVALUATION_COMPLETE"
    atomic_json(args.private / "result_private.json", result)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cpu-gate", type=Path, required=True)
    run_parser.add_argument("--dev-inputs", type=Path, required=True)
    run_parser.add_argument("--roles", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--runtime-lock", type=Path, required=True)
    run_parser.add_argument("--expected-code-commit", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.set_defaults(func=run)
    final_parser = sub.add_parser("finalize")
    final_parser.add_argument("--private", type=Path, required=True)
    final_parser.add_argument("--judge-output", type=Path, required=True)
    final_parser.add_argument("--public", type=Path, required=True)
    final_parser.set_defaults(func=finalize)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
