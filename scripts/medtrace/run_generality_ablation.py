#!/usr/bin/env python3
"""Run paired 80-step native-only versus native-plus-paraphrase CP continuation."""

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
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer  # noqa: E402
from scripts.medtrace.run_dev16 import (  # noqa: E402
    atomic_json,
    derive_seed,
    evaluation_rows,
    generate,
    optimizer_lock,
    read_jsonl,
    save_checkpoint,
    sha256_file,
    sha256_json,
    validate_dev_rows,
)
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402

CONDITIONS = ("CP_NATIVE_CONTINUE_80", "CP_NATIVE_PLUS_PARAPHRASE_80")


def micro_plan(condition: str, step: int, paraphrase_count: int = 4) -> tuple[int, int]:
    if condition == CONDITIONS[0]:
        return 0, 0
    if condition == CONDITIONS[1] and 1 <= step <= 80 and paraphrase_count > 0:
        return 0, 1 + (step - 1) % paraphrase_count
    raise ValueError("invalid paired micro-forward schedule")


def batch_mask(batch: Any) -> torch.Tensor:
    mask = torch.zeros_like(batch.labels, dtype=torch.bool)
    mask[:, :-1] = batch.labels[:, 1:] != -100
    return mask


def diagnostics(runtime: Any, hook: MedTraceLayerHook, batches: list[Any], step: int) -> dict[str, Any]:
    values = []
    for batch in batches:
        hook.set_teacher_routing(batch.labels)
        with torch.no_grad():
            values.append(runtime.score_target(batch))
    return {"step": step, "native": values[0], "paraphrases": values[1:]}


def train_condition(runtime: Any, record: EditorRecord, questions: list[str], checkpoint: dict[str, Any], condition: str) -> tuple[AsymmetricCPExpert, dict[str, Any]]:
    layer = runtime.get_module(LAYER)
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"])
    start_state = {name: value.detach().cpu().clone() for name, value in expert.state_dict().items()}
    seed = derive_seed(record.record_id, base=20260906)
    seed_everything(seed)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3, weight_decay=0)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in expert.parameters()} or any(id(parameter) in optimized for parameter in runtime.model.parameters()):
        raise RuntimeError("paired optimizer boundary failure")
    native = runtime.build_edit_batch(record)
    paraphrases = [runtime.build_edit_batch(replace(record, question=question)) for question in questions]
    batches = [native, *paraphrases]
    masks = [batch_mask(batch) for batch in batches]
    hook = MedTraceLayerHook(layer, expert)
    hook.attach()
    trajectory, micro_forwards, sequence_tokens, target_tokens = [], 0, 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        trajectory.append(diagnostics(runtime, hook, batches, 0))
        for step in range(1, 81):
            optimizer.zero_grad(set_to_none=True)
            indices = micro_plan(condition, step, len(paraphrases))
            selected = tuple((batches[index], masks[index]) for index in indices)
            losses = []
            for batch, expected_mask in selected:
                hook.set_teacher_routing(batch.labels)
                if not torch.equal(hook.token_mask, expected_mask):
                    raise RuntimeError("paired assistant predictor mask mismatch")
                loss = runtime.compute_loss(batch)
                (0.5 * loss).backward()
                losses.append(float(loss.item()))
                micro_forwards += 1
                sequence_tokens += int(batch.inputs_embeds.shape[1])
                target_tokens += int((batch.labels != -100).sum().item())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0).item())
            if not math.isfinite(grad_norm) or any(not math.isfinite(value) for value in losses):
                raise FloatingPointError("non-finite paired CP continuation")
            optimizer.step()
            expert.normalize_factors_()
            if step in (40, 80):
                row = diagnostics(runtime, hook, batches, step)
                row |= {"micro_losses": losses, "gradient_norm": grad_norm}
                trajectory.append(row)
    finally:
        hook.detach()
    if hook.enabled or hook.generation_routing or hook.token_mask is not None:
        raise RuntimeError("paired hook state survived detach")
    if micro_forwards != 160:
        raise RuntimeError("paired condition did not execute 160 micro-forwards")
    return expert, {
        "condition": condition,
        "seed": seed,
        "optimizer": optimizer_lock(optimizer),
        "optimizer_steps": 80,
        "micro_forwards": micro_forwards,
        "sequence_token_proxy": sequence_tokens,
        "supervised_target_tokens": target_tokens,
        "trajectory": trajectory,
        "training_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "start_state_sha256": sha256_json({name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in start_state.items()}),
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("generality runner code commit mismatch")
    dev = read_jsonl(args.dev_inputs)
    validate_dev_rows(dev)
    frozen = json.loads(args.data.read_text())
    paraphrases = frozen["generality_paraphrases"]
    if set(paraphrases) != {event["edit_record"]["record_id"] for event in dev}:
        raise RuntimeError("frozen paraphrases do not cover DEV16")
    base = {row["query_id"]: row for row in read_jsonl(args.base_predictions)}
    needed = {item["query_id"] for event in dev for item in evaluation_rows(event)}
    if not needed <= base.keys():
        raise RuntimeError("base predictions do not cover DEV16")
    args.out.mkdir(parents=True)
    atomic_json(args.out / "run_lock.json", {
        "schema_version": "medtrace-generality-paired-run-lock-v1",
        "status": "RUNNING_STARTUP_SNAPSHOT",
        "code_commit": actual_commit,
        "conditions": list(CONDITIONS),
        "events": 16,
        "steps_per_condition": 80,
        "micro_forwards_per_step": 2,
        "checkpoint_root_sha256": sha256_file(args.old_run / "run_lock.json"),
        "data_sha256": sha256_file(args.data),
    })
    runtime = load_real_runtime(args)
    all_predictions, event_results = [], []
    try:
        for index, event in enumerate(dev, 1):
            record = EditorRecord.from_dict(event["edit_record"])
            old_dir = args.old_run / "events" / f"{index:02d}"
            old_result = json.loads((old_dir / "result.json").read_text())
            checkpoint_path = old_dir / "expert.pt"
            if old_result["record_id"] != record.record_id or sha256_file(checkpoint_path) != old_result["checkpoint"]["sha256"]:
                raise RuntimeError(f"A0 checkpoint binding failure for event {index}")
            checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)
            event_out = args.out / "events" / f"{index:02d}"
            event_out.mkdir(parents=True)
            results = []
            for condition in CONDITIONS:
                expert, result = train_condition(runtime, record, [row["question"] for row in paraphrases[record.record_id]], checkpoint, condition)
                condition_out = event_out / condition
                condition_out.mkdir()
                hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
                hook.attach()
                try:
                    predictions = []
                    for row in evaluation_rows(event):
                        generated = generate(runtime, row, hook, int(runtime.generation_config["max_new_tokens"]))
                        tokens = generated["generated_token_ids"]
                        predictions.append({
                            **row,
                            **generated,
                            "condition": condition,
                            "edit_id": record.record_id,
                            "normalized_exact_reference_match": normalize_medical_answer(generated["decoded_text"]) == normalize_medical_answer(row["reference"]),
                            "ended_with_eos": bool(tokens and tokens[-1] == runtime.adapter.tokenizer.eos_token_id),
                            "truncated_without_eos": generated["cap_hit"] and not (tokens and tokens[-1] == runtime.adapter.tokenizer.eos_token_id),
                        })
                finally:
                    hook.detach()
                checkpoint_sha = save_checkpoint(condition_out / "expert.pt", {
                    "rank": 4, "step": 80, "condition": condition, "record_id": record.record_id, "seed": result["seed"], "expert": expert.state_dict(),
                })
                reloaded = AsymmetricCPExpert(runtime.get_module(LAYER).in_features, runtime.get_module(LAYER).out_features, 4).to("cuda:0")
                reloaded.load_state_dict(torch.load(condition_out / "expert.pt", map_location="cuda:0", weights_only=True)["expert"])
                if any(not torch.equal(left, right) for left, right in zip(expert.state_dict().values(), reloaded.state_dict().values(), strict=True)):
                    raise RuntimeError("paired checkpoint reload mismatch")
                base_native = generate(runtime, {"question": record.question, "image_path": str(record.image_path)}, None, int(runtime.generation_config["max_new_tokens"]))
                expected = base[record.record_id]
                base_restored = base_native["generated_token_ids"] == expected["raw_generated_token_ids"] and base_native["decoded_text"] == expected["model_answer_raw"]
                guard = runtime.base_guard.verify() if runtime.base_guard else None
                if not base_restored or not guard or not guard["unchanged"]:
                    raise RuntimeError("paired base restoration or guard failure")
                result |= {
                    "status": "TRAINING_AND_GENERATION_COMPLETE__JUDGE_PENDING",
                    "event_index": index,
                    "record_id": record.record_id,
                    "starting_checkpoint_sha256": old_result["checkpoint"]["sha256"],
                    "checkpoint_sha256": checkpoint_sha,
                    "prediction_count": len(predictions),
                    "base_restored_exact": True,
                    "base_guard": guard,
                    "request_state_cleared": True,
                }
                atomic_json(condition_out / "result.json", result)
                all_predictions.extend(predictions)
                results.append(result)
                del hook, expert, reloaded
                torch.cuda.empty_cache()
            if results[0]["starting_checkpoint_sha256"] != results[1]["starting_checkpoint_sha256"] or results[0]["start_state_sha256"] != results[1]["start_state_sha256"]:
                raise RuntimeError("paired conditions did not share the same initial expert")
            event_results.extend(results)
            atomic_json(args.out / "progress.json", {"status": "RUNNING", "completed_events": index, "completed_conditions": len(event_results)})
    finally:
        del runtime
        torch.cuda.empty_cache()
    if len(all_predictions) != 326 or len(event_results) != 32:
        raise RuntimeError("paired DEV16 output coverage mismatch")
    atomic_json(args.out / "result_private.json", {
        "schema_version": "medtrace-generality-paired-private-v1",
        "status": "TRAINING_AND_GENERATION_COMPLETE__JUDGE_PENDING",
        "code_commit": actual_commit,
        "condition_results": event_results,
        "predictions": all_predictions,
    })
    atomic_json(args.out / "progress.json", {"status": "TRAINING_AND_GENERATION_COMPLETE__JUDGE_PENDING", "completed_events": 16, "completed_conditions": 32, "prediction_count": 326})


def prepare_judge(args: argparse.Namespace) -> None:
    if args.packet.exists() or args.sidecar.exists():
        raise FileExistsError("generality Judge packet already exists")
    result = json.loads(args.result.read_text())
    a0 = json.loads(args.a0_closure.read_text())["predictions"]
    rows = []
    for row in a0:
        rows.append({"condition": "CP_NATIVE_ORIGINAL", "edit_id": row["edit_id"], "query_id": row["query_id"], "task": row["task"], "question": row["question"], "reference": row["reference"], "raw_answer": row["raw_answer"], "exact": row["normalized_exact_reference_match"], "truncated_without_eos": row["truncated_without_eos"]})
    for row in result["predictions"]:
        rows.append({"condition": row["condition"], "edit_id": row["edit_id"], "query_id": row["query_id"], "task": row["task"], "question": row["question"], "reference": row["reference"], "raw_answer": row["decoded_text"], "exact": row["normalized_exact_reference_match"], "truncated_without_eos": row["truncated_without_eos"]})
    if len(rows) != 489 or len({(row["condition"], row["edit_id"], row["query_id"]) for row in rows}) != 489:
        raise RuntimeError("generality Judge comparison coverage mismatch")
    packet, sidecar = [], []
    for row in rows:
        opaque = sha256_json((row["condition"], row["edit_id"], row["query_id"], args.execution_code_commit))
        packet.append({"opaque_query_id": opaque, "question": row["question"], "gold_answer": row["reference"], "raw_base_answer": row["raw_answer"], "adjudication_pass": 1})
        sidecar.append({"opaque_query_id": opaque, **{key: row[key] for key in ("condition", "edit_id", "query_id", "task", "exact", "truncated_without_eos")}})
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in packet))
    atomic_json(args.sidecar, sidecar)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cpu-gate", type=Path, required=True)
    run_parser.add_argument("--dev-inputs", type=Path, required=True)
    run_parser.add_argument("--base-predictions", type=Path, required=True)
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--old-run", type=Path, required=True)
    run_parser.add_argument("--expected-code-commit", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.set_defaults(func=run)
    judge = sub.add_parser("prepare-judge")
    judge.add_argument("--result", type=Path, required=True)
    judge.add_argument("--a0-closure", type=Path, required=True)
    judge.add_argument("--execution-code-commit", required=True)
    judge.add_argument("--packet", type=Path, required=True)
    judge.add_argument("--sidecar", type=Path, required=True)
    judge.set_defaults(func=prepare_judge)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
