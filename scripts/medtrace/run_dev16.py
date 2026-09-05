#!/usr/bin/env python3
"""Run 16 independent forced-on MedTRACE CP experts with one resident backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import ids_sha256, normalize_medical_answer  # noqa: E402
from scripts.medtrace.run_realmodel_core import LAYER, cp_condition, load_real_runtime, make_canonical  # noqa: E402

METHOD = "TIME_INSPIRED_CP_R4_DEV16_FORCED_ON"
SEED_BASE = 20260905
TRAIN_CAP = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def derive_seed(record_id: str, base: int = SEED_BASE) -> int:
    return int(hashlib.sha256(f"{base}\0{record_id}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def validate_dev_rows(rows: list[dict[str, Any]]) -> None:
    ids = [row.get("edit_record", {}).get("record_id") for row in rows]
    positions = [row.get("event_position") for row in rows]
    if len(rows) != 16 or len(set(ids)) != 16 or any(not value for value in ids):
        raise RuntimeError("DEV16 must contain 16 unique edit records")
    if positions != sorted(positions) or len(set(positions)) != 16:
        raise RuntimeError("DEV16 event order is not frozen and unique")


def evaluation_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    edit = event["edit_record"]
    rows = [{
        "query_id": edit["record_id"], "task": "T0", "question": edit["question"],
        "image_path": edit["image_path"], "reference": edit["gold_answer"],
        "lineage": {"kind": "native", "event_id": event["event_id"]},
    }]
    rows.extend({
        "query_id": row["probe_id"], "task": row["task"], "question": row["question"],
        "image_path": row["image_path"], "reference": row["reference"],
        "lineage": {k: row.get(k) for k in ("edit_id", "probe_index", "variant_type", "sequence_position")},
    } for row in event["probes"])
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate query ID inside DEV16 event")
    return rows


def rng_lock(seed: int) -> dict[str, Any]:
    seed_everything(seed)
    cpu = torch.get_rng_state().numpy().tobytes()
    cuda = torch.cuda.get_rng_state(0).cpu().numpy().tobytes()
    return {"seed": seed, "cpu_rng_sha256": hashlib.sha256(cpu).hexdigest(), "cuda_rng_sha256": hashlib.sha256(cuda).hexdigest()}


def generate(runtime: Any, row: dict[str, Any], hook: MedTraceLayerHook | None, max_new_tokens: int) -> dict[str, Any]:
    canonical = make_canonical(runtime, row)
    config = {
        "do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
        "use_cache": True, "eos_token_id": runtime.adapter.tokenizer.eos_token_id,
    }
    started = time.monotonic()
    if hook is None:
        result = runtime.adapter.generate_prepared_with_result({
            "input_ids": canonical.prompt_ids,
            "attention_mask": torch.ones_like(canonical.prompt_ids),
            "images": canonical.image,
        }, config)
        lifecycle: tuple[dict[str, int], ...] = ()
    else:
        with hook.generation_request():
            result = runtime.adapter.generate_prepared_with_result({
                "input_ids": canonical.prompt_ids,
                "attention_mask": torch.ones_like(canonical.prompt_ids),
                "images": canonical.image,
            }, config)
        lifecycle = hook.last_generation_trace
    torch.cuda.synchronize()
    return {
        "prompt_token_ids": canonical.prompt_ids[0].detach().cpu().tolist(),
        "prompt_token_ids_sha256": ids_sha256(canonical.prompt_ids),
        "image_tensor_sha256": canonical.pixel_hash,
        "generated_token_ids": list(result.raw_token_ids),
        "decoded_text": result.decoded_text,
        "generated_token_count": len(result.raw_token_ids),
        "cap_hit": len(result.raw_token_ids) >= max_new_tokens,
        "elapsed_seconds": time.monotonic() - started,
        "request_lifecycle": lifecycle,
    }


def optimizer_lock(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    defaults = optimizer.defaults
    return {
        "name": type(optimizer).__name__, "lr": defaults["lr"], "weight_decay": defaults["weight_decay"],
        "betas": list(defaults["betas"]), "eps": defaults["eps"], "gradient_clip": 1.0,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def run_event(runtime: Any, event: dict[str, Any], base: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    record = EditorRecord.from_dict(event["edit_record"])
    random_lock = rng_lock(derive_seed(record.record_id))
    layer = runtime.get_module(LAYER)
    if (layer.in_features, layer.out_features) != (14336, 4096):
        raise RuntimeError(f"unexpected down_proj dimensions: {(layer.in_features, layer.out_features)}")
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    hook = MedTraceLayerHook(layer, expert)
    batch = runtime.build_edit_batch(record)
    with torch.no_grad():
        base_score = runtime.score_target(batch)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3, weight_decay=0)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in expert.parameters()} or any(
        id(parameter) in optimized for parameter in runtime.model.parameters()
    ):
        raise RuntimeError("optimizer boundary failure")
    expected_mask = torch.zeros_like(batch.labels, dtype=torch.bool)
    expected_mask[:, :-1] = batch.labels[:, 1:] != -100
    hook.attach()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    trajectory = []
    selected = None
    early_stop_reached = False
    canonical_row = {"query_id": record.record_id, "question": record.question, "image_path": str(record.image_path)}
    for step in range(201):
        grad_norm = rho_grad_norm = 0.0
        if step:
            hook.set_teacher_routing(batch.labels)
            if not torch.equal(hook.token_mask, expected_mask):
                raise RuntimeError("assistant predictor mask mismatch")
            optimizer.zero_grad(set_to_none=True)
            loss = runtime.compute_loss(batch)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0).item())
            rho_grad_norm = float(expert.rho.grad.float().norm().item())
            if not math.isfinite(grad_norm) or not math.isfinite(rho_grad_norm):
                raise FloatingPointError("non-finite CP gradient")
            optimizer.step()
            expert.normalize_factors_()
        condition = cp_condition(expert)
        if not math.isfinite(condition) or condition > 1e4:
            raise RuntimeError(f"CP input basis condition hard stop: {condition}")
        if step % 20:
            continue
        hook.set_teacher_routing(batch.labels)
        with torch.no_grad():
            score = runtime.score_target(batch)
        generated = generate(runtime, canonical_row, hook, TRAIN_CAP)
        literal = normalize_medical_answer(generated["decoded_text"]) == normalize_medical_answer(record.target)
        selected = {
            "step": step, "score": score, "literal_normalized_target_match": literal,
            "first_token_rank_one": score["first_target_token_rank"] == 1,
            "cap_hit": generated["cap_hit"], "generated_token_ids": generated["generated_token_ids"],
            "decoded_text": generated["decoded_text"], "gradient_norm": grad_norm,
            "rho_gradient_norm": rho_grad_norm, "input_basis_condition": condition,
            "residual_map_norm": float(expert.materialize_dense().float().norm().item()),
            "rho_norm": float(expert.rho.float().norm().item()), "elapsed_seconds": time.monotonic() - started,
        }
        trajectory.append(selected)
        if step and literal and selected["first_token_rank_one"] and not generated["cap_hit"]:
            early_stop_reached = True
            break
    if selected is None:
        raise RuntimeError("DEV16 training produced no selected checkpoint")
    checkpoint = out / "expert.pt"
    checkpoint_sha = save_checkpoint(checkpoint, {
        "method": METHOD, "record_id": record.record_id, "seed": random_lock["seed"],
        "rank": 4, "step": selected["step"], "expert": expert.state_dict(),
    })
    eval_started = time.monotonic()
    outputs = []
    for row in evaluation_rows(event):
        generated = generate(runtime, row, hook, int(runtime.generation_config["max_new_tokens"]))
        normalized = normalize_medical_answer(generated["decoded_text"])
        reference = normalize_medical_answer(row["reference"])
        outputs.append({
            **row, **generated, "normalized_output": normalized,
            "exact_normalized_reference_match": normalized == reference,
            "native_target_copy": normalize_medical_answer(record.target) in normalized,
            "semantic_verdict": None,
        })
    hook.detach()
    if hook.enabled or hook.generation_routing or hook.token_mask is not None:
        raise RuntimeError("MedTRACE request state survived detach")
    reloaded = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    reloaded.load_state_dict(saved["expert"])
    replay_hook = MedTraceLayerHook(layer, reloaded)
    replay_hook.attach()
    reloaded_native = generate(runtime, canonical_row, replay_hook, int(runtime.generation_config["max_new_tokens"]))
    replay_hook.detach()
    base_native = generate(runtime, canonical_row, None, int(runtime.generation_config["max_new_tokens"]))
    expected = base[record.record_id]
    base_restored = (
        base_native["generated_token_ids"] == expected["raw_generated_token_ids"]
        and base_native["decoded_text"] == expected["model_answer_raw"]
    )
    guard = runtime.base_guard.verify() if runtime.base_guard else None
    if not base_restored or not guard or not guard["unchanged"]:
        raise RuntimeError("base restoration or sampled guard failed")
    if reloaded_native["generated_token_ids"] != outputs[0]["generated_token_ids"]:
        raise RuntimeError("native checkpoint reload replay failed")
    result = {
        "status": "EARLY_STOP_REACHED" if early_stop_reached else "EARLY_STOP_NOT_REACHED_WITHIN_BUDGET",
        "method": METHOD, "event_position": event["event_position"], "record_id": record.record_id,
        "rng": random_lock, "input": {
            "image_sha256": batch.image_sha256, "prompt_input_ids_sha256": ids_sha256(batch.raw_input_ids),
            "target_token_ids_sha256": sha256_json(list(batch.target_token_ids)),
            "assistant_predictor_indices": torch.where(expected_mask[0])[0].detach().cpu().tolist(),
        },
        "expert": {
            "layer": LAYER, "dimensions": [layer.in_features, layer.out_features], "rank": 4,
            "factorization": {"input": [expert.p_in, expert.q_in], "output": [expert.p_out, expert.q_out]},
            "parameter_count": sum(parameter.numel() for parameter in expert.parameters()),
            "beta": expert.beta, "epsilon": expert.epsilon,
        },
        "optimizer": optimizer_lock(optimizer), "base_score": base_score, "selected": selected,
        "nll_decreased": selected["score"]["nll"] < base_score["nll"],
        "checkpoint": {"artifact": "expert.pt", "sha256": checkpoint_sha, "step": selected["step"]},
        "trajectory": trajectory, "evaluation": outputs,
        "timing": {
            "training_seconds": selected["elapsed_seconds"], "evaluation_seconds": time.monotonic() - eval_started,
            "end_to_end_seconds": time.monotonic() - started,
        },
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "reload_native_exact": True, "base_restored_exact": True,
        "request_state_cleared": True, "base_sampled_guard": guard,
    }
    del replay_hook, reloaded, hook, expert, optimizer, batch
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-gate", type=Path, required=True)
    parser.add_argument("--dev-inputs", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha256_file(args.dev_inputs) != args.expected_dev_sha256 or sha256_file(args.base_predictions) != args.expected_base_sha256:
        raise RuntimeError("DEV16 or base-prediction input lock mismatch")
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("running code commit differs from frozen commit")
    rows = read_jsonl(args.dev_inputs)
    validate_dev_rows(rows)
    base_rows = read_jsonl(args.base_predictions)
    base = {row["query_id"]: row for row in base_rows}
    needed = {item["query_id"] for event in rows for item in evaluation_rows(event)}
    if not needed <= base.keys():
        raise RuntimeError("frozen base predictions do not cover DEV16 evaluation inputs")
    args.out.mkdir(parents=True)
    atomic_json(args.out / "run_lock.json", {
        "status": "RUNNING", "method": METHOD, "code_commit": actual_commit,
        "dev_inputs_sha256": args.expected_dev_sha256, "base_predictions_sha256": args.expected_base_sha256,
        "event_count": len(rows), "seed_base": SEED_BASE, "seed_derivation": "sha256(base\\0record_id) first32 & 0x7fffffff",
        "training": {"rank": 4, "layer": LAYER, "steps": 200, "eval_every": 20, "native_cap": TRAIN_CAP},
    })
    runtime = load_real_runtime(args)
    results = []
    try:
        for index, event in enumerate(rows, 1):
            event_dir = args.out / "events" / f"{index:02d}"
            event_dir.mkdir(parents=True)
            try:
                result = run_event(runtime, event, base, event_dir)
            except Exception as error:
                atomic_json(event_dir / "error.json", {"status": "ENGINEERING_ERROR", "type": type(error).__name__, "message": str(error)})
                atomic_json(args.out / "progress.json", {"status": "CP_DEV16_PARTIAL_ENGINEERING_BLOCK", "attempted": index, "completed": len(results), "error": str(error)})
                raise
            atomic_json(event_dir / "result.json", result)
            results.append(result)
            atomic_json(args.out / "progress.json", {"status": "RUNNING", "attempted": index, "completed": len(results)})
        aggregate = {
            "status": "CP_DEV16_COMPLETED", "attempted": 16, "completed": 16, "engineering_errors": 0,
            "early_stop_reached": sum(row["status"] == "EARLY_STOP_REACHED" for row in results),
            "native_exact_success": sum(row["evaluation"][0]["exact_normalized_reference_match"] for row in results),
            "nll_decreased": sum(row["nll_decreased"] for row in results),
            "probe_counts": {
                task: sum(item["task"] == task for row in results for item in row["evaluation"])
                for task in ("T1L", "T1G", "T2G")
            },
            "total_seconds": sum(row["timing"]["end_to_end_seconds"] for row in results),
        }
        atomic_json(args.out / "result.json", aggregate)
        atomic_json(args.out / "progress.json", aggregate)
    finally:
        del runtime
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
