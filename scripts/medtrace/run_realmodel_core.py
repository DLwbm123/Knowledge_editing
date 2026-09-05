#!/usr/bin/env python3
"""Run the frozen MedTRACE real-model zero-effect and one-edit CP gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook  # noqa: E402
from scripts.editor_paperspec_formal import (  # noqa: E402
    assert_authorized_device,
    assert_official_llavamed_source,
    load_runtime,
)
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    CanonicalInputs,
    ids_sha256,
    manual_cached_greedy_trace,
    manual_greedy_trace,
    normalize_medical_answer,
    tensor_sha256,
)

LAYER = "model.layers.21.mlp.down_proj"
CAP = 128
SHORT_INSTRUCTION = "Answer with only the final medical answer. Do not provide an explanation."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def selected_event(dev_path: Path) -> dict[str, Any]:
    rows = read_jsonl(dev_path)
    if len(rows) != 16:
        raise RuntimeError("frozen DEV manifest must contain exactly 16 events")
    return rows[0]


def query_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    edit = event["edit_record"]
    rows = [{"query_id": edit["record_id"], "question": edit["question"], "image_path": edit["image_path"]}]
    rows.extend(
        {"query_id": row["probe_id"], "question": row["question"], "image_path": row["image_path"]}
        for row in event["probes"][:7]
    )
    if len(rows) != 8 or len({row["query_id"] for row in rows}) != 8:
        raise RuntimeError("MedTRACE canary set must contain eight unique query IDs")
    return rows


def make_canonical(runtime: Any, row: dict[str, Any], target_ids: tuple[int, ...] = ()) -> CanonicalInputs:
    batch = runtime.adapter.prepare_inputs(row["image_path"], row["question"], None)
    image = batch["images"]
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("real-model core currently requires one tensor image")
    ids = batch["input_ids"]
    return CanonicalInputs(
        prompt_text=batch["prompt"],
        full_text=batch["prompt"],
        prompt_ids=ids,
        full_ids=ids,
        image=image,
        answer_start=int(ids.shape[1]),
        target_ids=torch.tensor(target_ids, dtype=ids.dtype, device=ids.device),
        prompt_hash=ids_sha256(ids),
        full_hash=ids_sha256(ids),
        pixel_hash=tensor_sha256(image),
    )


def audit_model(runtime: Any) -> Any:
    return SimpleNamespace(llava_model=runtime.model, llava_tokenizer=runtime.adapter.tokenizer)


def eos_ids(runtime: Any) -> list[int]:
    value = runtime.adapter.tokenizer.eos_token_id
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else [int(value)]


def hf_trace(runtime: Any, canonical: CanonicalInputs) -> dict[str, Any]:
    result = runtime.adapter.generate_prepared_with_result(
        {
            "input_ids": canonical.prompt_ids,
            "attention_mask": torch.ones_like(canonical.prompt_ids),
            "images": canonical.image,
        },
        {"do_sample": False, "num_beams": 1, "max_new_tokens": CAP, "use_cache": True},
    )
    return {"token_ids": list(result.raw_token_ids), "raw_output": result.decoded_text}


def all_paths(runtime: Any, canonical: CanonicalInputs) -> dict[str, Any]:
    model = audit_model(runtime)
    eos = eos_ids(runtime)
    no_cache = manual_greedy_trace(model, canonical, CAP, eos, top_k=1)
    cached = manual_cached_greedy_trace(model, canonical, CAP, eos, top_k=1)
    hf = hf_trace(runtime, canonical)
    return {
        "manual_no_cache": {key: no_cache[key] for key in ("token_ids", "raw_output", "eos_step", "cap_hit")},
        "manual_cached": {key: cached[key] for key in ("token_ids", "raw_output", "eos_step", "cap_hit")},
        "hf": hf,
        "pass": no_cache["token_ids"] == cached["token_ids"] == hf["token_ids"] and not no_cache["cap_hit"] and not cached["cap_hit"],
    }


def load_real_runtime(args: argparse.Namespace) -> Any:
    assert_authorized_device()
    assert_official_llavamed_source()
    return load_runtime(args.cpu_gate, "cuda:0")


def zero_effect(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    event = selected_event(args.dev_manifest)
    queries = query_rows(event)
    expected = {row["query_id"]: row for row in read_jsonl(args.base_predictions)}
    if any(row["query_id"] not in expected for row in queries):
        raise RuntimeError("a frozen base prediction is missing")
    runtime = load_real_runtime(args)
    layer = runtime.get_module(LAYER)
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to("cuda:0")
    hook = MedTraceLayerHook(layer, expert)
    guard = runtime.base_guard
    if guard is None:
        raise RuntimeError("base guard missing")
    canonical = {row["query_id"]: make_canonical(runtime, row) for row in queries}

    def state_rows() -> list[dict[str, Any]]:
        rows = []
        for query in queries:
            value = canonical[query["query_id"]]
            hf = hf_trace(runtime, value)
            rows.append({
                "query_id": query["query_id"],
                "prompt_token_ids": value.prompt_ids[0].detach().cpu().tolist(),
                "generated_token_ids": hf["token_ids"],
                "decoded_text": hf["raw_output"],
            })
        return rows

    base = state_rows()
    hook.attach()
    disabled = state_rows()
    hook.set_generation_routing()
    zero = state_rows()
    hook.clear_request_routing()
    hook.detach()
    detached = state_rows()
    path_parity = {}
    for row in queries[:2]:
        hook.attach()
        hook.set_generation_routing()
        try:
            path_parity[row["query_id"]] = all_paths(runtime, canonical[row["query_id"]])
        finally:
            hook.clear_request_routing()
            hook.detach()
    frozen = all(
        row["prompt_token_ids"] == expected[row["query_id"]]["prompt_token_ids"]
        and row["generated_token_ids"] == expected[row["query_id"]]["raw_generated_token_ids"]
        and row["decoded_text"] == expected[row["query_id"]]["model_answer_raw"]
        for row in base
    )
    state_equal = base == disabled == zero == detached
    result = {
        "status": "MEDTRACE_ZERO_EFFECT_PASS" if frozen and state_equal and all(row["pass"] for row in path_parity.values()) else "MEDTRACE_ZERO_EFFECT_FAIL",
        "primary_record_id": event["edit_record"]["record_id"],
        "primary_migration": "first row of frozen DEV16 manifest; independent of LoRA results",
        "canary_count": len(queries),
        "layer": LAYER,
        "real_dimensions": {"input": layer.in_features, "output": layer.out_features},
        "factor_shapes": {"input": [expert.p_in, expert.q_in], "output": [expert.p_out, expert.q_out]},
        "assistant_predictor_contract": "last prompt position predicts first answer token; each one-token cached decode is active",
        "frozen_base_match": frozen,
        "four_state_exact_identity": state_equal,
        "path_parity": path_parity,
        "base_guard": guard.verify(),
        "states": {"base": base, "disabled": disabled, "active_zero": zero, "detached": detached},
    }
    write_json(args.out, result)
    if result["status"] != "MEDTRACE_ZERO_EFFECT_PASS":
        raise RuntimeError(result["status"])


def cp_condition(expert: AsymmetricCPExpert) -> float:
    q = expert.input_basis().float()
    return float(torch.linalg.cond(q.T @ q + expert.epsilon * torch.eye(expert.rank, device=q.device)).item())


def train(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    zero = json.loads(args.zero_effect.read_text())
    if zero.get("status") != "MEDTRACE_ZERO_EFFECT_PASS":
        raise RuntimeError("forced-on training requires MEDTRACE_ZERO_EFFECT_PASS")
    event = selected_event(args.dev_manifest)
    record = EditorRecord.from_dict(event["edit_record"])
    runtime = load_real_runtime(args)
    layer = runtime.get_module(LAYER)
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, args.rank).to("cuda:0")
    hook = MedTraceLayerHook(layer, expert)
    hook.attach()
    batch = runtime.build_edit_batch(record)
    hook.set_teacher_routing(batch.labels)
    expected_predictors = torch.zeros_like(batch.labels, dtype=torch.bool)
    expected_predictors[:, :-1] = batch.labels[:, 1:] != -100
    if not torch.equal(hook.token_mask, expected_predictors):
        raise RuntimeError("assistant predictor mask mismatch")
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3, weight_decay=0)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in expert.parameters()} or any(id(parameter) in optimized for parameter in runtime.model.parameters()):
        raise RuntimeError("optimizer boundary failure")
    primary = {"query_id": record.record_id, "question": record.question, "image_path": str(record.image_path)}
    canonical = make_canonical(runtime, primary, batch.target_token_ids)
    base_guard = runtime.base_guard
    if base_guard is None:
        raise RuntimeError("base guard missing")
    args.out.mkdir(parents=True)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir()
    started = time.monotonic()
    trajectory = args.out / "trajectory.jsonl"
    stop = None
    peak = 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps + 1):
        if step:
            hook.set_teacher_routing(batch.labels)
            optimizer.zero_grad(set_to_none=True)
            loss = runtime.compute_loss(batch)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0).item())
            rho_grad_norm = float(expert.rho.grad.float().norm().item())
            optimizer.step()
            expert.normalize_factors_()
            condition = cp_condition(expert)
            if not math.isfinite(condition) or condition > 1e4:
                raise RuntimeError(f"CP input basis condition hard stop: {condition}")
        else:
            grad_norm = rho_grad_norm = 0.0
            condition = cp_condition(expert)
        peak = max(peak, torch.cuda.max_memory_allocated())
        if step % args.eval_every:
            continue
        hook.set_teacher_routing(batch.labels)
        with torch.no_grad():
            score = runtime.score_target(batch)
        hook.set_generation_routing()
        generation = hf_trace(runtime, canonical)
        literal = normalize_medical_answer(generation["raw_output"]) == normalize_medical_answer(record.target)
        checkpoint = checkpoints / f"step_{step:03d}.pt"
        torch.save({"rank": args.rank, "step": step, "expert": expert.state_dict()}, checkpoint)
        row = {
            "step": step,
            **score,
            "generation": generation,
            "literal_normalized_target_match": literal,
            "first_token_rank_one": score["first_target_token_rank"] == 1,
            "residual_map_norm": float(expert.materialize_dense().float().norm().item()),
            "rho_norm": float(expert.rho.float().norm().item()),
            "gradient_norm": grad_norm,
            "rho_gradient_norm": rho_grad_norm,
            "input_basis_condition": condition,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint),
        }
        append_jsonl(trajectory, row)
        if step and literal and row["first_token_rank_one"]:
            stop = row
            break
    hook.clear_request_routing()
    disabled = hf_trace(runtime, canonical)
    hook.detach()
    base_expected = next(row for row in read_jsonl(args.base_predictions) if row["query_id"] == record.record_id)
    result = {
        "status": "MEDTRACE_CP_CANDIDATE_NEEDS_FIXED_JUDGE" if stop else "MEDTRACE_CP_CAPACITY_NOT_PASSED",
        "rank": args.rank,
        "layer": LAYER,
        "steps_executed": int(stop["step"] if stop else args.steps),
        "selected_candidate": stop,
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0, "gradient_clip": 1.0},
        "budget": {"max_steps": args.steps, "generation_every": args.eval_every, "max_new_tokens": CAP},
        "assistant_predictor_positions": torch.where(expected_predictors[0])[0].detach().cpu().tolist(),
        "disabled_restores_frozen_s0": disabled["token_ids"] == base_expected["raw_generated_token_ids"] and disabled["raw_output"] == base_expected["model_answer_raw"],
        "base_guard": base_guard.verify(),
        "peak_allocated_bytes": peak,
        "elapsed_seconds": time.monotonic() - started,
        "trajectory": str(trajectory),
    }
    write_json(args.out / "result.json", result)


def verify_candidate(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    event = selected_event(args.dev_manifest)
    record = EditorRecord.from_dict(event["edit_record"])
    runtime = load_real_runtime(args)
    layer = runtime.get_module(LAYER)
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    expert = AsymmetricCPExpert(layer.in_features, layer.out_features, int(saved["rank"])).to("cuda:0")
    expert.load_state_dict(saved["expert"])
    batch = runtime.build_edit_batch(record)
    primary = {"query_id": record.record_id, "question": record.question, "image_path": str(record.image_path)}
    canonical = make_canonical(runtime, primary, batch.target_token_ids)
    short = make_canonical(runtime, primary | {"question": record.question + "\n\n" + SHORT_INSTRUCTION}, batch.target_token_ids)
    hook = MedTraceLayerHook(layer, expert)
    hook.attach()
    captured: list[torch.Tensor] = []
    capture = layer.register_forward_pre_hook(lambda _module, values: captured.append(values[0].detach()))
    try:
        hook.set_teacher_routing(batch.labels)
        with torch.no_grad():
            score = runtime.score_target(batch)
    finally:
        capture.remove()
    if len(captured) != 1:
        raise RuntimeError("real activation capture failed")
    activation = captured[0][:, -1:]
    factorized = expert.residual(activation)
    dense = expert.normalize_activation(activation) @ expert.materialize_dense().T
    dense_error = float((factorized - dense).abs().max().item())
    hook.set_generation_routing()
    original = all_paths(runtime, canonical)
    original_short = all_paths(runtime, short)
    hook.clear_request_routing()
    hook.detach()

    reloaded = AsymmetricCPExpert(layer.in_features, layer.out_features, int(saved["rank"])).to("cuda:0")
    reloaded.load_state_dict(saved["expert"])
    replay_hook = MedTraceLayerHook(layer, reloaded)
    replay_hook.attach()
    replay_hook.set_generation_routing()
    replay = all_paths(runtime, canonical)
    replay_short = all_paths(runtime, short)
    replay_hook.clear_request_routing()
    disabled = hf_trace(runtime, canonical)
    disabled_short = hf_trace(runtime, short)
    replay_hook.detach()
    detached_short = hf_trace(runtime, short)
    base = next(row for row in read_jsonl(args.base_predictions) if row["query_id"] == record.record_id)
    guard = runtime.base_guard.verify() if runtime.base_guard else None
    passed = (
        original["pass"]
        and original_short["pass"]
        and replay["pass"]
        and replay_short["pass"]
        and original["hf"] == replay["hf"]
        and original_short["hf"] == replay_short["hf"]
        and score["first_target_token_rank"] == 1
        and dense_error <= 1e-5
        and disabled["token_ids"] == base["raw_generated_token_ids"]
        and disabled["raw_output"] == base["model_answer_raw"]
        and disabled_short == detached_short
        and bool(guard and guard["unchanged"])
    )
    write_json(args.out, {
        "status": "MEDTRACE_CP_ENGINEERING_PASS" if passed else "MEDTRACE_CP_ENGINEERING_FAIL",
        "checkpoint": str(args.checkpoint),
        "rank": int(saved["rank"]),
        "step": int(saved["step"]),
        "score": score,
        "dense_factor_max_abs_error": dense_error,
        "original_paths": original,
        "original_short_paths": original_short,
        "reload_paths": replay,
        "reload_short_paths": replay_short,
        "save_reload_generation_replay": original["hf"] == replay["hf"],
        "save_reload_short_generation_replay": original_short["hf"] == replay_short["hf"],
        "short_literal_normalized_target_match": normalize_medical_answer(original_short["hf"]["raw_output"]) == normalize_medical_answer(record.target),
        "disabled_restores_frozen_s0": disabled["token_ids"] == base["raw_generated_token_ids"] and disabled["raw_output"] == base["model_answer_raw"],
        "short_disable_detach_identity": disabled_short == detached_short,
        "base_guard": guard,
    })
    if not passed:
        raise RuntimeError("MEDTRACE_CP_ENGINEERING_FAIL")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cpu-gate", type=Path, required=True)
    common.add_argument("--dev-manifest", type=Path, required=True)
    common.add_argument("--base-predictions", type=Path, required=True)
    sub = value.add_subparsers(dest="mode", required=True)
    zero = sub.add_parser("zero-effect", parents=[common])
    zero.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("train", parents=[common])
    run.add_argument("--zero-effect", type=Path, required=True)
    run.add_argument("--rank", type=int, choices=(4, 8, 16), required=True)
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--eval-every", type=int, default=20)
    run.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify", parents=[common])
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    {"zero-effect": zero_effect, "train": train, "verify": verify_candidate}[args.mode](args)
