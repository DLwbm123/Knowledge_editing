#!/usr/bin/env python3
"""Continuous-checkpoint DEV/QUAL runner for LoRA-Perf-v1."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from m3bench_repro.editors.llava_runtime import EditorRecord, canonical_sha256
from m3bench_repro.editors.methods import LoraPaperSpecEditor, LoraRuntimeConfig, finite_gradients
from scripts.editor_paperspec_formal import (
    assert_authorized_device,
    assert_official_llavamed_source,
    editor_empty,
    generate_probe,
    load_runtime,
    save_state_atomic,
    sha256,
    utc_now,
    write_frozen_json,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def adapter_delta(editor: LoraPaperSpecEditor) -> dict[str, float]:
    current = dict(editor.peft_model.named_parameters())
    deltas = [
        (current[name].detach().float().cpu() - initial).reshape(-1)
        for name, initial in editor._initial_adapter_state.items()
    ]
    joined = torch.cat(deltas)
    return {"l2": float(torch.linalg.vector_norm(joined)), "max_abs": float(joined.abs().max())}


def save_optimizer_rng(path: Path, optimizer: torch.optim.Optimizer, step: int) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite optimizer/RNG checkpoint: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    assert_official_llavamed_source()
    assert_authorized_device()
    events = read_jsonl(args.inputs)
    if len(events) != 16 or len({row["event_id"] for row in events}) != 16:
        raise RuntimeError("LoRA-Perf split must contain 16 unique frozen events")
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("positive checkpoints are required")
    config = LoraRuntimeConfig(
        profile_name=args.profile_name,
        learning_rate=args.learning_rate,
        steps_per_edit=checkpoints[-1],
        rank=args.rank,
        alpha=args.alpha,
        layer_scope=args.layer_scope,
        target_modules=tuple(args.target_modules),
    )
    profile = {**config.__dict__, "target_modules": list(config.target_modules)}
    run_lock = {
        "schema_version": "lora-perf-v1-run-lock-v1",
        "created_at_utc": utc_now(),
        "split": args.split,
        "profile": profile,
        "checkpoints": list(checkpoints),
        "inputs_sha256": sha256(args.inputs),
        "generation_lock_sha256": sha256(args.cpu_gate / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json"),
        "code_commit": args.code_commit,
        "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
        "gpu_uuid": os.environ["M3BENCH_FORMAL_EXPECTED_GPU_UUID"],
    }
    run_lock["lock_sha256"] = canonical_sha256(run_lock)
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "RUN_LOCK.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        ignored = {"created_at_utc", "lock_sha256"}
        if {k: v for k, v in existing.items() if k not in ignored} != {k: v for k, v in run_lock.items() if k not in ignored}:
            raise RuntimeError("existing LoRA-Perf output has a different run lock")
    else:
        write_frozen_json(lock_path, run_lock)

    runtime = load_runtime(args.cpu_gate, "cuda:0")
    editor = LoraPaperSpecEditor(runtime, config=config)
    generation_lock = run_lock["generation_lock_sha256"]
    completed = []
    for ordinal, event in enumerate(events, 1):
        event_dir = args.output / f"event_{ordinal:02d}"
        report_path = event_dir / "raw_event.json"
        if report_path.is_file():
            completed.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        if event_dir.exists():
            os.replace(event_dir, args.output / f"event_{ordinal:02d}.orphan.{int(time.time())}")
        event_dir.mkdir()
        if not editor_empty(editor, "lora")["pass"]:
            raise RuntimeError(f"nonempty adapter before event {ordinal}")
        record = EditorRecord.from_dict(event["edit_record"])
        editor._set_enabled(True)
        batch = runtime.build_edit_batch(record)
        pre_target = editor.score_target_nll(record)
        base_target = editor.generate(record)
        parameters = editor.trainable()
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
        initial_state = editor.adapter_state_sha256()
        trajectory, gradients = [], []
        started = time.perf_counter()
        editor.peft_model.eval()
        with runtime.peak_memory() as peak:
            for step in range(1, checkpoints[-1] + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = runtime.compute_loss(batch)
                loss.backward()
                gradients.append(finite_gradients(parameters))
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
                optimizer.step()
                if step in checkpoints:
                    target_score = editor.score_target_nll(record)
                    target_generation = editor.generate(record)
                    probes = [generate_probe(editor, probe, generation_lock) for probe in event["probes"]]
                    trajectory.append({
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm_before_clip": grad_norm,
                        "target_score": target_score,
                        "target_generation": target_generation,
                        "probes": probes,
                        "adapter_delta": adapter_delta(editor),
                    })
            state = save_state_atomic(editor, "lora", event_dir / "editor_state")
            save_optimizer_rng(event_dir / "optimizer_rng.pt", optimizer, checkpoints[-1])
        base_integrity = editor.base_integrity()
        result = {
            "schema_version": "lora-perf-v1-event-v1",
            "status": "PASS" if all(gradients) and base_integrity["unchanged"] else "FAIL",
            "event_id": event["event_id"],
            "event_position": ordinal,
            "edit_record": event["edit_record"],
            "pre_target": pre_target,
            "base_target": base_target,
            "target_mask": batch.mask_report(),
            "trajectory": trajectory,
            "initial_adapter_sha256": initial_state,
            "final_adapter_sha256": editor.adapter_state_sha256(),
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "target_paths": editor.targets,
            "finite_gradients": all(gradients),
            "base_integrity": base_integrity,
            "state": state,
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_memory": peak,
        }
        write_frozen_json(report_path, result)
        completed.append(result)
        editor.reset_editor_state()
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": "lora-perf-v1-raw-manifest-v1",
        "status": "PASS" if len(completed) == 16 and all(row["status"] == "PASS" for row in completed) else "FAIL",
        "split": args.split,
        "event_count": len(completed),
        "profile": profile,
        "checkpoints": list(checkpoints),
        "raw_only_semantic_metrics_pending": True,
    }
    if not (args.output / "RAW_MANIFEST.json").exists():
        write_frozen_json(args.output / "RAW_MANIFEST.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cpu-gate", type=Path, required=True)
    result.add_argument("--inputs", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--split", choices=("DEV16", "QUAL16"), required=True)
    result.add_argument("--profile-name", default="LoRA-Perf-v1")
    result.add_argument("--learning-rate", type=float, required=True)
    result.add_argument("--checkpoints", type=int, nargs="+", default=(5, 10, 20, 40, 80))
    result.add_argument("--rank", type=int, default=16)
    result.add_argument("--alpha", type=int, default=16)
    result.add_argument("--layer-scope", choices=("all", "last_16", "last_8"), default="all")
    result.add_argument("--target-modules", nargs="+", default=("gate_proj", "up_proj", "down_proj"))
    result.add_argument("--code-commit", required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
