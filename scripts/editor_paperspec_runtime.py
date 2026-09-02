#!/usr/bin/env python3
"""GPU entrypoint for M3Bench editor inventory, preflight, smoke, and mini-stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml

from m3bench_repro.editors.llava_runtime import (
    RUN_ROOT,
    LlavaMedEditorRuntime,
    canonical_sha256,
    load_frozen_smoke_records,
    sha256_file,
    write_json_atomic,
)
from m3bench_repro.editors.methods import CLASSIFICATION, PaperSpecEditor, create_editor
from m3bench_repro.editors.routing import canonical_float32, route_dict_equal
from scripts.editor_effect_probe import inventory_topology, state_delta, target_lock_topology


WORKTREE = Path(os.environ.get("M3BENCH_WORKTREE", Path(__file__).resolve().parents[1]))
MINI_SOURCE = RUN_ROOT / (
    "carry_forward/foundation_v4/editor_gate/cohorts/PROPOSED_SEQUENTIAL_MINISTREAM_4.jsonl"
)
METHODS = ("lora", "grace", "balancedit", "belora")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(WORKTREE), *args], text=True).strip()


def generation_equal(left: dict, right: dict) -> bool:
    return (
        left["decoded_text"] == right["decoded_text"]
        and left["raw_token_ids"] == right["raw_token_ids"]
        and left["sequence_contract"] == right["sequence_contract"]
    )


def write_frozen_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace differing frozen artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o444)


def write_frozen_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    write_frozen_text(path, payload)


def runtime_environment(device: str) -> dict[str, Any]:
    logical = torch.device(device)
    props = torch.cuda.get_device_properties(logical)
    return {
        "device_argument": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(logical),
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "capability": [props.major, props.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def load_runtime(device: str, *, seed: int = 20260828) -> LlavaMedEditorRuntime:
    runtime = LlavaMedEditorRuntime(device=device)
    runtime.load_frozen_backbone(seed=seed)
    inventory, target_lock = runtime.resolve_module_inventory(freeze=False)
    inventory_path = RUN_ROOT / "runtime/LLAVA_MED_MODULE_INVENTORY.json"
    target_path = RUN_ROOT / "runtime/LLAVA_MED_EDIT_TARGET_LOCK.json"
    if inventory_path.exists():
        frozen_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if canonical_sha256(inventory_topology(frozen_inventory)) != canonical_sha256(
            inventory_topology(inventory)
        ):
            raise RuntimeError("real-model module inventory differs from frozen inventory")
    if target_path.exists():
        frozen_targets = json.loads(target_path.read_text(encoding="utf-8"))
        if canonical_sha256(target_lock_topology(frozen_targets)) != canonical_sha256(
            target_lock_topology(target_lock)
        ):
            raise RuntimeError("real-model target lock differs from frozen target lock")
    return runtime


def state_path(method: str, output_dir: Path, name: str = "editor_state") -> Path:
    if method == "lora":
        return output_dir / name
    return output_dir / f"{name}.pt"


def route_hits_record(method: str, result: dict[str, Any], record_id: str) -> bool:
    if method == "lora":
        return True
    route = result.get("route") or {}
    return bool(route.get("activated") and route.get("logical_edit_id") == record_id)


def state_count(editor: PaperSpecEditor) -> int:
    summary = editor.state_summary()
    return int(summary.get("entry_count", len(summary.get("edit_history", []))))


def freeze_method_documents(editor: PaperSpecEditor, method_dir: Path | None = None) -> None:
    method_dir = method_dir or RUN_ROOT / editor.method
    method_dir.mkdir(parents=True, exist_ok=True)
    config = editor.config_lock()
    config["config_sha256"] = canonical_sha256(config)
    yaml_path = method_dir / "METHOD_CONFIG_LOCK_V2.yaml"
    yaml_payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    write_frozen_text(yaml_path, yaml_payload)

    sources = {
        "lora": {
            "repository": "https://github.com/huggingface/peft",
            "commit": "ba6a19060d6ab54a87538a6e77e3e4d5a907375b",
            "adaptation": "exact real-model target resolution and shared Foundation multimodal runtime",
        },
        "grace": {
            "repository": "https://github.com/Thartvigsen/GRACE.git",
            "commit": "f674183f17a995d109e10ee6140d4c3e6d016115",
            "adaptation": "authorized cosine retrieval and LLaVA-Med final-up-projection adapter; source insert/collision/value semantics retained",
        },
        "balancedit": {
            "repository": "https://github.com/donglgcn/BalancEdit.git",
            "commit": "83749e52a1d27331d21cfec845b6089294730c2f",
            "adaptation": "LLaVA-Med pooled hidden key, official rephrase/black anchors, non-destructive routed full up-projection copy",
        },
        "belora": {
            "repository": None,
            "commit": None,
            "adaptation": "independent paper-spec implementation of BalanceEdit routing plus coordinated per-edit LoRA on final LM MLP",
        },
    }
    code_files = [
        WORKTREE / "m3bench_repro/editors/llava_runtime.py",
        WORKTREE / "m3bench_repro/editors/routing.py",
        WORKTREE / "m3bench_repro/editors/routed_layers.py",
        WORKTREE / "m3bench_repro/editors/methods.py",
        WORKTREE / "scripts/editor_paperspec_runtime.py",
    ]
    provenance = {
        "schema_version": "m3bench-editor-method-provenance-amendment-v2",
        "method": editor.method,
        "classification": CLASSIFICATION,
        "not_author_runtime": True,
        "source": sources[editor.method],
        "child_commit": git("rev-parse", "HEAD"),
        "child_git_status": git("status", "--short"),
        "target_lock_sha256": sha256_file(RUN_ROOT / "runtime/LLAVA_MED_EDIT_TARGET_LOCK.json"),
        "code_sha256": {str(path.relative_to(WORKTREE)): sha256_file(path) for path in code_files},
        "author_belora_implementation_available": False,
        "formal_experiment": False,
    }
    write_frozen_json(method_dir / "METHOD_PROVENANCE_AMENDMENT.json", provenance)

    if editor.method == "grace":
        write_frozen_text(
            method_dir / "GRACE_COSINE_ADAPTATION_DIFF.md",
            "# GRACE Cosine Adaptation Diff\n\n"
            "Primary retrieval replaces the locked source's hard-coded Euclidean `torch.cdist(..., p=2)` with "
            "`1 - cosine_similarity`, and keeps the inclusive `nearest_distance <= stored_radius` gate. "
            "Initial radius 1.0, source insert/update/collision policy, cold value initialization, prompt value "
            "application, 100 Adam steps at lr 1.0, reset, and persistence semantics are retained. "
            "The locked source checkout is unmodified; Euclidean remains diagnostic-only.\n",
        )
    if editor.method == "belora":
        write_frozen_text(
            method_dir / "METHOD_STATE_CONTRACT_V2.md",
            "# BELoRA State Contract V2\n\n"
            "Implementation label: `BELORA_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED`. "
            "Each logical edit owns one pooled "
            "key, one radius, and a coordinated set of LoRA A/B tensors across the final LM MLP gate/up/down "
            "linears. Exactly one routed logical ID may be active. A miss activates none and executes the frozen "
            "base linears exactly. No full module weight copies are stored. State never crosses method processes.\n",
        )
        write_frozen_text(
            method_dir / "IMPLEMENTATION_SPEC.md",
            "# BELoRA Paper-Spec Implementation\n\n"
            "This is an independent implementation based on M3Bench Appendix D.5: BalanceEdit pooled-key/radius "
            "routing plus per-edit rank-16 LoRA adapters. The primary scope is the final language-model MLP only. "
            "Projector, vision encoder, public SDXL B-LoRA, MEND substitution, and architecture sweeps are excluded.\n",
        )
    for path in method_dir.iterdir():
        if path.is_file() and path.name != "LLAVA_MED_RUNTIME_TEST_REPORT.md":
            os.chmod(path, 0o444)


def command_inventory(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    runtime = LlavaMedEditorRuntime(device=args.device)
    with runtime.peak_memory() as memory:
        runtime.load_frozen_backbone(seed=20260828)
        inventory, target_lock = runtime.resolve_module_inventory(freeze=True)
        record = load_frozen_smoke_records()[0]
        mask = runtime.verify_target_mask_determinism(record)
        cached = runtime.generate(record, use_cache=True)
        uncached = runtime.generate(record, use_cache=False)
    checks = {
        "module_inventory_resolved": inventory["language_block_count"] > 0,
        "target_lock_resolved": target_lock["lora"]["target_count"] == inventory["language_block_count"] * 3,
        "target_mask_deterministic": mask["pass"],
        "cached_no_cache_exact_generation": generation_equal(cached, uncached),
        "base_guard_clean": runtime.base_guard.verify()["unchanged"],
    }
    report = {
        "schema_version": "m3bench-editor-llava-runtime-integration-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": CLASSIFICATION,
        "checks": checks,
        "environment": runtime_environment(args.device),
        "target_mask": mask,
        "cached_generation": cached,
        "uncached_generation": uncached,
        "peak_memory": memory,
        "runtime_seconds": time.perf_counter() - started,
        "base_integrity": runtime.base_guard.verify(),
    }
    path = RUN_ROOT / "runtime/LLAVA_MED_RUNTIME_INTEGRATION.json"
    write_json_atomic(path, report, read_only=True)
    os.chmod(RUN_ROOT / "runtime/LLAVA_MED_MODULE_INVENTORY.json", 0o444)
    os.chmod(RUN_ROOT / "runtime/LLAVA_MED_EDIT_TARGET_LOCK.json", 0o444)
    print(json.dumps({"status": report["status"], "checks": checks, "peak_memory": memory}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_single_run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record = load_frozen_smoke_records()[args.record_index]
    started = time.perf_counter()
    runtime = load_runtime(args.device)
    editor = create_editor(args.method, runtime)
    freeze_method_documents(editor, output / "method_lock")
    mask = runtime.verify_target_mask_determinism(record)
    pre_cached = runtime.generate(record, use_cache=True)
    pre_uncached = runtime.generate(record, use_cache=False)
    pre_nll = editor.score_target_nll(record)
    with runtime.peak_memory() as memory:
        edit = editor.apply_edit(record)
        post_nll = editor.score_target_nll(record)
        post_cached = editor.generate(record, use_cache=True)
        post_uncached = editor.generate(record, use_cache=False)
    summary = editor.state_summary()
    state = editor.save_editor_state(state_path(args.method, output))
    base_integrity = editor.base_integrity()
    editor.reset_editor_state()
    reset_generation = runtime.generate(record, use_cache=True)
    checks = {
        "target_mask": mask["pass"],
        "pre_cached_no_cache": generation_equal(pre_cached, pre_uncached),
        "edit_finite_losses": edit["finite_losses"],
        "edit_finite_gradients": edit["finite_gradients"],
        "state_created": state["size_bytes"] > 0,
        "post_cached_no_cache": generation_equal(post_cached["generation"], post_uncached["generation"]),
        "base_unchanged": base_integrity["unchanged"],
        "base_frozen": len(base_integrity["base_parameters_requiring_grad"]) == 0,
        "reset_exact_base_generation": generation_equal(pre_cached, reset_generation),
    }
    report = {
        "schema_version": "m3bench-editor-single-record-run-v1",
        "stage": args.stage,
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "record_index": args.record_index,
        "record_id": record.record_id,
        "classification": CLASSIFICATION,
        "checks": checks,
        "environment": runtime_environment(args.device),
        "pre_generation_cached": pre_cached,
        "pre_generation_uncached": pre_uncached,
        "pre_nll": pre_nll,
        "edit": edit,
        "post_nll": post_nll,
        "post_generation_cached": post_cached,
        "post_generation_uncached": post_uncached,
        "state": state,
        "state_summary": summary,
        "base_integrity": base_integrity,
        "reset_generation": reset_generation,
        "peak_memory": memory,
        "runtime_seconds": time.perf_counter() - started,
        "code_commit": git("rev-parse", "HEAD"),
    }
    write_json_atomic(output / "single_run.json", report, read_only=True)
    print(json.dumps({"status": report["status"], "method": args.method, "record_id": record.record_id, "checks": checks, "runtime_seconds": report["runtime_seconds"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_single_replay(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    prior = json.loads((output / "single_run.json").read_text(encoding="utf-8"))
    record = load_frozen_smoke_records()[prior["record_index"]]
    started = time.perf_counter()
    runtime = load_runtime(args.device)
    editor = create_editor(args.method, runtime)
    editor.load_editor_state(state_path(args.method, output))
    replay_cached = editor.generate(record, use_cache=True)
    replay_uncached = editor.generate(record, use_cache=False)
    replay_nll = editor.score_target_nll(record)
    base_integrity = editor.base_integrity()
    state_summary = editor.state_summary()
    editor.reset_editor_state()
    reset = runtime.generate(record, use_cache=True)
    checks = {
        "fresh_process_post_generation_exact": generation_equal(
            prior["post_generation_cached"]["generation"], replay_cached["generation"]
        ),
        "fresh_process_cached_no_cache": generation_equal(
            replay_cached["generation"], replay_uncached["generation"]
        ),
        "fresh_process_route_exact": prior["post_generation_cached"]["route"] == replay_cached["route"],
        "fresh_process_nll_exact": prior["post_nll"]["nll"] == replay_nll["nll"],
        "base_unchanged": base_integrity["unchanged"],
        "base_frozen": len(base_integrity["base_parameters_requiring_grad"]) == 0,
        "reset_exact_base_generation": generation_equal(prior["pre_generation_cached"], reset),
    }
    report = {
        "schema_version": "m3bench-editor-single-record-replay-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "record_id": record.record_id,
        "checks": checks,
        "environment": runtime_environment(args.device),
        "replay_cached": replay_cached,
        "replay_uncached": replay_uncached,
        "replay_nll": replay_nll,
        "state_summary": state_summary,
        "base_integrity": base_integrity,
        "reset_generation": reset,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "single_replay.json", report, read_only=True)
    marker = output / (
        "PASS" if report["status"] == "PASS" else f"M3BENCH_EDITOR_RUNTIME_GATE_BLOCKED__{args.method.upper()}__FRESH_REPLAY"
    )
    marker.write_text(report["status"] + "\n", encoding="utf-8")
    os.chmod(marker, 0o444)
    print(json.dumps({"status": report["status"], "method": args.method, "checks": checks}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_smoke_eight(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_frozen_smoke_records()
    runtime = load_runtime(args.device)
    editor = create_editor(args.method, runtime)
    freeze_method_documents(editor, output / "method_lock")
    rows, all_checks = [], []
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        record_output = output / f"record_{index:02d}"
        record_output.mkdir()
        with editor.disabled():
            base_generation = runtime.generate(record, use_cache=True)
        batch = runtime.build_edit_batch(record)
        pre = editor.score_target_nll(record)
        edit = editor.apply_edit(record)
        delta = state_delta(editor, record.record_id)
        post = editor.score_target_nll(record)
        generated = editor.generate(record, use_cache=True)
        base = editor.base_integrity()
        state = editor.save_editor_state(state_path(args.method, record_output))
        editor.reset_editor_state()
        reset_generation = runtime.generate(record, use_cache=True)
        editor.load_editor_state(state_path(args.method, record_output))
        reloaded = editor.generate(record, use_cache=True)
        reloaded_nll = editor.score_target_nll(record)
        radius_mode = "float32" if args.method in {"grace", "balancedit", "belora"} else "exact"
        checks = {
            "multimodal_batch": bool(batch.image_tensor_shape)
            and batch.mask_report()["multimodal_expansion_verified"],
            "target_mask": batch.mask_report()["target_token_count"] > 0
            and batch.mask_report()["prompt_and_image_positions_masked"],
            "finite_update": bool(edit["finite_losses"] and edit["finite_gradients"])
            and torch.isfinite(torch.tensor(delta)).item()
            and delta > 0,
            "post_nll_lower": post["nll"] < pre["nll"],
            "base_unchanged": base["unchanged"],
            "generation_nonempty": bool(generated["generation"]["raw_token_ids"]),
            "save_reload_parity": generation_equal(generated["generation"], reloaded["generation"])
            and route_dict_equal(generated.get("route"), reloaded.get("route"), radius_mode=radius_mode)
            and post["nll"] == reloaded_nll["nll"],
            "reset_parity": generation_equal(base_generation, reset_generation),
            "self_route": route_hits_record(args.method, generated, record.record_id),
        }
        all_checks.extend(checks.values())
        rows.append({
            "opaque_record_index": index,
            "record_id": record.record_id,
            "dataset": record.dataset,
            "question_type": record.question_type,
            "question": record.question,
            "gold_or_reference": record.target,
            "pre_target_nll": pre["nll"],
            "post_target_nll": post["nll"],
            "target_logprob_delta": pre["nll"] - post["nll"],
            "state_delta_norm": delta,
            "base_generation": base_generation,
            "post_generation": generated,
            "reload_generation": reloaded,
            "state": state,
            "checks": checks,
        })
        editor.reset_editor_state()
    summary = {
        "schema_version": "m3bench-editor-eight-record-effect-smoke-v2",
        "status": "PASS" if all(all_checks) else "FAIL",
        "method": args.method,
        "records": 8,
        "record_gate_pass_count": sum(all(row["checks"].values()) for row in rows),
        "raw_output_changed_count": sum(
            row["post_generation"]["generation"]["raw_token_ids"]
            != row["base_generation"]["raw_token_ids"] for row in rows
        ),
        "median_target_nll_decrease": median(row["target_logprob_delta"] for row in rows),
        "route_hit_count": sum(row["checks"]["self_route"] for row in rows),
        "empty_output_count": sum(
            not row["post_generation"]["generation"]["raw_token_ids"] for row in rows
        ),
        "t0_corrected_count": None,
        "correctness_status": "PENDING_FROZEN_EXACT_FUZZY_AND_SEMANTIC_JUDGE",
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "SMOKE_8_PRIVATE.json", {"summary": summary, "rows": rows}, read_only=True)
    write_json_atomic(output / "SMOKE_8_SUMMARY_PREJUDGE.json", summary, read_only=True)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


def mini_records():
    all_records = {record.record_id: record for record in load_frozen_smoke_records()}
    rows = [json.loads(line) for line in MINI_SOURCE.read_text(encoding="utf-8").splitlines() if line]
    records = [all_records[row["record_id"]] for row in rows]
    if [row["formal_sequence_position"] for row in rows] != sorted(row["formal_sequence_position"] for row in rows):
        raise RuntimeError("mini stream order is not frozen formal order")
    return records


def command_stream_run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = mini_records()
    runtime = load_runtime(args.device)
    editor = create_editor(args.method, runtime)
    freeze_method_documents(editor, output / "method_lock")
    started = time.perf_counter()
    steps = []
    all_checks = []
    for step, record in enumerate(records, 1):
        pre_nll = editor.score_target_nll(record)
        prior_adapter_hashes = (
            {item.record_id: editor.adapter_state_sha256(item.record_id) for item in records[: step - 1]}
            if args.method == "belora" else {}
        )
        edit = editor.apply_edit(record)
        post_nll = editor.score_target_nll(record)
        new_generation = editor.generate(record, use_cache=True)
        replays = {
            old.record_id: {
                "generation": editor.generate(old, use_cache=True),
                "target_nll": editor.score_target_nll(old),
            }
            for old in records[:step]
        }
        state = editor.state_summary()
        adapter_hashes = (
            {item.record_id: editor.adapter_state_sha256(item.record_id) for item in records[:step]}
            if args.method == "belora" else {}
        )
        checkpoint = editor.save_editor_state(state_path(args.method, output / "checkpoints", f"step_{step}"))
        base = editor.base_integrity()
        checks = {
            "finite_loss": edit["finite_losses"],
            "finite_gradients": edit["finite_gradients"],
            "current_target_nll_lower": post_nll["nll"] < pre_nll["nll"],
            "current_self_route": route_hits_record(args.method, new_generation, record.record_id),
            "new_edit_generation": bool(new_generation["generation"]["raw_token_ids"]),
            "old_edit_replay_count": len(replays) == step,
            "old_edit_generations_nonempty": all(
                value["generation"]["generation"]["raw_token_ids"] for value in replays.values()
            ),
            "state_count": state_count(editor) == step,
            "base_unchanged": base["unchanged"],
            "checkpoint_nonempty": checkpoint["size_bytes"] > 0,
        }
        if args.method == "belora":
            checks.update({
                "unique_logical_ids": len(editor.edit_to_adapter) == step,
                "unique_adapter_names": len(set(editor.edit_to_adapter.values())) == step,
                "prior_adapter_hashes_unchanged": all(
                    adapter_hashes[key] == value for key, value in prior_adapter_hashes.items()
                ),
                "all_self_routes": all(
                    route_hits_record(args.method, value["generation"], key)
                    for key, value in replays.items()
                ),
            })
        all_checks.extend(checks.values())
        steps.append(
            {
                "step": step,
                "record_id": record.record_id,
                "pre_target_nll": pre_nll,
                "post_target_nll": post_nll,
                "edit": edit,
                "new_generation": new_generation,
                "replays": replays,
                "state_summary": state,
                "adapter_hashes": adapter_hashes,
                "checkpoint": checkpoint,
                "base_integrity": base,
                "checks": checks,
            }
        )
    report = {
        "schema_version": "m3bench-editor-four-edit-stream-run-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(all_checks) else "FAIL",
        "method": args.method,
        "classification": CLASSIFICATION,
        "record_ids": [record.record_id for record in records],
        "steps": steps,
        "runtime_seconds": time.perf_counter() - started,
        "environment": runtime_environment(args.device),
    }
    write_json_atomic(output / "stream_run.json", report, read_only=True)
    print(json.dumps({"status": report["status"], "method": args.method, "steps": len(steps), "runtime_seconds": report["runtime_seconds"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_stream_replay(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    source = Path(args.source_dir) if args.source_dir else output
    prior_path = source / "stream_run.json"
    checkpoint_path = state_path(args.method, source / "checkpoints", "step_4")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    records = mini_records()
    radius_mode = args.grace_route_radius_canonicalization
    if radius_mode != "exact" and args.method != "grace":
        raise RuntimeError("route-radius canonicalization is authorized only for GRACE")
    runtime = load_runtime(args.device)
    editor = create_editor(args.method, runtime)
    editor.load_editor_state(checkpoint_path)
    replays = {
        record.record_id: {
            "generation": editor.generate(record, use_cache=True),
            "target_nll": editor.score_target_nll(record),
        }
        for record in records
    }
    expected = prior["steps"][-1]["replays"]
    per_record = {}
    replay_diagnostics = {}
    for record in records:
        record_id = record.record_id
        expected_generation = expected[record_id]["generation"]["generation"]
        replay_generation = replays[record_id]["generation"]["generation"]
        expected_route = expected[record_id]["generation"]["route"]
        replay_route = replays[record_id]["generation"]["route"]
        generation_exact = generation_equal(expected_generation, replay_generation)
        route_exact = route_dict_equal(
            expected_route,
            replay_route,
            radius_mode=radius_mode,
        )
        nll_exact = expected[record_id]["target_nll"]["nll"] == replays[record_id]["target_nll"]["nll"]
        per_record[record_id] = generation_exact and route_exact and nll_exact
        expected_radius = expected_route.get("radius") if expected_route else None
        replay_radius = replay_route.get("radius") if replay_route else None
        replay_diagnostics[record_id] = {
            "generation_exact": generation_exact,
            "raw_token_ids_exact": expected_generation["raw_token_ids"]
            == replay_generation["raw_token_ids"],
            "decoded_text_exact": expected_generation["decoded_text"]
            == replay_generation["decoded_text"],
            "route_exact_under_contract": route_exact,
            "target_nll_exact": nll_exact,
            "radius_mode": radius_mode,
            "expected_radius": expected_radius,
            "replay_radius": replay_radius,
            "expected_radius_float32": canonical_float32(expected_radius)
            if expected_radius is not None
            else None,
            "replay_radius_float32": canonical_float32(replay_radius)
            if replay_radius is not None
            else None,
        }
    base = editor.base_integrity()
    checks = {
        "all_four_fresh_process_replay_exact": all(per_record.values()),
        "state_entry_count_exact": state_count(editor) == 4,
        "base_unchanged": base["unchanged"],
        "base_frozen": len(base["base_parameters_requiring_grad"]) == 0,
    }
    report = {
        "schema_version": "m3bench-editor-four-edit-stream-replay-v2",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "checks": checks,
        "per_record_exact": per_record,
        "replay_diagnostics": replay_diagnostics,
        "replays": replays,
        "state_summary": editor.state_summary(),
        "base_integrity": base,
        "environment": runtime_environment(args.device),
        "source": {
            "source_dir": str(source),
            "stream_run_path": str(prior_path),
            "stream_run_sha256": sha256_file(prior_path),
            "checkpoint_path": str(checkpoint_path),
        },
        "comparison_contract": {
            "generation": "exact decoded text, raw token IDs, and sequence contract",
            "route_non_radius_fields": "exact",
            "route_radius": radius_mode,
            "numeric_tolerance": None,
        },
    }
    write_json_atomic(output / "stream_replay.json", report, read_only=True)
    marker = output / ("PASS" if report["status"] == "PASS" else f"M3BENCH_EDITOR_SMOKE_BLOCKED__{args.method.upper()}__STREAM_REPLAY")
    marker.write_text(report["status"] + "\n", encoding="utf-8")
    os.chmod(marker, 0o444)
    print(json.dumps({"status": report["status"], "method": args.method, "checks": checks}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--device", default="cuda:0")
    inventory.set_defaults(func=command_inventory)
    for name, func in (
        ("single-run", command_single_run),
        ("single-replay", command_single_replay),
    ):
        command = sub.add_parser(name)
        command.add_argument("--method", required=True, choices=METHODS)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--record-index", type=int, default=0)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--stage", default="preflight")
        command.set_defaults(func=func)
    smoke = sub.add_parser("smoke-eight")
    smoke.add_argument("--method", required=True, choices=METHODS)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--output-dir", required=True)
    smoke.set_defaults(func=command_smoke_eight)
    stream_run = sub.add_parser("stream-run")
    stream_run.add_argument("--method", required=True, choices=METHODS)
    stream_run.add_argument("--device", default="cuda:0")
    stream_run.add_argument("--output-dir", required=True)
    stream_run.set_defaults(func=command_stream_run)
    stream_replay = sub.add_parser("stream-replay")
    stream_replay.add_argument("--method", required=True, choices=METHODS)
    stream_replay.add_argument("--device", default="cuda:0")
    stream_replay.add_argument("--output-dir", required=True)
    stream_replay.add_argument("--source-dir")
    stream_replay.add_argument(
        "--grace-route-radius-canonicalization", choices=("exact", "float32"), default="exact"
    )
    stream_replay.set_defaults(func=command_stream_replay)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
