#!/usr/bin/env python3
"""Raw-only formal M3Bench single and sequential paper-spec editor runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from m3bench_repro.editors.llava_runtime import (
    EditorRecord,
    LlavaMedEditorRuntime,
    canonical_sha256,
)
from m3bench_repro.editors.methods import LoraPaperSpecEditor, PaperSpecEditor, create_editor
from m3bench_repro.editors.routing import route_dict_equal


METHODS = ("lora", "grace", "balancedit", "belora")
DEFAULT_RECORDS_PATH = "inputs/frozen/FORMAL_EDITOR_RECORDS_200.jsonl"
DEFAULT_SEQUENCE_LABEL = "M3BENCH_FORMAL_ORIGINAL_200"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_frozen_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        preserve_orphan(temporary, path.parent / "orphans", temporary.name)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    os.chmod(path, 0o444)


def write_frozen_json(path: Path, value: object) -> None:
    write_frozen_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def tree_summary(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size, "file_count": 1}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = str(item.relative_to(path))
        file_sha = sha256(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": sum(item.stat().st_size for item in files),
        "file_count": len(files),
    }


def freeze_tree(path: Path) -> None:
    if path.is_file():
        os.chmod(path, 0o444)
        return
    for item in path.rglob("*"):
        os.chmod(item, 0o555 if item.is_dir() else 0o444)
    os.chmod(path, 0o555)


def preserve_orphan(path: Path, orphan_root: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    orphan_root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().replace(":", "").replace("-", "")
    target = orphan_root / f"{label}.{timestamp}"
    counter = 1
    while target.exists():
        target = orphan_root / f"{label}.{timestamp}.{counter:03d}"
        counter += 1
    before = tree_summary(path)
    os.replace(path, target)
    return {"source": before, "preserved_as": tree_summary(target)}


def full_base_hash(runtime: LlavaMedEditorRuntime) -> str:
    guard = runtime.base_guard
    if guard is None:
        raise RuntimeError("base guard is unavailable")
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in zip(guard.names, guard.parameters, strict=True):
            value = parameter.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_environment(device: str) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.device(device))
    return {
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
    }


def assert_authorized_device() -> None:
    override_name = "M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES"
    expected_visible_device = os.environ.get(override_name, "2")
    allowed = {
        value.strip()
        for value in os.environ.get("M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES", "2,3").split(",")
        if value.strip()
    }
    if expected_visible_device not in allowed:
        raise RuntimeError(f"invalid {override_name}={expected_visible_device!r}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible_device:
        raise RuntimeError(
            "formal runner CUDA_VISIBLE_DEVICES does not match the authorized "
            f"physical device {expected_visible_device}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal runner requires exactly one visible CUDA device")
    expected_uuid = os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID")
    if not expected_uuid:
        raise RuntimeError("M3BENCH_FORMAL_EXPECTED_GPU_UUID is required")
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", expected_visible_device],
        text=True,
    ).strip()
    if actual_uuid != expected_uuid:
        raise RuntimeError(
            f"formal runner GPU UUID mismatch for physical device {expected_visible_device}"
        )


def resolve_run_path(run: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run / path


def load_records(run: Path, records_path: str, expected_count: int) -> list[EditorRecord]:
    rows = read_jsonl(resolve_run_path(run, records_path))
    records = [EditorRecord.from_dict(row) for row in rows]
    if len(records) != expected_count or len({record.record_id for record in records}) != expected_count:
        raise RuntimeError("formal editor record lock failed")
    if [record.formal_sequence_position for record in records] != list(range(1, expected_count + 1)):
        raise RuntimeError("formal editor record order drift")
    return records


def sequence_lock(run: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    code_paths = (
        Path(__file__).resolve(),
        root / "m3bench_repro/editors/llava_runtime.py",
        root / "m3bench_repro/editors/methods.py",
        root / "m3bench_repro/editors/routed_layers.py",
        root / "m3bench_repro/editors/routing.py",
    )
    return {
        "sequence_label": args.sequence_label,
        "expected_record_count": args.expected_record_count,
        "prefixes": list(args.prefix_values),
        "final_prefix": args.final_prefix,
        "records_sha256": sha256(resolve_run_path(run, args.records_path)),
        "model_lock_sha256": sha256(run / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json"),
        "method_config_bundle_sha256": sha256(run / "locks/FORMAL_METHOD_CONFIG_BUNDLE.json"),
        "canonical_runtime_lock_sha256": sha256(run / "locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json"),
        "code_sha256": {str(path.relative_to(root)): sha256(path) for path in code_paths},
    }


def checkpoint_manifest_path(output: Path, position: int) -> Path:
    return output / "checkpoints" / f"step_{position:03d}.manifest.json"


def load_catalog(run: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(run / "inputs/frozen/FORMAL_PROBE_CATALOG.jsonl")
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["edit_id"], []).append(row)
    return result


def normalized_generation_contract(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value.get(key)
        for key in ("batch_size", "do_sample", "max_new_tokens", "num_beams", "temperature", "use_cache")
    }
    if result["do_sample"] is False and result["temperature"] is None:
        result["temperature"] = 0
    return result


def normalized_inventory_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Keep model topology immutable while ignoring host-specific provenance fields."""
    linears = [
        {key: item.get(key) for key in (
            "path", "block", "projection", "in_features", "out_features",
            "bias", "parameter_count", "dtype",
        )}
        for item in value.get("candidate_internal_linears", [])
    ]
    return {
        key: value.get(key)
        for key in (
            "model_class", "language_block_count", "language_blocks",
            "final_block_path", "final_mlp_path", "projector_candidates",
            "projector_path", "vision_encoder_candidates", "vision_encoder_path",
            "model_dtype", "total_model_parameters",
        )
    } | {"candidate_internal_linears": linears}


def normalized_target_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "methods": {
            method: value.get(method)
            for method in METHODS
        },
        "projector_excluded": value.get("projector_excluded"),
        "vision_encoder_excluded": value.get("vision_encoder_excluded"),
        "target_lists_sha256": value.get("target_lists_sha256"),
    }


def load_runtime(run: Path, device: str) -> LlavaMedEditorRuntime:
    canonical_runtime = read_json(run / "locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json")
    if canonical_runtime.get("selected_runtime") != "runtime_b_official_native":
        raise RuntimeError("formal runner requires the frozen official-native runtime")
    generation_path = run / "inputs/frozen/llava_med_generation_frozen.json"
    if normalized_generation_contract(read_json(generation_path)) != canonical_runtime.get("generation"):
        raise RuntimeError("formal generation config differs from the canonical runtime lock")
    runtime = LlavaMedEditorRuntime(
        device=device,
        run_root=run,
        generation_config_path=generation_path,
        loader_mode="official_native",
    )
    runtime.load_frozen_backbone(seed=20260828)
    inventory, target_lock = runtime.resolve_module_inventory(freeze=False)
    expected_inventory = read_json(run / "inputs/frozen/LLAVA_MED_MODULE_INVENTORY.json")
    expected_targets = read_json(run / "inputs/frozen/LLAVA_MED_EDIT_TARGET_LOCK.json")
    if normalized_inventory_contract(inventory) != normalized_inventory_contract(expected_inventory):
        raise RuntimeError("formal module inventory differs from frozen parent")
    if normalized_target_contract(target_lock) != normalized_target_contract(expected_targets):
        raise RuntimeError("formal target lock differs from frozen parent")
    return runtime


def create_formal_editor(
    method: str,
    runtime: LlavaMedEditorRuntime,
    output: Path,
    *,
    run: Path,
    sequential: bool,
) -> PaperSpecEditor:
    state_store = output / "state_store" if sequential and method == "balancedit" else None
    editor = create_editor(method, runtime, balancedit_inactive_store_dir=state_store)
    expected = read_json(run / "locks/FORMAL_METHOD_CONFIG_BUNDLE.json")["method_configs"][method]
    expected_hash = expected.get("config_sha256")
    if expected_hash != canonical_sha(editor.config_lock()):
        raise RuntimeError(f"{method} runtime config differs from the frozen method bundle")
    return editor


def editor_empty(editor: PaperSpecEditor, method: str) -> dict[str, Any]:
    summary = editor.state_summary()
    checks = {"edit_history_empty": summary.get("edit_history") == []}
    if method == "lora":
        if not isinstance(editor, LoraPaperSpecEditor):
            raise TypeError("LoRA editor type mismatch")
        checks["initial_adapter_exact"] = editor.adapter_state_sha256() == editor.initial_adapter_sha256
    else:
        checks["entry_count_zero"] = summary.get("entry_count") == 0
    return {"pass": all(checks.values()), "checks": checks, "state_summary": summary}


def grace_state_checks(
    summary: dict[str, Any],
    expected_edits: int,
    *,
    prior_entry_count: int | None = None,
    insertion_action: str | None = None,
) -> dict[str, bool]:
    """Validate GRACE's request-to-entry state without rejecting source-compatible reuse."""
    entry_count = summary.get("entry_count")
    logical_edit_ids = summary.get("logical_edit_ids")
    radii = summary.get("radii")
    entry_count_valid = (
        isinstance(entry_count, int)
        and not isinstance(entry_count, bool)
        and 0 < entry_count <= expected_edits
    )
    checks = {
        "entry_count_bounded": entry_count_valid,
        "logical_entry_count_exact": isinstance(logical_edit_ids, list)
        and entry_count_valid
        and len(logical_edit_ids) == entry_count,
        "logical_entry_ids_unique": isinstance(logical_edit_ids, list)
        and len(logical_edit_ids) == len(set(logical_edit_ids)),
        "radii_entry_count_exact": isinstance(radii, list)
        and entry_count_valid
        and len(radii) == entry_count,
        "value_entry_count_exact": summary.get("value_entry_count") == entry_count,
        "requested_mapping_count_exact": summary.get("requested_to_effective_count")
        == expected_edits,
        "requested_mapping_keys_match_history": summary.get(
            "requested_mapping_keys_match_history"
        )
        is True,
        "requested_mapping_values_resident": summary.get(
            "requested_mapping_values_resident"
        )
        is True,
    }
    if prior_entry_count is not None:
        checks["insert_action_known"] = insertion_action in {
            "insert",
            "insert_far",
            "collision_split",
            "same_label_reuse",
        }
        expected_delta = 0 if insertion_action == "same_label_reuse" else 1
        checks["entry_count_transition_exact"] = (
            entry_count_valid and entry_count == prior_entry_count + expected_delta
        )
    return checks


def method_state_contract(method: str, summary: dict[str, Any], expected_edits: int) -> bool:
    if method == "lora":
        return summary.get("adapter_parameter_count", 0) > 0
    if method == "grace":
        return all(grace_state_checks(summary, expected_edits).values())
    return summary.get("entry_count") == expected_edits


def save_state_atomic(editor: PaperSpecEditor, method: str, final: Path) -> dict[str, Any]:
    if final.exists():
        raise RuntimeError(f"refusing to overwrite editor state: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + ".tmp")
    if temporary.exists():
        preserve_orphan(
            temporary,
            final.parent / "orphans",
            f"{temporary.name}.incomplete",
        )
    editor.save_editor_state(temporary)
    os.replace(temporary, final)
    freeze_tree(final)
    return tree_summary(final)


def route_equal(method: str, expected: dict[str, Any] | None, actual: dict[str, Any] | None) -> bool:
    return route_dict_equal(
        expected,
        actual,
        radius_mode="float32" if method == "grace" else "exact",
    )


def probe_record(row: dict[str, Any]) -> EditorRecord:
    return EditorRecord(
        record_id=row["probe_id"],
        dataset=row["dataset"],
        question=row["question"],
        target=row["reference"],
        official_rephrase=row["question"],
        image_path=Path(row["image_path"]),
        relative_image_path=row["image_path"],
        formal_sequence_position=int(row["sequence_position"]),
        question_type=row["task"],
    )


def generate_probe(editor: PaperSpecEditor, row: dict[str, Any], generation_lock_hash: str) -> dict[str, Any]:
    started = time.perf_counter()
    generated = editor.generate(probe_record(row), use_cache=True)
    return {
        "probe_id": row["probe_id"],
        "task": row["task"],
        "sequence_position": row["sequence_position"],
        "variant_type": row.get("variant_type"),
        "raw_token_ids": generated["generation"]["raw_token_ids"],
        "raw_text": generated["generation"]["decoded_text"],
        "sequence_contract": generated["generation"]["sequence_contract"],
        "route": generated["route"],
        "generation_lock_hash": generation_lock_hash,
        "runtime_seconds": time.perf_counter() - started,
        "exit_status": "success",
    }


def load_single_events(run: Path, task: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(run / "inputs/frozen/FORMAL_SINGLE_EVENT_CATALOG.jsonl")
        if row["task"] == task
    ]
    expected = read_json(run / "locks/FORMAL_EXPECTED_COUNTS.json")["single"][
        "events_per_task_per_method"
    ][task]
    positions = [int(row["event_position"]) for row in rows]
    if len(rows) != expected or positions != list(range(1, expected + 1)):
        raise RuntimeError(f"{task} single-event catalog lock failed")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{task} single-event IDs are not unique")
    return rows


def command_single_events(args: argparse.Namespace) -> None:
    assert_authorized_device()
    run, output = Path(args.run_root), Path(args.output_dir)
    if output.exists():
        raise RuntimeError(f"single-event chunk output already exists: {output}")
    events = load_single_events(run, args.task)
    if args.start < 1 or args.end > len(events) or args.end < args.start or args.end - args.start + 1 > 25:
        raise ValueError("single-event chunk must be a valid <=25-event interval")
    selected = events[args.start - 1 : args.end]
    output.mkdir(parents=True)
    generation_lock_hash = sha256(run / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json")
    runtime = load_runtime(run, args.device)
    editor = create_formal_editor(args.method, runtime, output, run=run, sequential=False)
    environment = runtime_environment(args.device)
    chunk_base_before = full_base_hash(runtime)
    artifacts = []
    started = time.perf_counter()
    for event in selected:
        position = int(event["event_position"])
        record = EditorRecord.from_dict(event["edit_record"])
        event_dir = output / f"event_{position:04d}"
        event_dir.mkdir()
        empty_before = editor_empty(editor, args.method)
        if not empty_before["pass"]:
            raise RuntimeError(f"nonempty editor state before event {position}")
        pre_nll = editor.score_target_nll(record)
        event_started = time.perf_counter()
        with runtime.peak_memory() as peak:
            edit = editor.apply_edit(record)
            post_nll = editor.score_target_nll(record)
            state_summary = editor.state_summary()
            state = save_state_atomic(
                editor,
                args.method,
                event_dir / ("editor_state" if args.method == "lora" else "editor_state.pt"),
            )
            outputs = [generate_probe(editor, row, generation_lock_hash) for row in event["probes"]]
        base_after_edit = editor.base_integrity()
        editor.reset_editor_state()
        empty_after = editor_empty(editor, args.method)
        base_after_reset = editor.base_integrity()
        if not empty_after["pass"] or not base_after_edit["unchanged"] or not base_after_reset["unchanged"]:
            raise RuntimeError(f"state isolation/base integrity failure at event {position}")
        report = {
            "schema_version": "m3bench-formal-single-event-raw-v1",
            "created_at_utc": utc_now(),
            "status": "PASS",
            "method": args.method,
            "mode": "single",
            "task": args.task,
            "event_position": position,
            "event_id": event["event_id"],
            "edit_record_id": record.record_id,
            "router_positive_source": event["edit_record"].get("router_positive_source"),
            "pre_target_nll": pre_nll,
            "post_target_nll": post_nll,
            "edit": edit,
            "editor_state": state,
            "state_summary": state_summary,
            "base_integrity_after_edit": base_after_edit,
            "base_integrity_after_reset": base_after_reset,
            "empty_state_before": empty_before,
            "empty_state_after": empty_after,
            "generation_lock_hash": generation_lock_hash,
            "raw_outputs": outputs,
            "raw_output_count": len(outputs),
            "runtime_seconds": time.perf_counter() - event_started,
            "peak_gpu_memory": peak,
            "exit_status": "success",
            "semantic_metrics_computed": False,
        }
        report_path = event_dir / "raw_event.json"
        write_frozen_json(report_path, report)
        artifacts.append({
            "event_position": position,
            "event_id": event["event_id"],
            "report_sha256": sha256(report_path),
            "state_sha256": state["sha256"],
            "raw_output_count": len(outputs),
        })
        os.chmod(event_dir, 0o555)
        torch.cuda.empty_cache()
    chunk_base_after = full_base_hash(runtime)
    checks = {
        "event_count_exact": len(artifacts) == len(selected),
        "event_positions_exact": [item["event_position"] for item in artifacts]
        == list(range(args.start, args.end + 1)),
        "base_full_hash_exact": chunk_base_before == chunk_base_after,
        "all_raw_counts_positive": all(item["raw_output_count"] > 0 for item in artifacts),
        "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
        "editor_empty_at_end": editor_empty(editor, args.method)["pass"],
    }
    manifest = {
        "schema_version": "m3bench-formal-single-event-chunk-manifest-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "task": args.task,
        "start": args.start,
        "end": args.end,
        "checks": checks,
        "events": artifacts,
        "raw_output_count": sum(item["raw_output_count"] for item in artifacts),
        "runtime_seconds": time.perf_counter() - started,
        "environment": environment,
        "semantic_metrics_computed": False,
    }
    write_frozen_json(output / "CHUNK_MANIFEST.json", manifest)
    write_frozen_text(output / ("PASS" if manifest["status"] == "PASS" else "FAIL"), manifest["status"] + "\n")
    os.chmod(output, 0o555)
    print(json.dumps({"status": manifest["status"], "method": args.method, "task": args.task, "start": args.start, "end": args.end, "raw_outputs": manifest["raw_output_count"]}, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(1)


def command_single_chunk(args: argparse.Namespace) -> None:
    assert_authorized_device()
    run, output = Path(args.run_root), Path(args.output_dir)
    if output.exists():
        raise RuntimeError(f"single chunk output already exists: {output}")
    output.mkdir(parents=True)
    records = load_records(run, args.records_path, args.expected_record_count)
    if (
        args.start < 1
        or args.end > args.expected_record_count
        or args.end < args.start
        or args.end - args.start + 1 > 25
    ):
        raise ValueError("single chunk must be a valid <=25-record interval")
    selected = records[args.start - 1 : args.end]
    catalog = load_catalog(run)
    generation_lock_hash = sha256(run / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json")
    runtime = load_runtime(run, args.device)
    editor = create_formal_editor(args.method, runtime, output, run=run, sequential=False)
    environment = runtime_environment(args.device)
    chunk_base_before = full_base_hash(runtime)
    record_artifacts = []
    started = time.perf_counter()
    for record in selected:
        position = record.formal_sequence_position
        record_dir = output / f"record_{position:03d}"
        if record_dir.exists():
            raise RuntimeError(f"record output collision: {record_dir}")
        record_dir.mkdir()
        empty_before = editor_empty(editor, args.method)
        if not empty_before["pass"]:
            raise RuntimeError(f"nonempty editor state before record {position}")
        pre_nll = editor.score_target_nll(record)
        record_started = time.perf_counter()
        with runtime.peak_memory() as peak:
            edit = editor.apply_edit(record)
            post_nll = editor.score_target_nll(record)
            state_summary = editor.state_summary()
            state = save_state_atomic(
                editor,
                args.method,
                record_dir / ("editor_state" if args.method == "lora" else "editor_state.pt"),
            )
            outputs = [generate_probe(editor, row, generation_lock_hash) for row in catalog[record.record_id]]
        base_after_edit = editor.base_integrity()
        editor.reset_editor_state()
        empty_after = editor_empty(editor, args.method)
        base_after_reset = editor.base_integrity()
        if not empty_after["pass"] or not base_after_edit["unchanged"] or not base_after_reset["unchanged"]:
            raise RuntimeError(f"state isolation/base integrity failure at record {position}")
        report = {
            "schema_version": "m3bench-formal-single-record-raw-v1",
            "created_at_utc": utc_now(),
            "status": "PASS",
            "method": args.method,
            "mode": "single",
            "formal_position": position,
            "edit_record_id": record.record_id,
            "pre_target_nll": pre_nll,
            "post_target_nll": post_nll,
            "edit": edit,
            "editor_state": state,
            "state_summary": state_summary,
            "base_integrity_after_edit": base_after_edit,
            "base_integrity_after_reset": base_after_reset,
            "empty_state_before": empty_before,
            "empty_state_after": empty_after,
            "generation_lock_hash": generation_lock_hash,
            "raw_outputs": outputs,
            "raw_output_count": len(outputs),
            "runtime_seconds": time.perf_counter() - record_started,
            "peak_gpu_memory": peak,
            "exit_status": "success",
            "semantic_metrics_computed": False,
        }
        report_path = record_dir / "raw_record.json"
        write_frozen_json(report_path, report)
        record_artifacts.append(
            {
                "position": position,
                "record_id": record.record_id,
                "report_sha256": sha256(report_path),
                "state_sha256": state["sha256"],
                "raw_output_count": len(outputs),
            }
        )
        os.chmod(record_dir, 0o555)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    chunk_base_after = full_base_hash(runtime)
    chunk_checks = {
        "record_count_exact": len(record_artifacts) == len(selected),
        "record_positions_exact": [item["position"] for item in record_artifacts]
        == list(range(args.start, args.end + 1)),
        "base_full_hash_exact": chunk_base_before == chunk_base_after,
        "all_raw_counts_positive": all(item["raw_output_count"] > 0 for item in record_artifacts),
        "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
        "editor_empty_at_end": editor_empty(editor, args.method)["pass"],
    }
    manifest = {
        "schema_version": "m3bench-formal-single-chunk-manifest-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(chunk_checks.values()) else "FAIL",
        "method": args.method,
        "sequence": sequence_lock(run, args),
        "start": args.start,
        "end": args.end,
        "checks": chunk_checks,
        "base_full_sha256_before": chunk_base_before,
        "base_full_sha256_after": chunk_base_after,
        "records": record_artifacts,
        "raw_output_count": sum(item["raw_output_count"] for item in record_artifacts),
        "runtime_seconds": time.perf_counter() - started,
        "environment": environment,
        "semantic_metrics_computed": False,
    }
    write_frozen_json(output / "CHUNK_MANIFEST.json", manifest)
    write_frozen_text(output / ("PASS" if manifest["status"] == "PASS" else "FAIL"), manifest["status"] + "\n")
    os.chmod(output, 0o555)
    print(json.dumps({"status": manifest["status"], "method": args.method, "start": args.start, "end": args.end, "raw_outputs": manifest["raw_output_count"], "runtime_seconds": manifest["runtime_seconds"]}, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(1)


def checkpoint_path(output: Path, method: str, position: int) -> Path:
    base = output / "checkpoints" / f"step_{position:03d}"
    return base if method == "lora" else base.with_suffix(".pt")


def generate_edit_outputs(
    editor: PaperSpecEditor,
    edit_id: str,
    catalog: dict[str, list[dict[str, Any]]],
    generation_lock_hash: str,
) -> list[dict[str, Any]]:
    return [generate_probe(editor, row, generation_lock_hash) for row in catalog[edit_id]]


def run_prefix(
    *,
    run: Path,
    output: Path,
    editor: PaperSpecEditor,
    records: list[EditorRecord],
    catalog: dict[str, list[dict[str, Any]]],
    prefix: int,
    method: str,
    generation_lock_hash: str,
    base_full_sha256: str,
    sequence: dict[str, Any],
) -> dict[str, Any]:
    prefix_dir = output / "prefixes" / f"prefix_{prefix:03d}"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prefix_dir / "PREFIX_MANIFEST.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") != "PASS" or not all(existing.get("checks", {}).values()):
            raise RuntimeError(f"existing prefix manifest is not PASS: {manifest_path}")
        write_frozen_text(prefix_dir / "PASS", "PASS\n")
        return existing
    started = time.perf_counter()
    artifacts = []
    task_counts: dict[str, int] = {}
    with editor.runtime.peak_memory() as peak:
        for record in records[:prefix]:
            path = prefix_dir / f"edit_{record.formal_sequence_position:03d}.json"
            if path.exists():
                report = read_json(path)
            else:
                outputs = generate_edit_outputs(editor, record.record_id, catalog, generation_lock_hash)
                report = {
                    "schema_version": "m3bench-formal-sequential-prefix-edit-raw-v1",
                    "created_at_utc": utc_now(),
                    "status": "PASS",
                    "method": method,
                    "mode": "sequential",
                    "prefix": prefix,
                    "formal_position": record.formal_sequence_position,
                    "edit_record_id": record.record_id,
                    "raw_outputs": outputs,
                    "raw_output_count": len(outputs),
                    "semantic_metrics_computed": False,
                }
                write_frozen_json(path, report)
            for item in report["raw_outputs"]:
                task_counts[item["task"]] = task_counts.get(item["task"], 0) + 1
            artifacts.append({"path": path.name, "sha256": sha256(path), "raw_output_count": report["raw_output_count"]})
    expected = read_json(run / "locks/FORMAL_EXPECTED_COUNTS.json")["sequential"]["prefixes"][str(prefix)]
    checks = {
        "edit_files_exact": len(artifacts) == prefix,
        "raw_output_count_exact": sum(item["raw_output_count"] for item in artifacts)
        == expected["raw_outputs_per_method"],
        "task_counts_exact": task_counts == {
            key: value for key, value in expected["raw_outputs_per_task_per_method"].items() if value
        },
        "base_full_hash_exact": full_base_hash(editor.runtime) == base_full_sha256,
        "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
    }
    manifest = {
        "schema_version": "m3bench-formal-sequential-prefix-manifest-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": method,
        "sequence": sequence,
        "prefix": prefix,
        "checks": checks,
        "task_counts": task_counts,
        "raw_output_count": sum(item["raw_output_count"] for item in artifacts),
        "artifacts": artifacts,
        "peak_gpu_memory": peak,
        "runtime_seconds": time.perf_counter() - started,
        "semantic_metrics_computed": False,
    }
    write_frozen_json(manifest_path, manifest)
    write_frozen_text(prefix_dir / ("PASS" if manifest["status"] == "PASS" else "FAIL"), manifest["status"] + "\n")
    if manifest["status"] != "PASS":
        raise RuntimeError(f"prefix {prefix} closure failed: {checks}")
    return manifest


def command_sequential_run(args: argparse.Namespace) -> None:
    assert_authorized_device()
    run, output = Path(args.run_root), Path(args.output_dir)
    final_report_path = output / "SEQUENTIAL_RUN_REPORT.json"
    if final_report_path.exists():
        existing = read_json(final_report_path)
        if (
            existing.get("status") != "PASS"
            or not all(existing.get("checks", {}).values())
            or existing.get("sequence") != sequence_lock(run, args)
        ):
            raise RuntimeError("existing sequential report is not PASS")
        write_frozen_text(output / "RUN_PASS", "PASS\n")
        print(json.dumps({"status": "PASS", "method": args.method, "resumed_from_final_report": True}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(run, args.records_path, args.expected_record_count)
    catalog = load_catalog(run)
    generation_lock_hash = sha256(run / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json")
    runtime = load_runtime(run, args.device)
    editor = create_formal_editor(args.method, runtime, output, run=run, sequential=True)
    environment = runtime_environment(args.device)
    base_full = full_base_hash(runtime)
    locked_sequence = sequence_lock(run, args)
    steps_dir = output / "steps"
    steps_dir.mkdir(exist_ok=True)
    existing_positions = sorted(
        int(path.stem.split("_")[-1]) for path in steps_dir.glob("step_*.json")
    )
    if existing_positions and existing_positions != list(range(1, max(existing_positions) + 1)):
        raise RuntimeError("non-contiguous sequential step reports")
    contiguous = max(existing_positions, default=0)
    orphan_audits = []
    next_position = contiguous + 1
    if next_position <= args.expected_record_count:
        next_checkpoint = checkpoint_path(output, args.method, next_position)
        for candidate, label in (
            (next_checkpoint, f"step_{next_position:03d}.checkpoint_without_report"),
            (next_checkpoint.with_name(next_checkpoint.name + ".tmp"), f"step_{next_position:03d}.temporary"),
        ):
            audit = preserve_orphan(candidate, output / "orphans", label)
            if audit is not None:
                orphan_audits.append(audit)
    resume_audit = None
    if contiguous:
        state = checkpoint_path(output, args.method, contiguous)
        if not state.exists():
            raise RuntimeError("latest sequential report lacks checkpoint")
        state_manifest = checkpoint_manifest_path(output, contiguous)
        if not state_manifest.exists() or read_json(state_manifest).get("sequence") != locked_sequence:
            raise RuntimeError("latest sequential checkpoint lock differs from current run")
        editor.load_editor_state(state)
        prior = read_json(steps_dir / f"step_{contiguous:03d}.json")
        replay = editor.generate(records[contiguous - 1], use_cache=True)
        expected = prior["post_edit_generation"]
        resume_checks = {
            "raw_tokens_exact": replay["generation"]["raw_token_ids"] == expected["generation"]["raw_token_ids"],
            "decoded_text_exact": replay["generation"]["decoded_text"] == expected["generation"]["decoded_text"],
            "sequence_contract_exact": replay["generation"]["sequence_contract"] == expected["generation"]["sequence_contract"],
            "route_exact": route_equal(args.method, expected["route"], replay["route"]),
            "state_history_exact": editor.state_summary().get("edit_history")
            == [record.record_id for record in records[:contiguous]],
            "base_full_hash_exact": full_base_hash(runtime) == base_full,
            "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
        }
        resume_audit = {
            "created_at_utc": utc_now(),
            "position": contiguous,
            "status": "PASS" if all(resume_checks.values()) else "FAIL",
            "checks": resume_checks,
        }
        write_frozen_json(output / f"RESUME_AUDIT_{contiguous:03d}_{utc_now().replace(':', '')}.json", resume_audit)
        if resume_audit["status"] != "PASS":
            raise RuntimeError("sequential resume replay failed")
        if contiguous in args.prefix_values and not (
            output / "prefixes" / f"prefix_{contiguous:03d}" / "PASS"
        ).exists():
            run_prefix(
                run=run,
                output=output,
                editor=editor,
                records=records,
                catalog=catalog,
                prefix=contiguous,
                method=args.method,
                generation_lock_hash=generation_lock_hash,
                base_full_sha256=base_full,
                sequence=locked_sequence,
            )
    started = time.perf_counter()
    for position in range(contiguous + 1, args.expected_record_count + 1):
        record = records[position - 1]
        step_started = time.perf_counter()
        pre_nll = editor.score_target_nll(record)
        pre_state_summary = editor.state_summary()
        with runtime.peak_memory() as peak:
            edit = editor.apply_edit(record)
            post_nll = editor.score_target_nll(record)
            post_generation = editor.generate(record, use_cache=True)
            state_summary = editor.state_summary()
            state = save_state_atomic(editor, args.method, checkpoint_path(output, args.method, position))
            write_frozen_json(
                checkpoint_manifest_path(output, position),
                {
                    "schema_version": "m3bench-formal-checkpoint-manifest-v2",
                    "created_at_utc": utc_now(),
                    "method": args.method,
                    "position": position,
                    "sequence": locked_sequence,
                    "state": state,
                },
            )
        base = editor.base_integrity()
        expected_history = [item.record_id for item in records[:position]]
        state_checks = {
            "edit_history_exact": state_summary.get("edit_history") == expected_history,
            "base_sentinel_unchanged": base["unchanged"],
            "checkpoint_nonempty": state["size_bytes"] > 0,
            "finite_losses": edit["finite_losses"],
            "finite_gradients": edit["finite_gradients"],
        }
        if args.method == "lora":
            state_checks["adapter_nonempty"] = state_summary.get("adapter_parameter_count", 0) > 0
        elif args.method == "grace":
            state_checks.update(
                grace_state_checks(
                    state_summary,
                    position,
                    prior_entry_count=pre_state_summary.get("entry_count"),
                    insertion_action=(edit.get("insert") or {}).get("action"),
                )
            )
        else:
            state_checks["entry_count_exact"] = state_summary.get("entry_count") == position
        report = {
            "schema_version": "m3bench-formal-sequential-step-v1",
            "created_at_utc": utc_now(),
            "status": "PASS" if all(state_checks.values()) else "FAIL",
            "method": args.method,
            "mode": "sequential",
            "position": position,
            "edit_record_id": record.record_id,
            "checks": state_checks,
            "pre_target_nll": pre_nll,
            "post_target_nll": post_nll,
            "edit": edit,
            "post_edit_generation": post_generation,
            "state_summary": state_summary,
            "checkpoint": state,
            "base_integrity": base,
            "runtime_seconds": time.perf_counter() - step_started,
            "peak_gpu_memory": peak,
            "semantic_metrics_computed": False,
        }
        write_frozen_json(steps_dir / f"step_{position:03d}.json", report)
        if report["status"] != "PASS":
            raise RuntimeError(f"sequential step {position} failed")
        if position in args.prefix_values:
            run_prefix(
                run=run,
                output=output,
                editor=editor,
                records=records,
                catalog=catalog,
                prefix=position,
                method=args.method,
                generation_lock_hash=generation_lock_hash,
                base_full_sha256=base_full,
                sequence=locked_sequence,
            )
    end_full = full_base_hash(runtime)
    final_summary = editor.state_summary()
    final_checks = {
        "steps_complete": len(list(steps_dir.glob("step_*.json"))) == args.expected_record_count,
        "prefixes_complete": all(
            (output / "prefixes" / f"prefix_{prefix:03d}" / "PASS").exists()
            for prefix in args.prefix_values
        ),
        "base_full_hash_exact": end_full == base_full,
        "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
        "edit_history_complete": final_summary.get("edit_history") == [record.record_id for record in records],
        "method_state_contract": method_state_contract(
            args.method, final_summary, args.expected_record_count
        ),
    }
    report = {
        "schema_version": "m3bench-formal-sequential-run-report-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(final_checks.values()) else "FAIL",
        "method": args.method,
        "sequence": locked_sequence,
        "checks": final_checks,
        "base_full_sha256_before": base_full,
        "base_full_sha256_after": end_full,
        "final_state_summary": final_summary,
        "resume_audit": resume_audit,
        "preserved_incomplete_artifacts": orphan_audits,
        "runtime_seconds_this_process": time.perf_counter() - started,
        "environment": environment,
        "semantic_metrics_computed": False,
    }
    write_frozen_json(final_report_path, report)
    write_frozen_text(output / ("RUN_PASS" if report["status"] == "PASS" else "RUN_FAIL"), report["status"] + "\n")
    print(json.dumps({"status": report["status"], "method": args.method, "steps": args.expected_record_count, "runtime_seconds_this_process": report["runtime_seconds_this_process"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_sequential_replay(args: argparse.Namespace) -> None:
    assert_authorized_device()
    run, output = Path(args.run_root), Path(args.output_dir)
    replay_report_path = output / "SEQUENTIAL_FRESH_REPLAY_REPORT.json"
    if replay_report_path.exists():
        existing = read_json(replay_report_path)
        if (
            existing.get("status") != "PASS"
            or not all(existing.get("checks", {}).values())
            or existing.get("sequence") != sequence_lock(run, args)
        ):
            raise RuntimeError("existing sequential replay report is not PASS")
        write_frozen_text(output / "REPLAY_PASS", "PASS\n")
        print(json.dumps({"status": "PASS", "method": args.method, "resumed_from_replay_report": True}, indent=2))
        return
    records = load_records(run, args.records_path, args.expected_record_count)
    runtime = load_runtime(run, args.device)
    editor = create_formal_editor(args.method, runtime, output, run=run, sequential=True)
    locked_sequence = sequence_lock(run, args)
    final_position = args.final_prefix
    state_manifest = checkpoint_manifest_path(output, final_position)
    if not state_manifest.exists() or read_json(state_manifest).get("sequence") != locked_sequence:
        raise RuntimeError("final checkpoint lock differs from current run")
    editor.load_editor_state(checkpoint_path(output, args.method, final_position))
    expected = read_json(output / f"steps/step_{final_position:03d}.json")["post_edit_generation"]
    actual = editor.generate(records[-1], use_cache=True)
    summary = editor.state_summary()
    checks = {
        "raw_tokens_exact": actual["generation"]["raw_token_ids"] == expected["generation"]["raw_token_ids"],
        "decoded_text_exact": actual["generation"]["decoded_text"] == expected["generation"]["decoded_text"],
        "sequence_contract_exact": actual["generation"]["sequence_contract"] == expected["generation"]["sequence_contract"],
        "route_exact": route_equal(args.method, expected["route"], actual["route"]),
        "edit_history_complete": summary.get("edit_history") == [record.record_id for record in records],
        "method_state_contract": method_state_contract(
            args.method, summary, args.expected_record_count
        ),
        "base_sentinel_unchanged": editor.base_integrity()["unchanged"],
    }
    report = {
        "schema_version": "m3bench-formal-sequential-fresh-replay-v1",
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "position": final_position,
        "sequence": locked_sequence,
        "checks": checks,
        "state_summary": summary,
        "base_integrity": editor.base_integrity(),
        "comparison_contract": {
            "generation": "exact",
            "route_non_radius": "exact",
            "grace_radius": "IEEE-754 binary32 then exact" if args.method == "grace" else "exact",
            "numeric_tolerance": None,
        },
    }
    write_frozen_json(replay_report_path, report)
    write_frozen_text(output / ("REPLAY_PASS" if report["status"] == "PASS" else "REPLAY_FAIL"), report["status"] + "\n")
    print(json.dumps({"status": report["status"], "method": args.method, "checks": checks}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    def add_sequence_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--records-path", default=DEFAULT_RECORDS_PATH)
        command.add_argument("--expected-record-count", type=int, default=200)
        command.add_argument("--prefixes", default="1,50,100,200")
        command.add_argument("--final-prefix", type=int, default=200)
        command.add_argument("--sequence-label", default=DEFAULT_SEQUENCE_LABEL)

    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    single = sub.add_parser("single-chunk")
    single.add_argument("--run-root", required=True)
    single.add_argument("--output-dir", required=True)
    single.add_argument("--method", choices=METHODS, required=True)
    single.add_argument("--start", type=int, required=True)
    single.add_argument("--end", type=int, required=True)
    single.add_argument("--device", default="cuda:0")
    add_sequence_arguments(single)
    single.set_defaults(func=command_single_chunk)
    events = sub.add_parser("single-events")
    events.add_argument("--run-root", required=True)
    events.add_argument("--output-dir", required=True)
    events.add_argument("--method", choices=METHODS, required=True)
    events.add_argument("--task", choices=("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G"), required=True)
    events.add_argument("--start", type=int, required=True)
    events.add_argument("--end", type=int, required=True)
    events.add_argument("--device", default="cuda:0")
    add_sequence_arguments(events)
    events.set_defaults(func=command_single_events)
    sequential = sub.add_parser("sequential-run")
    sequential.add_argument("--run-root", required=True)
    sequential.add_argument("--output-dir", required=True)
    sequential.add_argument("--method", choices=METHODS, required=True)
    sequential.add_argument("--device", default="cuda:0")
    add_sequence_arguments(sequential)
    sequential.set_defaults(func=command_sequential_run)
    replay = sub.add_parser("sequential-replay")
    replay.add_argument("--run-root", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--method", choices=METHODS, required=True)
    replay.add_argument("--device", default="cuda:0")
    add_sequence_arguments(replay)
    replay.set_defaults(func=command_sequential_replay)
    return result


def normalize_sequence_args(args: argparse.Namespace) -> argparse.Namespace:
    try:
        args.prefix_values = tuple(int(value) for value in args.prefixes.split(","))
    except ValueError as exc:
        raise ValueError("prefixes must be comma-separated integers") from exc
    if (
        args.expected_record_count < 1
        or not args.sequence_label.strip()
        or not args.prefix_values
        or tuple(sorted(set(args.prefix_values))) != args.prefix_values
        or args.prefix_values[0] < 1
        or args.prefix_values[-1] > args.expected_record_count
        or args.final_prefix != args.expected_record_count
        or args.final_prefix not in args.prefix_values
    ):
        raise ValueError("invalid sequence count/prefix/final-prefix contract")
    return args


def main() -> None:
    args = normalize_sequence_args(parser().parse_args())
    run = Path(args.run_root)
    if not (run / "M3BENCH_FORMAL_EDITOR_PREFLIGHT_PASS").exists():
        raise RuntimeError("formal preflight PASS marker is absent")
    args.func(args)


if __name__ == "__main__":
    main()
