#!/usr/bin/env python3
"""Run one private, effect-level trace against the bound editor runtime."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

from m3bench_repro.editors.llava_runtime import (
    LlavaMedEditorRuntime,
    canonical_sha256,
    write_json_atomic,
)
from m3bench_repro.editors.methods import (
    BalanceEditPaperSpecEditor,
    BeloraPaperSpecEditor,
    GracePaperSpecEditor,
    LoraPaperSpecEditor,
    create_editor,
    record_seed,
)
from m3bench_repro.editors.routing import route_dict_equal


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_topology(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("classification", None)
    return result


def target_lock_topology(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("inventory_sha256", None)
    return result


def read_records(path: Path):
    from m3bench_repro.editors.llava_runtime import EditorRecord

    return [
        EditorRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tensor_shape(value: Any) -> list[int] | list[list[int]]:
    if isinstance(value, list):
        return [list(item.shape) for item in value]
    return list(value.shape)


def norm(parameters: list[torch.Tensor]) -> float:
    total = sum(float(value.detach().float().square().sum().cpu()) for value in parameters)
    return math.sqrt(total)


def state_delta(editor: Any, record_id: str) -> float:
    if isinstance(editor, LoraPaperSpecEditor):
        current = dict(editor.peft_model.named_parameters())
        return norm(
            [
                current[name] - initial.to(device=current[name].device, dtype=current[name].dtype)
                for name, initial in editor._initial_adapter_state.items()
            ]
        )
    if isinstance(editor, GracePaperSpecEditor):
        value = editor.wrapper.get_value(editor.requested_to_effective[record_id])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(record_seed(record_id, editor.method))
        initial = torch.rand(value.numel(), generator=generator, dtype=torch.float32)
        return norm([value - initial.to(value.device)])
    if isinstance(editor, BalanceEditPaperSpecEditor):
        edited = editor.wrapper.get_edit(record_id)
        values = [edited.weight - editor.wrapper.base.weight.float()]
        if edited.bias is not None:
            values.append(edited.bias - editor.wrapper.base.bias.float())
        return norm(values)
    if isinstance(editor, BeloraPaperSpecEditor):
        # Every B matrix starts at exactly zero; its post-edit norm is a strict
        # lower bound on total adapter-state movement.
        adapter_name = editor.edit_to_adapter[record_id]
        return norm(
            [wrapper.lora_B[wrapper.logical_to_slot[adapter_name]] for wrapper in editor.wrappers.values()]
        )
    raise TypeError(type(editor))


def editor_parameters(editor: Any) -> list[torch.Tensor]:
    if isinstance(editor, LoraPaperSpecEditor):
        return [p for n, p in editor.peft_model.named_parameters() if "lora_" in n]
    if isinstance(editor, GracePaperSpecEditor):
        return list(editor.wrapper.values.values())
    if isinstance(editor, BalanceEditPaperSpecEditor):
        return list(editor.wrapper.get_edit(editor.edit_history[-1]).parameters())
    if isinstance(editor, BeloraPaperSpecEditor):
        return [
            parameter
            for wrapper in editor.wrappers.values()
            for parameter in (*wrapper.lora_A.values(), *wrapper.lora_B.values())
        ]
    raise TypeError(type(editor))


def active_name(editor: Any) -> str | list[str] | None:
    if isinstance(editor, LoraPaperSpecEditor):
        value = getattr(editor.peft_model, "active_adapter", None)
        return tuple(value) if isinstance(value, (list, tuple)) else value
    if isinstance(editor, GracePaperSpecEditor):
        return editor.wrapper.active_logical_id
    if isinstance(editor, BalanceEditPaperSpecEditor):
        return editor.wrapper.active_logical_id
    if isinstance(editor, BeloraPaperSpecEditor):
        values = {wrapper.active_logical_id for wrapper in editor.wrappers.values()}
        return next(iter(values)) if len(values) == 1 else sorted(str(value) for value in values)
    return None


def trace_module(runtime: LlavaMedEditorRuntime, editor: Any) -> torch.nn.Module:
    if isinstance(editor, (GracePaperSpecEditor, BalanceEditPaperSpecEditor)):
        return editor.wrapper
    if isinstance(editor, BeloraPaperSpecEditor):
        return editor.wrappers[editor.target]
    candidates = [
        (name, module)
        for name, module in runtime.model.named_modules()
        if name.endswith(editor.targets[-1]) and hasattr(module, "forward")
    ]
    if not candidates:
        raise RuntimeError("cannot resolve a LoRA trace module")
    return min(candidates, key=lambda item: item[0].count("."))[1]


def record_forward(events: list[dict[str, Any]], editor: Any):
    def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        value = args[0] if args and isinstance(args[0], torch.Tensor) else None
        events.append(
            {
                "sequence_length": int(value.shape[-2]) if value is not None and value.ndim >= 2 else None,
                "active_adapter": active_name(editor),
            }
        )

    return hook


def generation_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in ("decoded_text", "raw_token_ids", "sequence_contract")
    )


def saved_state_path(method: str, output_dir: Path) -> Path:
    return output_dir / ("editor_state" if method == "lora" else "editor_state.pt")


def adapter_active_before_generation(
    prefill: list[dict[str, Any]], expected_adapter: str
) -> bool:
    return bool(prefill) and any(
        event["active_adapter"] == expected_adapter for event in prefill
    ) and all(event["active_adapter"] in {None, expected_adapter} for event in prefill)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: editor_effect_probe.py CONFIG.json")
    config = read_json(Path(sys.argv[1]))
    source_root = Path(config["source_root"])
    output_dir = Path(config["output_dir"])
    phase = str(config.get("phase", "PREPATCH")).upper()
    if phase not in {"PREPATCH", "POSTPATCH"}:
        raise ValueError(f"unsupported phase: {phase}")
    method = config["method"]
    records = read_records(Path(config["records_path"]))
    wanted_id = config["record_id"]
    record = next(item for item in records if item.record_id == wanted_id)

    runtime = LlavaMedEditorRuntime(
        device=config.get("device", "cuda:0"),
        run_root=Path(config["run_root"]),
        generation_config_path=Path(config["generation_config_path"]),
    )
    runtime.load_frozen_backbone(seed=20260828)
    inventory, target_lock = runtime.resolve_module_inventory(freeze=False)
    if canonical_sha256(inventory_topology(inventory)) != canonical_sha256(
        inventory_topology(read_json(source_root / "runtime/LLAVA_MED_MODULE_INVENTORY.json"))
    ):
        raise RuntimeError("module inventory differs from frozen smoke source")
    if canonical_sha256(target_lock_topology(target_lock)) != canonical_sha256(
        target_lock_topology(read_json(source_root / "runtime/LLAVA_MED_EDIT_TARGET_LOCK.json"))
    ):
        raise RuntimeError("target lock differs from frozen smoke source")

    editor = create_editor(method, runtime)
    module = trace_module(runtime, editor)
    with editor.disabled():
        base_generation = runtime.generate(record, use_cache=True)
    pre = editor.score_target_nll(record)
    batch = runtime.build_edit_batch(record)
    raw = runtime.adapter.prepare_inputs(record.image_path, record.question, record.target)
    from llava.constants import IMAGE_TOKEN_INDEX

    base_before = runtime.base_guard.aggregate_sha256
    edit_events: list[dict[str, Any]] = []
    handle = module.register_forward_pre_hook(record_forward(edit_events, editor))
    try:
        edit = editor.apply_edit(record)
    finally:
        handle.remove()
    parameters = editor_parameters(editor)
    gradient_norm = norm([p.grad for p in parameters if p.grad is not None])
    delta_norm = state_delta(editor, record.record_id)
    post = editor.score_target_nll(record)
    generation_events: list[dict[str, Any]] = []
    handle = module.register_forward_pre_hook(record_forward(generation_events, editor))
    try:
        generated = editor.generate(record, use_cache=True)
    finally:
        handle.remove()
    base_after = editor.base_integrity()
    route = post.get("route") or {}
    processor = runtime.adapter.image_processor
    processor_config = processor.to_dict() if hasattr(processor, "to_dict") else vars(processor)
    prompt_hash = hashlib.sha256(batch.prompt.encode("utf-8")).hexdigest()
    prefill = [event for event in generation_events if (event["sequence_length"] or 0) > 1]
    decode = [event for event in generation_events if event["sequence_length"] == 1]
    reload_report: dict[str, Any] = {}
    if phase == "POSTPATCH":
        output_dir.mkdir(parents=True, exist_ok=False)
        state = editor.save_editor_state(saved_state_path(method, output_dir))
        state_summary = editor.state_summary()
        editor.reset_editor_state()
        reset_generation = runtime.generate(record, use_cache=True)
        editor.load_editor_state(saved_state_path(method, output_dir))
        reloaded_generation = editor.generate(record, use_cache=True)
        reloaded_nll = editor.score_target_nll(record)
        with editor.disabled():
            miss_generation = runtime.generate(record, use_cache=True)
        route_mode = "float32" if method in {"grace", "balancedit", "belora"} else "exact"
        reload_report = {
            "state": state,
            "state_summary": state_summary,
            "save_reload_generation_parity": generation_equal(
                generated["generation"], reloaded_generation["generation"]
            ),
            "save_reload_route_parity": route_dict_equal(
                generated.get("route"), reloaded_generation.get("route"), radius_mode=route_mode
            ),
            "save_reload_nll_parity": post["nll"] == reloaded_nll["nll"],
            "reset_base_generation_parity": generation_equal(base_generation, reset_generation),
            "miss_base_token_parity": generation_equal(base_generation, miss_generation),
        }
    report = {
        "schema_version": "m3bench-one-edit-effect-trace-v2",
        "phase": phase,
        "record_id": record.record_id,
        "method": method,
        "actual_source_files": {
            "runtime": inspect.getsourcefile(LlavaMedEditorRuntime),
            "editor": inspect.getsourcefile(editor.__class__),
            "editor_class": editor.__class__.__name__,
        },
        "image_path": str(record.image_path),
        "image_tensor_shape": tensor_shape(raw["images"]),
        "text_token_length": int(batch.raw_input_ids.shape[1]),
        "multimodal_expanded_sequence_length": int(batch.inputs_embeds.shape[1]),
        "image_token_count": int((batch.raw_input_ids == IMAGE_TOKEN_INDEX).sum().item()),
        "conversation_template_hash": prompt_hash,
        "processor_config_hash": canonical_sha256(processor_config),
        "generation_config_hash": canonical_sha256(runtime.generation_config),
        "labels_shape": list(batch.labels.shape),
        "target_token_count": int((batch.labels != -100).sum().item()),
        "prompt_image_masked_token_count": int((batch.labels == -100).sum().item()),
        "target_start_expanded": batch.target_start_expanded,
        "key_token_index_expanded": batch.key_token_index,
        "pre_target_nll": pre["nll"],
        "post_target_nll": post["nll"],
        "target_logprob_delta": pre["nll"] - post["nll"],
        "trainable_parameter_count": edit["trainable_parameter_count"],
        "gradient_norm": gradient_norm,
        "state_delta_norm": delta_norm,
        "route_key_norm": edit.get("key_norm") or (
            float(torch.linalg.vector_norm(editor.router.keys[-1]).cpu()) if hasattr(editor, "router") else None
        ),
        "nearest_distance": route.get("nearest_distance"),
        "radius": route.get("radius"),
        "self_route_hit": None if method == "lora" else bool(
            route.get("activated") and route.get("logical_edit_id") == record.record_id
        ),
        "selected_logical_edit_id": route.get("logical_edit_id"),
        "active_adapter_during_prefill": [event["active_adapter"] for event in prefill],
        "active_adapter_during_decode": [event["active_adapter"] for event in decode],
        "hook_call_count_in_edit_forward": len(edit_events),
        "hook_call_count_in_generation_prefill": len(prefill),
        "hook_call_count_in_decode": len(decode),
        "generation_forward_trace": generation_events,
        "post_raw_token_ids": generated["generation"]["raw_token_ids"],
        "post_raw_text": generated["generation"]["decoded_text"],
        "base_raw_token_ids": base_generation["raw_token_ids"],
        "base_raw_text": base_generation["decoded_text"],
        "post_generation_nonempty": bool(generated["generation"]["raw_token_ids"]),
        "post_generation_equals_base": generated["generation"]["raw_token_ids"] == base_generation["raw_token_ids"],
        "base_weights_hash_before": base_before,
        "base_weights_hash_after": base_after["after_sha256"],
        "base_weights_unchanged": base_after["unchanged"],
        "finite_gradient": math.isfinite(gradient_norm),
        "finite_state_delta": math.isfinite(delta_norm),
        "reload": reload_report,
        "edit": edit,
    }
    expected_adapter = (
        "default"
        if method == "lora"
        else editor.edit_to_adapter[record.record_id]
        if method == "belora"
        else record.record_id
    )
    route_context_stable = (
        bool(prefill)
        and bool(decode)
        and any(event["active_adapter"] == expected_adapter for event in prefill)
        and all(event["active_adapter"] in {None, expected_adapter} for event in prefill)
        and all(event["active_adapter"] == expected_adapter for event in decode)
    )
    checks = {
        "image_present_in_edit_forward": bool(batch.image_tensor_shape),
        "multimodal_expansion_verified": batch.mask_report()["multimodal_expansion_verified"],
        "target_token_count_positive": batch.mask_report()["target_token_count"] > 0,
        "finite_positive_gradient_or_value_update": math.isfinite(gradient_norm)
        and gradient_norm > 0
        and math.isfinite(delta_norm)
        and delta_norm > 0,
        "post_target_nll_lower": post["nll"] < pre["nll"],
        "target_logprob_delta_positive": pre["nll"] - post["nll"] > 0,
        "base_weights_unchanged": base_after["unchanged"],
        "post_generation_nonempty": bool(generated["generation"]["raw_token_ids"]),
        "save_reload_parity": all(
            reload_report.get(key, False)
            for key in (
                "save_reload_generation_parity",
                "save_reload_route_parity",
                "save_reload_nll_parity",
            )
        ),
        "reset_or_disable_base_parity": reload_report.get("reset_base_generation_parity", False)
        and reload_report.get("miss_base_token_parity", False),
        "route_context_stable_across_decode": route_context_stable,
    }
    if method != "lora":
        checks.update(
            {
                "self_route_hit": report["self_route_hit"] is True,
                "selected_id_is_current": report["selected_logical_edit_id"] == record.record_id,
                "miss_base_token_parity": reload_report.get("miss_base_token_parity", False),
            }
        )
    if method in {"lora", "belora"}:
        checks["active_adapter_before_model_forward"] = adapter_active_before_generation(
            prefill, expected_adapter
        )
    report["checks"] = checks
    report["status"] = "PASS" if phase == "POSTPATCH" and all(checks.values()) else (
        "TRACE_COMPLETE" if phase == "PREPATCH" else "FAIL"
    )
    stem = "prepatch_one_edit_trace" if phase == "PREPATCH" else "ONE_EDIT_EFFECT_GATE"
    if phase == "PREPATCH":
        output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / f"{stem}_{method}.json"
    write_json_atomic(json_path, report, read_only=True)
    md = [
        f"# {phase.title()} one-edit effect trace: {method}",
        "",
        f"- Record: `{record.record_id}`",
        f"- NLL: `{pre['nll']:.6f}` -> `{post['nll']:.6f}`",
        f"- Target log-prob delta: `{report['target_logprob_delta']:.6f}`",
        f"- Gradient norm: `{gradient_norm:.6f}`",
        f"- State delta norm: `{delta_norm:.6f}`",
        f"- Self-route hit: `{report['self_route_hit']}`",
        f"- Generation equals base: `{report['post_generation_equals_base']}`",
        f"- Post generation nonempty: `{report['post_generation_nonempty']}`",
        f"- Base weights unchanged: `{report['base_weights_unchanged']}`",
        f"- Status: `{report['status']}`",
        "",
        "The adjacent JSON contains the private raw trace required for root-cause analysis.",
    ]
    md_path = output_dir / f"{stem}_{method}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    os.chmod(md_path, 0o444)
    print(json.dumps({key: report[key] for key in (
        "method", "record_id", "pre_target_nll", "post_target_nll", "gradient_norm",
        "state_delta_norm", "self_route_hit", "post_generation_equals_base",
        "post_generation_nonempty", "base_weights_unchanged", "status", "checks"
    )}, indent=2))
    if phase == "POSTPATCH" and report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
