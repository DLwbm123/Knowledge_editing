#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_cure_20edit_modelknown import _run_pytest, _write_env_report  # noqa: E402
from scripts.engram.run_cure_20edit_rescue_nonseq import (  # noqa: E402
    GATE_POLICY,
    _build_rescue_entries,
    _load_or_eval_baselines,
    _materialize_image_root,
    _match_bank_records,
    _row_projection_fields,
    _yaml_value,
)
from scripts.engram.run_cure_mededit_5edit import (  # noqa: E402
    EvalMixedDeltaPatch,
    _cache_shapes,
    _clone_layer_to_cache,
    _collect_reference_crisp_cache,
    _merge_layer_to_cache,
    _projection_caches_for_thresholds_from_kfac,
)
from scripts.engram.run_localized_replacement_5edit import (  # noqa: E402
    EXPECTED_MODULES,
    EvalLoraPatch,
    _configure_hparams,
    _evaluate_current,
    _finite,
    _json_dump,
    _make_eval_row,
    _max_snapshot_diff,
    _mean,
    _module_map,
    _project_factors,
    _restore_modules,
    _safe_div,
    _snapshot_modules,
    _train_tiny_lora,
    _write_csv,
    _write_git_outputs,
)
from scripts.engram.run_token_module_ablation_5edit import _resolve_image  # noqa: E402


METHOD_C = "C_engram_projected_tiny_lora"
METHOD_E = "E_rescued_cure_dual_projected_tiny_lora"

RESCUED_CURE = {
    "config_id": "E_beta0.5_gamma0.3_lambda0.25_clamptrue",
    "method": METHOD_E,
    "beta": 0.5,
    "crisp_energy_threshold": 0.3,
    "cure_mix_lambda": 0.25,
    "cure_norm_clamp": True,
    "cure_norm_clamp_ratio": 1.0,
    "gate_policy": GATE_POLICY,
    "cache_update_policy": "streaming_average",
}


def _run_capture(command: List[str], cwd: Path = PROJECT_ROOT) -> str:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _write_stop_report(out_dir: Path, reason: str, payload: Dict[str, Any]) -> None:
    lines = [
        "# Final CURE 20-Edit Rescue Sequential Report",
        "",
        "- Status: `stopped`",
        f"- Stop reason: `{reason}`",
        "- No extra CURE sweep was launched.",
        "- Direct ENGRAM erase and full unrescued CURE were not rerun.",
        "- No metric was fabricated.",
        "",
        "## Details",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True)[:16000],
        "```",
    ]
    (out_dir / "FINAL_CURE_20EDIT_RESCUE_SEQUENTIAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tests(out_dir: Path, *, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    if not run_tests:
        payload = {"status": "skipped", "reason": "--skip-tests"}
        _json_dump(test_dir / "test_status.json", payload)
        return payload
    engram_tests = sorted(str(path) for path in PROJECT_ROOT.glob("tests/test_engram_*.py"))
    runs = [
        _run_pytest(test_dir / "test_cure_crisp_projection.log", ["tests/test_cure_crisp_projection.py", "-q"]),
        _run_pytest(test_dir / "test_cure_kfac_collector_tiny_mllm.log", ["tests/test_cure_kfac_collector_tiny_mllm.py", "-q"]),
        _run_pytest(test_dir / "test_engram_all.log", [*engram_tests, "-q"]),
    ]
    payload = {"status": "pass" if all(item["returncode"] == 0 for item in runs) else "fail", "runs": runs}
    _json_dump(test_dir / "test_status.json", payload)
    return payload


def _write_preflight(
    out_dir: Path,
    *,
    hparams_path: Path,
    source_output_dir: Path,
    rescue_nonseq_dir: Path,
    data_path: Path,
    image_root: Path,
    projector_bank_dir: Path,
    test_status: Dict[str, Any],
) -> Dict[str, Any]:
    model_path = _yaml_value(hparams_path, "name") or _yaml_value(hparams_path, "model_path") or _yaml_value(hparams_path, "model_name")
    vision_path = (
        _yaml_value(hparams_path, "llava_med_vision_tower")
        or _yaml_value(hparams_path, "clip_vision_path")
        or _yaml_value(hparams_path, "vision_tower")
    )
    import_checks: Dict[str, Any] = {}
    for module_name in ["torch", "transformers", "peft", "PIL"]:
        try:
            __import__(module_name)
            import_checks[f"{module_name}_import"] = True
        except Exception as exc:
            import_checks[f"{module_name}_import"] = f"{type(exc).__name__}: {exc}"
    checks = {
        "python_path_present": bool(sys.executable),
        "cuda_available": bool(torch.cuda.is_available()),
        "path_exists_hparams": hparams_path.exists(),
        "path_exists_model": bool(model_path and Path(model_path).exists()),
        "path_exists_vision_tower": bool(vision_path and Path(vision_path).exists()),
        "source_output_dir_exists": source_output_dir.exists(),
        "source_data_file_exists": data_path.exists(),
        "image_root_ready": image_root.exists(),
        "projector_bank_dir_exists": projector_bank_dir.exists(),
        "projector_bank_index_exists": (projector_bank_dir / "index.json").exists(),
        "rescue_nonseq_report_exists": (rescue_nonseq_dir / "FINAL_CURE_20EDIT_RESCUE_NONSEQ_REPORT.md").exists(),
        "rescue_nonseq_aggregates_exists": (rescue_nonseq_dir / "rescue_nonseq_aggregates.csv").exists(),
        "rescue_projection_diagnostics_exists": (rescue_nonseq_dir / "rescue_projection_diagnostics.json").exists(),
        "output_dir_writable": out_dir.exists() and out_dir.is_dir(),
        "tests_pass": test_status.get("status") == "pass",
        **import_checks,
    }
    payload = {
        "status": "pass" if all(value is True for value in checks.values()) else "fail",
        "checks": checks,
        "paths": {
            "hparams": str(hparams_path),
            "model_path": model_path,
            "vision_tower": vision_path,
            "source_output_dir": str(source_output_dir),
            "reused_data_file": str(data_path),
            "image_root": str(image_root),
            "projector_bank_dir": str(projector_bank_dir),
            "rescue_nonseq_dir": str(rescue_nonseq_dir),
            "output_dir": str(out_dir),
        },
        "rescued_cure_config": RESCUED_CURE,
    }
    lines = [
        "# CURE 20-Edit Rescue Sequential Preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Python: `{sys.executable}`",
        "- Main gate: sequential validation with `--skip-generation`.",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Paths", ""])
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "PREFLIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "preflight_status.json", payload)
    return payload


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Expected list records at {path}, got {type(records)}")
    return records


def _reuse_selected_data(
    *,
    source_output_dir: Path,
    rescue_nonseq_dir: Path,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Path, Path, Dict[str, Any]]:
    source_data = source_output_dir / "synthetic_root" / "data" / "medmkeb" / "raw" / "engram_replacement_20edit_modelknown.json"
    source_images = source_output_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    records = _load_records(source_data)
    raw_dir = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw"
    image_root = out_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_path = raw_dir / "engram_replacement_20edit_rescue_sequential.json"
    data_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    image_materialization = _materialize_image_root(source_images, image_root)

    source_summary_path = source_output_dir / "replacement_data_summary.json"
    source_filter_path = source_output_dir / "data_filter_report.json"
    source_record_preflight_path = source_output_dir / "record_id_preflight.json"
    rescue_report = rescue_nonseq_dir / "FINAL_CURE_20EDIT_RESCUE_NONSEQ_REPORT.md"
    rescue_aggregates = rescue_nonseq_dir / "rescue_nonseq_aggregates.csv"
    rescue_projection = rescue_nonseq_dir / "rescue_projection_diagnostics.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    source_filter = json.loads(source_filter_path.read_text(encoding="utf-8")) if source_filter_path.exists() else {}
    source_record_preflight = json.loads(source_record_preflight_path.read_text(encoding="utf-8")) if source_record_preflight_path.exists() else {}
    ids = [str(record.get("id") or record.get("record_id")) for record in records]
    summary_ids = [str(item) for item in source_summary.get("record_ids", [])]
    filter_ids = [str(item) for item in source_filter.get("selected_record_ids", [])]
    image_rows: List[Dict[str, Any]] = []
    for record in records:
        row = {"record_id": str(record.get("id")), "paths_resolve": True, "paths": {}}
        for key in ["image", "image_rephrase", "m_loc"]:
            try:
                resolved = _resolve_image(image_root, str(record.get(key)))
                row["paths"][key] = resolved
                row["paths_resolve"] = bool(row["paths_resolve"] and Path(resolved).exists())
            except Exception as exc:
                row["paths"][key] = f"{type(exc).__name__}: {exc}"
                row["paths_resolve"] = False
        image_rows.append(row)
    report = {
        "status": "pass"
        if len(records) == 20
        and ids == summary_ids
        and (not filter_ids or ids == filter_ids)
        and all(row["paths_resolve"] for row in image_rows)
        and source_record_preflight.get("record_id_match_rate") == 1.0
        and rescue_report.exists()
        and rescue_aggregates.exists()
        and rescue_projection.exists()
        else "fail",
        "source_output_dir": str(source_output_dir),
        "source_data_file": str(source_data),
        "source_data_filter_report": str(source_filter_path),
        "source_replacement_summary": str(source_summary_path),
        "source_record_id_preflight": str(source_record_preflight_path),
        "rescue_nonseq_report": str(rescue_report),
        "rescue_nonseq_aggregates": str(rescue_aggregates),
        "rescue_projection_diagnostics": str(rescue_projection),
        "reused_data_file": str(data_path),
        "image_root": str(image_root),
        "image_materialization": image_materialization,
        "record_count": len(records),
        "selected_record_ids": ids,
        "matches_source_replacement_summary_ids": ids == summary_ids,
        "matches_source_data_filter_selected_ids": (ids == filter_ids) if filter_ids else None,
        "record_id_match_rate": 1.0 if len(records) == 20 and ids == summary_ids else 0.0,
        "source_record_id_match_rate": source_record_preflight.get("record_id_match_rate"),
        "positional_matching_used": False,
        "private_or_patient_data_used": False,
        "original_data_modified": False,
        "image_rows": image_rows,
    }
    _json_dump(out_dir / "data_reuse_report.json", report)
    return records, data_path, image_root, report


def _current_weight_norms(model: nn.Module, module_names: List[str]) -> Dict[str, float]:
    modules = _module_map(model)
    return {
        name: float(modules[name].weight.detach().cpu().float().norm().item())
        for name in module_names
    }


def _rollback_rows(
    model: nn.Module,
    snapshots: Dict[str, Dict[str, torch.Tensor | None]],
    *,
    method: str,
    param_norm_after_step20: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    modules = _module_map(model)
    rows: List[Dict[str, Any]] = []
    max_diff = 0.0
    for name, tensors in snapshots.items():
        module = modules[name]
        before_weight = tensors["weight"].float()
        after_weight = module.weight.detach().cpu().float()
        diff = float((after_weight - before_weight).abs().max().item())
        max_diff = max(max_diff, diff)
        rows.append(
            {
                "method": method,
                "module_name": name,
                "param_norm_before": float(before_weight.norm().item()),
                "param_norm_after_step20": (param_norm_after_step20 or {}).get(name),
                "param_norm_after_final_rollback": float(after_weight.norm().item()),
                "max_abs_diff_before_vs_after_final_rollback": diff,
            }
        )
    return rows, max_diff


def _make_step_rows(
    *,
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    snapshots: Dict[str, Any],
    method: str,
    step: int,
    applied_record_ids: List[str],
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    record_id_match_rate: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        after = (
            baselines[str(record["id"])]
            if step == 0
            else _evaluate_current(
                model,
                record,
                image_root,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
            )
        )
        base_row = _make_eval_row(
            method=method,
            record=record,
            case_index=idx,
            before=baselines[str(record["id"])],
            after=after,
            rollback_diff=0.0,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            record_id_match_rate=record_id_match_rate,
            beta=beta,
            extra={
                "selected_modules": EXPECTED_MODULES,
                "config_id": "C_beta0.5" if method == METHOD_C and beta != 0.0 else ("C_beta0.0" if method == METHOD_C else RESCUED_CURE["config_id"]),
            },
        )
        new_decrease = base_row.get("new_answer_nll_decrease")
        row = {
            "method": method,
            "step": int(step),
            "applied_record_ids": list(applied_record_ids),
            "record_id": str(record["id"]),
            "case_index": int(idx),
            "is_edited_so_far": bool(idx < step),
            "is_current_edit": bool(step > 0 and idx == step - 1),
            "is_future_edit": bool(idx >= step),
            "old_answer_nll": base_row.get("old_answer_nll_after"),
            "old_answer_nll_delta_vs_step0": base_row.get("old_answer_nll_increase"),
            "new_answer_nll": base_row.get("new_answer_nll_after"),
            "new_answer_nll_decrease_vs_step0": new_decrease,
            "reference_nll": base_row.get("reference_nll_after"),
            "reference_delta_abs_vs_step0": base_row.get("reference_delta_abs"),
            "previous_edit_retention": new_decrease if idx < max(step - 1, 0) else None,
            "future_record_drift": abs(float(new_decrease)) if idx >= step and new_decrease is not None else None,
            "locality_damage": base_row.get("locality_damage"),
            "rollback_pass": None,
            "rollback_max_abs_diff": None,
            "record_id_match_rate": float(record_id_match_rate),
            "nan_inf_detected": base_row.get("nan_inf_detected"),
            "beta": float(beta),
            "old_answer": base_row.get("old_answer"),
            "new_answer": base_row.get("new_answer"),
        }
        if method == METHOD_E:
            row.update(
                {
                    "config_id": RESCUED_CURE["config_id"],
                    "crisp_energy_threshold": RESCUED_CURE["crisp_energy_threshold"],
                    "cure_mix_lambda": RESCUED_CURE["cure_mix_lambda"],
                    "cure_norm_clamp": RESCUED_CURE["cure_norm_clamp"],
                    "cure_norm_clamp_ratio": RESCUED_CURE["cure_norm_clamp_ratio"],
                    "gate_policy": RESCUED_CURE["gate_policy"],
                    "cache_update_policy": RESCUED_CURE["cache_update_policy"],
                }
            )
        row["nan_inf_detected"] = bool(row["nan_inf_detected"] or not _finite(row))
        rows.append(row)
    return rows


def _aggregate_step(rows: List[Dict[str, Any]], method: str, step: int) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method and int(row.get("step") or 0) == int(step)]
    edited_rows = [row for row in metric_rows if row.get("is_edited_so_far")]
    previous_rows = [row for row in metric_rows if row.get("previous_edit_retention") is not None]
    future_rows = [row for row in metric_rows if row.get("is_future_edit")]
    new_values = [float(row["new_answer_nll_decrease_vs_step0"]) for row in edited_rows if row.get("new_answer_nll_decrease_vs_step0") is not None]
    ref_values = [float(row["reference_delta_abs_vs_step0"]) for row in metric_rows if row.get("reference_delta_abs_vs_step0") is not None]
    previous_values = [float(row["previous_edit_retention"]) for row in previous_rows if row.get("previous_edit_retention") is not None]
    future_values = [float(row["future_record_drift"]) for row in future_rows if row.get("future_record_drift") is not None]
    row = {
        "method": method,
        "step": int(step),
        "record_count": len(metric_rows),
        "edited_record_count": len(edited_rows),
        "mean_new_answer_nll_decrease": _mean(new_values),
        "mean_reference_delta_abs_all_records": _mean(ref_values),
        "previous_edit_retention": _mean(previous_values),
        "mean_future_record_drift": _mean(future_values),
        "positive_new_answer_edits": sum(1 for value in new_values if value > 0.0),
        "locality_damage_records": sum(1 for row in metric_rows if row.get("locality_damage")),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows if row.get("rollback_pass") is not None]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
    }
    if method == METHOD_E:
        row.update(
            {
                "config_id": RESCUED_CURE["config_id"],
                "beta": RESCUED_CURE["beta"],
                "crisp_energy_threshold": RESCUED_CURE["crisp_energy_threshold"],
                "cure_mix_lambda": RESCUED_CURE["cure_mix_lambda"],
                "cure_norm_clamp": RESCUED_CURE["cure_norm_clamp"],
                "cure_norm_clamp_ratio": RESCUED_CURE["cure_norm_clamp_ratio"],
                "gate_policy": RESCUED_CURE["gate_policy"],
                "cache_update_policy": RESCUED_CURE["cache_update_policy"],
            }
        )
    return row


def _trace_projection_fields(cure_summary: Dict[str, Any]) -> Dict[str, Any]:
    fields = _row_projection_fields(cure_summary)
    return {
        "delta_candidate_norm": fields.get("delta_candidate_norm"),
        "delta_engram_norm": fields.get("delta_engram_norm"),
        "delta_crisp_norm": fields.get("delta_crisp_norm"),
        "delta_cure_preclamp_norm": fields.get("delta_cure_preclamp_norm"),
        "delta_cure_postclamp_norm": fields.get("delta_cure_postclamp_norm"),
        "projection_norm_ratio_preclamp": fields.get("projection_norm_ratio_preclamp"),
        "projection_norm_ratio_postclamp": fields.get("projection_norm_ratio_postclamp"),
        "clamp_applied": fields.get("clamp_applied"),
        "clamp_applied_module_count": fields.get("clamp_applied_module_count"),
        "mask_keep_ratio": fields.get("mask_keep_ratio"),
        "mask_keep_active": fields.get("mask_keep_active"),
        "mask_keep_total": fields.get("mask_keep_total"),
        "skipped_modules": fields.get("skipped_modules"),
        "skip_reasons": fields.get("skip_reasons"),
        "fallback_modules": fields.get("fallback_modules"),
        "fallback_module_count": fields.get("fallback_module_count"),
    }


def _run_one_sequential_method(
    *,
    model: nn.Module,
    method: str,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    record_id_match_rate: float,
    lambda_mix: float = 0.25,
    clamp: bool = True,
    clamp_ratio: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    active_patches: List[Any] = []
    cache_trace: List[Dict[str, Any]] = []
    applied_ids: List[str] = []
    accumulated_layer_to_cache: Dict[str, Dict[str, Any]] = {}
    accumulated_num_samples = 0
    threshold = float(RESCUED_CURE["crisp_energy_threshold"])

    if method == METHOD_E:
        initial_cache = _collect_reference_crisp_cache(model, records, image_root, module_names, hparams)
        accumulated_layer_to_cache = _clone_layer_to_cache(initial_cache.get("layer_to_cache", {}))
        accumulated_num_samples = int(initial_cache.get("num_samples") or 0)
        cache_trace.append(
            {
                "step": 0,
                "method": method,
                "record_id": None,
                "cache_update_policy": RESCUED_CURE["cache_update_policy"],
                "crisp_energy_threshold": threshold,
                "cure_mix_lambda": float(lambda_mix),
                "cure_norm_clamp": bool(clamp),
                "cure_norm_clamp_ratio": float(clamp_ratio),
                "gate_policy": RESCUED_CURE["gate_policy"],
                "accumulated_num_samples": accumulated_num_samples,
                "modules_with_cache": sorted(accumulated_layer_to_cache),
                "cache_shapes": _cache_shapes(accumulated_layer_to_cache),
                "cache_update_success": initial_cache.get("status") == "complete",
                "skipped_modules": [],
                "skip_reasons": {},
            }
        )

    try:
        rows.extend(
            _make_step_rows(
                model=model,
                records=records,
                image_root=image_root,
                baselines=baselines,
                snapshots=snapshots,
                method=method,
                step=0,
                applied_record_ids=[],
                beta=beta,
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
                record_id_match_rate=record_id_match_rate,
            )
        )
        for step, (record, edit_id) in enumerate(zip(records, edit_ids), start=1):
            factors, train_summary = _train_tiny_lora(
                model,
                record,
                image_root,
                module_names,
                rank=int(hparams.lora_rank),
                steps=int(hparams.lora_steps),
                lr=float(hparams.lora_lr),
                scale=scale,
                lambda_ref=float(hparams.replacement_lambda_ref),
            )
            engram_factors, engram_summary = _project_factors(factors, bank.load_edit(edit_id))
            cure_summary: Optional[Dict[str, Any]] = None
            projection_caches: Dict[str, Dict[str, Any]] = {}
            if method == METHOD_C:
                patch = EvalLoraPatch(model, engram_factors, beta=beta)
            elif method == METHOD_E:
                projection_caches = _projection_caches_for_thresholds_from_kfac(
                    accumulated_layer_to_cache,
                    hparams,
                    [threshold],
                ).get(threshold, {})
                entries, cure_summary = _build_rescue_entries(
                    candidate_factors=factors,
                    engram_factors=engram_factors,
                    projection_caches=projection_caches,
                    lambda_mix=float(lambda_mix),
                    clamp=bool(clamp),
                    clamp_ratio=float(clamp_ratio),
                    gate_policy=RESCUED_CURE["gate_policy"],
                )
                patch = EvalMixedDeltaPatch(model, entries, beta=beta)
            else:
                raise ValueError(f"Unsupported sequential method: {method}")
            patch.install()
            active_patches.append(patch)
            applied_ids.append(str(record["id"]))
            step_rows = _make_step_rows(
                model=model,
                records=records,
                image_root=image_root,
                baselines=baselines,
                snapshots=snapshots,
                method=method,
                step=step,
                applied_record_ids=applied_ids,
                beta=beta,
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
                record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
            )
            if method == METHOD_E and cure_summary is not None:
                projection_fields = _trace_projection_fields(cure_summary)
                for row in step_rows:
                    row.update(projection_fields)
            rows.extend(step_rows)
            if method == METHOD_E and cure_summary is not None:
                new_cache = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
                accumulated_layer_to_cache, accumulated_num_samples = _merge_layer_to_cache(
                    accumulated_layer_to_cache,
                    new_cache.get("layer_to_cache", {}),
                    accumulated_num_samples=accumulated_num_samples,
                    new_num_samples=int(new_cache.get("num_samples") or 0),
                )
                trace_row = {
                    "step": int(step),
                    "method": method,
                    "record_id": str(record["id"]),
                    "cache_update_policy": RESCUED_CURE["cache_update_policy"],
                    "crisp_energy_threshold": threshold,
                    "cure_mix_lambda": float(lambda_mix),
                    "cure_norm_clamp": bool(clamp),
                    "cure_norm_clamp_ratio": float(clamp_ratio),
                    "gate_policy": RESCUED_CURE["gate_policy"],
                    "accumulated_num_samples": accumulated_num_samples,
                    "modules_with_cache": sorted(accumulated_layer_to_cache),
                    "cache_shapes": _cache_shapes(accumulated_layer_to_cache),
                    "cache_update_success": new_cache.get("status") == "complete",
                    "engram_projection": engram_summary,
                    "crisp_projection": cure_summary,
                    "lora_train": train_summary,
                    "module_projection_rows": cure_summary.get("modules", []),
                    **_trace_projection_fields(cure_summary),
                }
                cache_trace.append(trace_row)
    finally:
        after_step20_norms = _current_weight_norms(model, module_names)
        for patch in reversed(active_patches):
            patch.remove()
        rollback_rows, rollback_diff = _rollback_rows(
            model,
            snapshots,
            method=method,
            param_norm_after_step20=after_step20_norms,
        )
        _restore_modules(model, snapshots)
        for row in rows:
            if int(row.get("step") or 0) == len(records):
                row["rollback_pass"] = bool(rollback_diff <= float(rollback_tolerance))
                row["rollback_max_abs_diff"] = rollback_diff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rollback_payload = {
        "method": method,
        "rollback_max_abs_diff": rollback_diff,
        "rollback_pass": bool(rollback_diff <= float(rollback_tolerance)),
        "modules": rollback_rows,
    }
    return rows, rollback_payload, cache_trace


def _lambda0_equivalence_check(
    *,
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    tolerance: float,
) -> Dict[str, Any]:
    bank = EngramBank(projector_bank_dir)
    edit_ids, _ = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    record = records[0]
    edit_id = edit_ids[0]
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    try:
        factors, train_summary = _train_tiny_lora(
            model,
            record,
            image_root,
            module_names,
            rank=int(hparams.lora_rank),
            steps=int(hparams.lora_steps),
            lr=float(hparams.lora_lr),
            scale=scale,
            lambda_ref=float(hparams.replacement_lambda_ref),
        )
        engram_factors, engram_summary = _project_factors(factors, bank.load_edit(edit_id))
        cache = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
        projection_caches = _projection_caches_for_thresholds_from_kfac(
            cache.get("layer_to_cache", {}),
            hparams,
            [float(RESCUED_CURE["crisp_energy_threshold"])],
        ).get(float(RESCUED_CURE["crisp_energy_threshold"]), {})
        entries, cure_summary = _build_rescue_entries(
            candidate_factors=factors,
            engram_factors=engram_factors,
            projection_caches=projection_caches,
            lambda_mix=0.0,
            clamp=False,
            clamp_ratio=1.0,
            gate_policy=RESCUED_CURE["gate_policy"],
        )
        for method, patch in [
            (METHOD_C, EvalLoraPatch(model, engram_factors, beta=float(RESCUED_CURE["beta"]))),
            (METHOD_E, EvalMixedDeltaPatch(model, entries, beta=float(RESCUED_CURE["beta"]))),
        ]:
            patch.install()
            try:
                after = _evaluate_current(
                    model,
                    record,
                    image_root,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                )
            finally:
                patch.remove()
            rows.append(
                _make_eval_row(
                    method=method,
                    record=record,
                    case_index=0,
                    before=baselines[str(record["id"])],
                    after=after,
                    rollback_diff=_max_snapshot_diff(model, snapshots),
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    record_id_match_rate=1.0,
                    beta=float(RESCUED_CURE["beta"]),
                    extra={
                        "config_id": f"lambda0_{method}",
                        "engram_projection": engram_summary,
                        "crisp_projection": cure_summary if method == METHOD_E else None,
                        "lora_train": train_summary,
                    },
                )
            )
    finally:
        _restore_modules(model, snapshots)
    fields = [
        "old_answer_nll_after",
        "old_answer_nll_increase",
        "new_answer_nll_after",
        "new_answer_nll_decrease",
        "reference_nll_after",
        "reference_delta_abs",
    ]
    c_row, e_row = rows[0], rows[1]
    diffs = {
        field: abs(float(c_row[field]) - float(e_row[field]))
        for field in fields
        if c_row.get(field) is not None and e_row.get(field) is not None
    }
    max_abs_diff = max(diffs.values()) if diffs else None
    payload = {
        "status": "pass" if max_abs_diff is not None and max_abs_diff <= float(tolerance) else "fail",
        "lambda0_max_abs_diff": max_abs_diff,
        "tolerance": float(tolerance),
        "record_id": str(record.get("id")),
        "field_diffs": diffs,
        "c_row": c_row,
        "lambda0_row": e_row,
    }
    _json_dump(out_dir / "lambda0_equivalence_check.json", payload)
    return payload


def _summarize_final(
    *,
    rows: List[Dict[str, Any]],
    rollback_checks: List[Dict[str, Any]],
    trace: List[Dict[str, Any]],
    tolerance: float,
) -> Dict[str, Any]:
    methods = [METHOD_C, METHOD_E]
    max_step = max(int(row.get("step") or 0) for row in rows) if rows else 0
    summary_rows = [_aggregate_step(rows, method, step) for method in methods for step in range(max_step + 1)]
    final_rows = [row for row in summary_rows if int(row.get("step") or 0) == int(max_step)]
    by_method = {row["method"]: row for row in final_rows}
    c_final = by_method.get(METHOD_C, {})
    e_final = by_method.get(METHOD_E, {})
    new_ratio = _safe_div(e_final.get("mean_new_answer_nll_decrease"), c_final.get("mean_new_answer_nll_decrease"))
    ref_ratio = _safe_div(e_final.get("mean_reference_delta_abs_all_records"), c_final.get("mean_reference_delta_abs_all_records"))
    ret_ratio = _safe_div(e_final.get("previous_edit_retention"), c_final.get("previous_edit_retention"))
    rollback_payload = {
        "status": "pass" if all(item.get("rollback_pass") for item in rollback_checks) else "fail",
        "rollback_tolerance": float(tolerance),
        "methods": rollback_checks,
        "max_abs_diff": max((float(item.get("rollback_max_abs_diff") or 0.0) for item in rollback_checks), default=None),
    }
    basic_checks = {
        "positive_new_answer_edits_gte_16_of_20": int(e_final.get("positive_new_answer_edits") or 0) >= 16,
        "mean_new_answer_nll_decrease_positive": e_final.get("mean_new_answer_nll_decrease") is not None
        and float(e_final["mean_new_answer_nll_decrease"]) > 0.0,
        "mean_reference_delta_abs_all_records_lt_mean_new": (
            e_final.get("mean_reference_delta_abs_all_records") is not None
            and e_final.get("mean_new_answer_nll_decrease") is not None
            and float(e_final["mean_reference_delta_abs_all_records"]) < float(e_final["mean_new_answer_nll_decrease"])
        ),
        "locality_damage_records_lte_2": int(e_final.get("locality_damage_records") or 0) <= 2,
        "rollback_pass": rollback_payload["status"] == "pass",
        "record_id_match_rate_is_1": float(e_final.get("record_id_match_rate") or 0.0) == 1.0,
        "nan_inf_count_is_0": int(e_final.get("nan_inf_count") or 0) == 0,
    }
    relative_checks = {
        "new_answer_ratio_gte_0p95": new_ratio is not None and float(new_ratio) >= 0.95,
        "retention_ratio_gte_0p95": ret_ratio is not None and float(ret_ratio) >= 0.95,
        "reference_ratio_lte_0p85": ref_ratio is not None and float(ref_ratio) <= 0.85,
    }
    strong_checks = {
        "positive_new_answer_edits_gte_18_of_20": int(e_final.get("positive_new_answer_edits") or 0) >= 18,
        "locality_damage_records_eq_0": int(e_final.get("locality_damage_records") or 0) == 0,
        "new_answer_ratio_gte_0p98": new_ratio is not None and float(new_ratio) >= 0.98,
        "retention_ratio_gte_0p98": ret_ratio is not None and float(ret_ratio) >= 0.98,
        "reference_ratio_lte_0p85": ref_ratio is not None and float(ref_ratio) <= 0.85,
    }
    status = "fail"
    if all(basic_checks.values()) and all(relative_checks.values()):
        status = "pass"
    elif all(basic_checks.values()):
        status = "partial"
    projection_diagnostics = _projection_diagnostics_from_trace(trace)
    payload = {
        "status": status,
        "strong": bool(all(strong_checks.values())),
        "rescued_cure_config": RESCUED_CURE,
        "summary_rows": summary_rows,
        "final_rows": final_rows,
        "relative_to_c": {
            "new_answer_ratio": new_ratio,
            "reference_ratio": ref_ratio,
            "retention_ratio": ret_ratio,
        },
        "acceptance": {
            "status": status,
            "basic_checks": basic_checks,
            "relative_checks": relative_checks,
            "strong_checks": strong_checks,
            "c_baseline_final": c_final,
            "rescued_cure_final": e_final,
        },
        "final_rollback_check": rollback_payload,
        "projection_diagnostics": projection_diagnostics,
    }
    return payload


def _projection_diagnostics_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    cure_rows = [row for row in trace if row.get("method") == METHOD_E and int(row.get("step") or 0) > 0]
    modules = sorted({module for row in cure_rows for module in (row.get("modules_with_cache") or [])})
    skipped = sorted({module for row in cure_rows for module in (row.get("skipped_modules") or [])})
    fallback = sorted({module for row in cure_rows for module in (row.get("fallback_modules") or [])})
    skip_reasons: Dict[str, Any] = {}
    for row in cure_rows:
        skip_reasons.update(row.get("skip_reasons") or {})
    return {
        "cache_update_policy": RESCUED_CURE["cache_update_policy"],
        "crisp_energy_threshold": RESCUED_CURE["crisp_energy_threshold"],
        "cure_mix_lambda": RESCUED_CURE["cure_mix_lambda"],
        "cure_norm_clamp": RESCUED_CURE["cure_norm_clamp"],
        "cure_norm_clamp_ratio": RESCUED_CURE["cure_norm_clamp_ratio"],
        "gate_policy": RESCUED_CURE["gate_policy"],
        "record_count": len({row.get("record_id") for row in cure_rows if row.get("record_id")}),
        "accumulated_num_samples": max((int(row.get("accumulated_num_samples") or 0) for row in cure_rows), default=None),
        "modules_with_cache": modules,
        "average_mask_keep_ratio": _mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None]),
        "mean_projection_norm_ratio_preclamp": _mean(
            [row.get("projection_norm_ratio_preclamp") for row in cure_rows if row.get("projection_norm_ratio_preclamp") is not None]
        ),
        "mean_projection_norm_ratio_postclamp": _mean(
            [row.get("projection_norm_ratio_postclamp") for row in cure_rows if row.get("projection_norm_ratio_postclamp") is not None]
        ),
        "mean_delta_candidate_norm": _mean([row.get("delta_candidate_norm") for row in cure_rows if row.get("delta_candidate_norm") is not None]),
        "mean_delta_engram_norm": _mean([row.get("delta_engram_norm") for row in cure_rows if row.get("delta_engram_norm") is not None]),
        "mean_delta_crisp_norm": _mean([row.get("delta_crisp_norm") for row in cure_rows if row.get("delta_crisp_norm") is not None]),
        "mean_delta_cure_preclamp_norm": _mean(
            [row.get("delta_cure_preclamp_norm") for row in cure_rows if row.get("delta_cure_preclamp_norm") is not None]
        ),
        "mean_delta_cure_postclamp_norm": _mean(
            [row.get("delta_cure_postclamp_norm") for row in cure_rows if row.get("delta_cure_postclamp_norm") is not None]
        ),
        "clamp_applied_count": sum(int(row.get("clamp_applied_module_count") or 0) for row in cure_rows),
        "fallback_modules": fallback,
        "fallback_module_count": sum(int(row.get("fallback_module_count") or 0) for row in cure_rows),
        "skipped_modules": skipped,
        "skipped_module_count": sum(len(row.get("skipped_modules") or []) for row in cure_rows),
        "skip_reasons": skip_reasons,
        "identity_like_warning": (
            "mask_keep_ratio near 1.0; delta-space Crisp projection may still be close to identity-like"
            if cure_rows
            and _mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None]) is not None
            and float(_mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None])) >= 0.99
            else None
        ),
    }


def _run_beta0_gate(
    *,
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    record_id_match_rate: float,
    tolerance: float,
) -> Dict[str, Any]:
    beta0_dir = out_dir / "beta0_gate"
    beta0_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    rollback_checks: List[Dict[str, Any]] = []
    cache_trace: List[Dict[str, Any]] = []
    for method in [METHOD_C, METHOD_E]:
        rows, rollback_payload, trace = _run_one_sequential_method(
            model=model,
            method=method,
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=projector_bank_dir,
            module_names=module_names,
            hparams=hparams,
            beta=0.0,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            record_id_match_rate=record_id_match_rate,
            lambda_mix=float(RESCUED_CURE["cure_mix_lambda"]),
            clamp=bool(RESCUED_CURE["cure_norm_clamp"]),
            clamp_ratio=float(RESCUED_CURE["cure_norm_clamp_ratio"]),
        )
        all_rows.extend(rows)
        rollback_checks.append(rollback_payload)
        cache_trace.extend(trace)
    metric_fields = [
        "old_answer_nll_delta_vs_step0",
        "new_answer_nll_decrease_vs_step0",
        "reference_delta_abs_vs_step0",
    ]
    max_metric_abs = 0.0
    bad_rows = []
    for row in all_rows:
        for field in metric_fields:
            value = row.get(field)
            if value is not None:
                abs_value = abs(float(value))
                max_metric_abs = max(max_metric_abs, abs_value)
                if abs_value > float(tolerance):
                    bad_rows.append({"method": row.get("method"), "step": row.get("step"), "record_id": row.get("record_id"), "field": field, "value": value})
    rollback_max = max((float(item.get("rollback_max_abs_diff") or 0.0) for item in rollback_checks), default=0.0)
    summary_rows = [_aggregate_step(all_rows, method, step) for method in [METHOD_C, METHOD_E] for step in range(len(records) + 1)]
    checks = {
        "twenty_edits_iterated_for_each_method": all(max(int(row.get("step") or 0) for row in all_rows if row.get("method") == method) == len(records) for method in [METHOD_C, METHOD_E]),
        "nll_metrics_unchanged": max_metric_abs <= float(tolerance),
        "selected_weights_unchanged": rollback_max <= float(rollback_tolerance),
        "final_rollback_pass": all(item.get("rollback_pass") for item in rollback_checks),
        "record_id_match_rate_is_1": float(record_id_match_rate) == 1.0,
        "no_nan_inf": not any(row.get("nan_inf_detected") for row in all_rows),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "beta": 0.0,
        "tolerance": float(tolerance),
        "checks": checks,
        "max_metric_abs_delta": max_metric_abs,
        "bad_rows_sample": bad_rows[:20],
        "summary_rows": summary_rows,
        "final_rollback_check": {
            "status": "pass" if all(item.get("rollback_pass") for item in rollback_checks) else "fail",
            "methods": rollback_checks,
            "max_abs_diff": rollback_max,
        },
        "crisp_cache_update_trace": cache_trace,
    }
    _json_dump(beta0_dir / "beta0_step_matrix.json", all_rows)
    _write_csv(beta0_dir / "beta0_step_matrix.csv", all_rows)
    _json_dump(beta0_dir / "beta0_summary.json", payload)
    _write_csv(beta0_dir / "beta0_summary.csv", summary_rows)
    _json_dump(beta0_dir / "beta0_crisp_cache_update_trace.json", cache_trace)
    lines = [
        "# CURE Rescue Sequential Beta-0 Gate",
        "",
        f"- Status: `{payload['status']}`",
        f"- Max metric abs delta: `{max_metric_abs:.8g}`",
        f"- Rollback max abs diff: `{rollback_max:.8g}`",
        f"- Record-id match rate: `{record_id_match_rate}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    if bad_rows:
        lines.extend(["", "## Bad Rows Sample", "", "```json", json.dumps(bad_rows[:20], indent=2, sort_keys=True), "```"])
    (beta0_dir / "REPORT_CURE_RESCUE_SEQUENTIAL_BETA0_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _run_sequential_comparison(
    *,
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    record_id_match_rate: float,
) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    rollback_checks: List[Dict[str, Any]] = []
    cache_trace: List[Dict[str, Any]] = []
    for method in [METHOD_C, METHOD_E]:
        rows, rollback_payload, trace = _run_one_sequential_method(
            model=model,
            method=method,
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=projector_bank_dir,
            module_names=module_names,
            hparams=hparams,
            beta=float(RESCUED_CURE["beta"]),
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            record_id_match_rate=record_id_match_rate,
            lambda_mix=float(RESCUED_CURE["cure_mix_lambda"]),
            clamp=bool(RESCUED_CURE["cure_norm_clamp"]),
            clamp_ratio=float(RESCUED_CURE["cure_norm_clamp_ratio"]),
        )
        all_rows.extend(rows)
        rollback_checks.append(rollback_payload)
        cache_trace.extend(trace)
    summary = _summarize_final(rows=all_rows, rollback_checks=rollback_checks, trace=cache_trace, tolerance=rollback_tolerance)
    payload = {
        **summary,
        "per_record_step_rows": all_rows,
        "crisp_cache_update_trace": cache_trace,
        "edit_record_matching": {"record_id_match_rate": record_id_match_rate, "positional_matching_used": False},
    }
    _json_dump(out_dir / "sequential_step_matrix.json", all_rows)
    _write_csv(out_dir / "sequential_step_matrix.csv", all_rows)
    _json_dump(out_dir / "sequential_summary.json", payload)
    _write_csv(out_dir / "sequential_summary.csv", summary["summary_rows"])
    _json_dump(out_dir / "projection_diagnostics.json", summary["projection_diagnostics"])
    _json_dump(out_dir / "crisp_cache_update_trace.json", cache_trace)
    _json_dump(out_dir / "final_rollback_check.json", summary["final_rollback_check"])
    return payload


def _write_generation_diagnostics(out_dir: Path, *, skipped: bool) -> Dict[str, Any]:
    payload = {
        "status": "skipped" if skipped else "not_implemented",
        "reason": "main gate uses --skip-generation; optional generation diagnostics were not launched",
        "primary_gate": "NLL/logprob based sequential validation",
    }
    _json_dump(out_dir / "generation_diagnostics.json", payload)
    return payload


def _write_plots(out_dir: Path, summary_rows: List[Dict[str, Any]], projection_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        _json_dump(plot_dir / "plot_status.json", payload)
        return payload
    methods = [METHOD_C, METHOD_E]
    labels = {METHOD_C: "C", METHOD_E: "rescued CURE"}
    plot_specs = [
        ("sequential_new_answer_curve.png", "mean_new_answer_nll_decrease", "mean new answer NLL decrease"),
        ("sequential_reference_delta_curve.png", "mean_reference_delta_abs_all_records", "mean reference delta abs"),
        ("sequential_retention_curve.png", "previous_edit_retention", "previous-edit retention"),
    ]
    paths: List[str] = []
    for filename, field, ylabel in plot_specs:
        plt.figure(figsize=(7, 4))
        for method in methods:
            rows = [row for row in summary_rows if row.get("method") == method]
            rows.sort(key=lambda row: int(row.get("step") or 0))
            plt.plot([row.get("step") for row in rows], [row.get(field) for row in rows], marker="o", label=labels[method])
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = plot_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    cure_trace = [row for row in projection_trace if row.get("method") == METHOD_E and int(row.get("step") or 0) > 0]
    plt.figure(figsize=(7, 4))
    plt.plot(
        [row.get("step") for row in cure_trace],
        [row.get("projection_norm_ratio_preclamp") for row in cure_trace],
        marker="o",
        label="preclamp",
    )
    plt.plot(
        [row.get("step") for row in cure_trace],
        [row.get("projection_norm_ratio_postclamp") for row in cure_trace],
        marker="o",
        label="postclamp",
    )
    plt.xlabel("step")
    plt.ylabel("projection norm ratio")
    plt.legend()
    plt.tight_layout()
    path = plot_dir / "projection_norm_curve.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))
    payload = {"status": "complete", "plots": paths}
    _json_dump(plot_dir / "plot_status.json", payload)
    return payload


def _write_final_report(
    out_dir: Path,
    *,
    data_reuse: Dict[str, Any],
    lambda0: Dict[str, Any],
    beta0: Dict[str, Any],
    sequential: Dict[str, Any],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
) -> None:
    final_rows = sequential.get("final_rows") or []
    rel = sequential.get("relative_to_c") or {}
    acceptance = sequential.get("acceptance") or {}
    projection = sequential.get("projection_diagnostics") or {}
    c_final = acceptance.get("c_baseline_final") or {}
    e_final = acceptance.get("rescued_cure_final") or {}
    status = sequential.get("status")
    decision = "C. Rescued CURE fails sequential. Stop CURE scale-up; focus on ENGRAM-projected LoRA and generation-level validation."
    if status == "pass" and sequential.get("strong"):
        decision = "A. Rescued CURE passes 20-edit sequential and remains Pareto-promising. Next: generation-level validation or public non-PHI medical benchmark."
    elif status in {"pass", "partial"}:
        decision = "B. Rescued CURE passes basic sequential but does not beat C baseline. Keep ENGRAM-projected LoRA as primary; CURE is optional conservative variant."
    lines = [
        "# Final CURE 20-Edit Rescue Sequential Report",
        "",
        "## Starting Point",
        "",
        "- Original 20-edit full CURE nonseq failed because reference drift was worse than `C_engram_projected_tiny_lora`.",
        "- Rescued nonseq CURE passed after adding lambda mixing, norm clamp, and gate fallback.",
        f"- Best rescued config: `{RESCUED_CURE['config_id']}`.",
        "",
        "## Dataset",
        "",
        f"- Status: `{data_reuse.get('status')}`",
        f"- Record count: `{data_reuse.get('record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        f"- Positional matching used: `{data_reuse.get('positional_matching_used')}`",
        f"- Selected record IDs: `{data_reuse.get('selected_record_ids')}`",
        "- Data source: synthetic non-PHI engineering fixtures; no private or patient data.",
        "",
        "## Methods",
        "",
        f"- `{METHOD_C}`",
        f"- `{METHOD_E}`",
        "- Direct ENGRAM erase, full unrescued CURE, and extra CURE sweeps were not run.",
        "",
        "## Verification",
        "",
        f"- Lambda=0 equivalence status: `{lambda0.get('status')}`; max abs diff `{lambda0.get('lambda0_max_abs_diff')}`.",
        f"- Beta=0 no-change gate: `{beta0.get('status')}`; max metric abs delta `{beta0.get('max_metric_abs_delta')}`.",
        f"- Beta=0 rollback: `{(beta0.get('final_rollback_check') or {}).get('status')}`.",
        "",
        "## Sequential Final-Step Aggregates",
        "",
        "| method | mean new decrease | mean reference delta all | retention | positive new | locality damage | rollback | match | nan/inf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            "| {method} | {new} | {ref} | {ret} | {pos} | {loc} | {roll} | {match} | {nan} |".format(
                method=row.get("method"),
                new=_fmt(row.get("mean_new_answer_nll_decrease")),
                ref=_fmt(row.get("mean_reference_delta_abs_all_records")),
                ret=_fmt(row.get("previous_edit_retention")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_records"),
                roll=_fmt(row.get("rollback_pass_rate")),
                match=_fmt(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Relative Comparison",
            "",
            f"- new_answer_ratio: `{_fmt(rel.get('new_answer_ratio'))}`",
            f"- reference_ratio: `{_fmt(rel.get('reference_ratio'))}`",
            f"- retention_ratio: `{_fmt(rel.get('retention_ratio'))}`",
            "",
            "## Acceptance",
            "",
            f"- Status: `{acceptance.get('status')}`",
            f"- Strong: `{sequential.get('strong')}`",
            f"- C final: `{_compact_final(c_final)}`",
            f"- Rescued CURE final: `{_compact_final(e_final)}`",
            "",
            "## Projection Diagnostics",
            "",
            f"- mask_keep_ratio: `{_fmt(projection.get('average_mask_keep_ratio'))}`",
            f"- projection_norm_ratio_preclamp: `{_fmt(projection.get('mean_projection_norm_ratio_preclamp'))}`",
            f"- projection_norm_ratio_postclamp: `{_fmt(projection.get('mean_projection_norm_ratio_postclamp'))}`",
            f"- clamp_applied_count: `{projection.get('clamp_applied_count')}`",
            f"- skipped modules: `{projection.get('skipped_modules')}`",
            f"- fallback modules: `{projection.get('fallback_modules')}`",
            f"- gate_policy: `{projection.get('gate_policy')}`",
            f"- norm clamp active: `{RESCUED_CURE['cure_norm_clamp']}`",
            "",
            "## Crisp Cache Diagnostics",
            "",
            f"- cache_update_policy: `{projection.get('cache_update_policy')}`",
            f"- accumulated samples: `{projection.get('accumulated_num_samples')}`",
            f"- modules with cache: `{projection.get('modules_with_cache')}`",
            f"- identity-like warning: `{projection.get('identity_like_warning')}`",
            "",
            "## Generation",
            "",
            f"- Status: `{generation.get('status')}`",
            "- Main gate used `--skip-generation`; evidence is NLL/logprob-based.",
            "",
            "## Plots",
            "",
            f"- Status: `{plots.get('status')}`",
            f"- Files: `{plots.get('plots')}`",
            "",
            "## Limitations",
            "",
            "- Non-PHI synthetic 20-edit engineering validation.",
            "- No medical or clinical efficacy claim.",
            "- Delta-space Crisp projection is not original CrispEdit gradient-projected training.",
            "- Generation-level behavior is not established by the main gate.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_CURE_20EDIT_RESCUE_SEQUENTIAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.6g}"


def _compact_final(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean_new_answer_nll_decrease": row.get("mean_new_answer_nll_decrease"),
        "mean_reference_delta_abs_all_records": row.get("mean_reference_delta_abs_all_records"),
        "previous_edit_retention": row.get("previous_edit_retention"),
        "positive_new_answer_edits": row.get("positive_new_answer_edits"),
        "locality_damage_records": row.get("locality_damage_records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 20-edit rescued CURE sequential validation.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml")
    parser.add_argument("--source-output-dir", default="outputs/cure_mededit_20edit_modelknown")
    parser.add_argument("--rescue-nonseq-dir", default="outputs/cure_mededit_20edit_rescue_nonseq")
    parser.add_argument("--output-dir", default="outputs/cure_mededit_20edit_rescue_sequential")
    parser.add_argument("--projector-bank-dir", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--beta0-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--lambda0-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)

    source_output_dir = Path(args.source_output_dir)
    rescue_nonseq_dir = Path(args.rescue_nonseq_dir)
    projector_bank_dir = Path(args.projector_bank_dir) if args.projector_bank_dir else source_output_dir / "projector_bank"
    records, data_path, image_root, data_reuse = _reuse_selected_data(
        source_output_dir=source_output_dir,
        rescue_nonseq_dir=rescue_nonseq_dir,
        out_dir=out_dir,
    )
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        source_output_dir=source_output_dir,
        rescue_nonseq_dir=rescue_nonseq_dir,
        data_path=data_path,
        image_root=image_root,
        projector_bank_dir=projector_bank_dir,
        test_status=test_status,
    )
    if preflight["status"] != "pass" or data_reuse["status"] != "pass":
        _write_stop_report(out_dir, "preflight_or_data_reuse_failed", {"preflight": preflight, "data_reuse": data_reuse})
        return 0

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(hparams, image_root=image_root, bank_dir=projector_bank_dir, device=args.device, edit_mode="erase")
    hparams.replacement_mode = "cure_delta_projected"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.use_crisp_projection = True
    hparams.crisp_energy_threshold = float(RESCUED_CURE["crisp_energy_threshold"])
    hparams.crisp_cache_update_policy = str(RESCUED_CURE["cache_update_policy"])
    hparams.lora_rank = 4
    hparams.lora_steps = 20
    hparams.token_scope = "all"
    shutil.copyfile(args.hparams, out_dir / "base_hparams.used.yaml")

    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    selected_status = {
        "status": "pass" if set(selected) == set(EXPECTED_MODULES) and len(selected) == len(EXPECTED_MODULES) else "fail",
        "selected_module_names": selected,
        "expected_module_names": EXPECTED_MODULES,
    }
    _json_dump(out_dir / "selected_modules_preflight.json", selected_status)
    if selected_status["status"] != "pass":
        _write_stop_report(out_dir, "selected_modules_preflight_failed", selected_status)
        return 0

    edit_ids, record_preflight = _match_bank_records(records, projector_bank_dir, out_dir)
    if record_preflight["status"] != "pass" or len(edit_ids) != len(records):
        _write_stop_report(out_dir, "record_id_preflight_failed", record_preflight)
        return 0
    record_id_match_rate = float(record_preflight.get("record_id_match_rate") or 0.0)

    baselines, baseline_report = _load_or_eval_baselines(
        model=editor.model,
        records=records,
        image_root=image_root,
        source_output_dir=source_output_dir,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
    )
    _json_dump(out_dir / "baseline_metrics.json", baselines)
    _json_dump(out_dir / "baseline_reuse_report.json", baseline_report)

    lambda0 = _lambda0_equivalence_check(
        model=editor.model,
        records=records,
        image_root=image_root,
        baselines=baselines,
        projector_bank_dir=projector_bank_dir,
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        out_dir=out_dir,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
        tolerance=args.lambda0_tolerance,
    )
    if lambda0["status"] != "pass":
        _write_stop_report(out_dir, "lambda0_equivalence_failed", lambda0)
        return 0

    beta0 = _run_beta0_gate(
        model=editor.model,
        records=records,
        image_root=image_root,
        baselines=baselines,
        projector_bank_dir=projector_bank_dir,
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        out_dir=out_dir,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
        record_id_match_rate=record_id_match_rate,
        tolerance=args.beta0_tolerance,
    )
    if beta0["status"] != "pass":
        _write_stop_report(out_dir, "beta0_gate_failed", beta0)
        return 0

    sequential = _run_sequential_comparison(
        model=editor.model,
        records=records,
        image_root=image_root,
        baselines=baselines,
        projector_bank_dir=projector_bank_dir,
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        out_dir=out_dir,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
        record_id_match_rate=record_id_match_rate,
    )
    generation = _write_generation_diagnostics(out_dir, skipped=True)
    plots = _write_plots(out_dir, sequential.get("summary_rows") or [], sequential.get("crisp_cache_update_trace") or [])
    _write_final_report(
        out_dir,
        data_reuse=data_reuse,
        lambda0=lambda0,
        beta0=beta0,
        sequential=sequential,
        generation=generation,
        plots=plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
