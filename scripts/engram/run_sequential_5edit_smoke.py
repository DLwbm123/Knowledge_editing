#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from easyeditor.dataset.coco_caption import CaptionDataset  # noqa: E402
from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_token_module_ablation_5edit import (  # noqa: E402
    _answer_metrics,
    _extract_layer_depth,
    _finite,
    _format,
    _json_dump,
    _max_snapshot_diff,
    _maybe_generate,
    _mean,
    _module_map,
    _reference_sample,
    _resolve_image,
    _restore_modules,
    _snapshot_modules,
    _target_sample,
    _write_csv,
)


EXPECTED_MODULES = [
    "llava_model.model.layers.0.mlp.gate_proj",
    "llava_model.model.layers.8.mlp.gate_proj",
    "llava_model.model.layers.16.mlp.gate_proj",
    "llava_model.model.layers.24.mlp.gate_proj",
    "llava_model.model.layers.16.self_attn.q_proj",
    "llava_model.model.layers.24.self_attn.q_proj",
    "llava_model.model.layers.16.self_attn.k_proj",
    "llava_model.model.layers.24.self_attn.k_proj",
]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 5:
        raise RuntimeError(f"Sequential smoke expects exactly 5 records, got {len(records) if isinstance(records, list) else type(records)}.")
    return records


def _run_capture(command: str, cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def _write_git_outputs(out_dir: Path) -> None:
    (out_dir / "git_status.txt").write_text(_run_capture("git status", PROJECT_ROOT), encoding="utf-8")
    (out_dir / "git_diff.patch").write_text(_run_capture("git diff", PROJECT_ROOT), encoding="utf-8")


def _exact_module_patterns() -> List[str]:
    return [rf"^{re.escape(name)}$" for name in EXPECTED_MODULES]


def _configure_hparams(
    hparams: EngramMultimodalHparams,
    *,
    alpha: float,
    bank_dir: Path,
    image_root: Path,
    device: str,
) -> None:
    hparams.device = int(device) if str(device).isdigit() else device
    dataset_image_root = image_root.parent if image_root.name == "images" else image_root
    hparams.coco_image = str(dataset_image_root)
    hparams.rephrase_image = str(dataset_image_root)
    hparams.edit_mode = "erase"
    hparams.engram_update_direction = "add"
    hparams.token_scope = "all"
    hparams.sequential_edit = True
    hparams.alpha = float(alpha)
    hparams.covariance_device = "cpu"
    hparams.solve_device = "cpu"
    hparams.max_cov_dim = 4097
    hparams.skip_if_dim_larger_than = 4097
    hparams.module_patterns = _exact_module_patterns()
    hparams.exclude_module_patterns = [r"lm_head$", r"down_proj$"]
    hparams.prioritize_module_selection = True
    hparams.module_priority_patterns = [r"gate_proj$", r"q_proj$", r"k_proj$"]
    hparams.engram_layers = [0, 8, 16, 24]
    hparams.engram_max_modules = None
    hparams.bank_dir = str(bank_dir)
    hparams.engram_bank_path = str(bank_dir)
    hparams.edit_id = None
    hparams.engram_edit_id = None


def _validate_required_hparams(hparams: EngramMultimodalHparams, alpha: float) -> Dict[str, Any]:
    checks = {
        "edit_mode": hparams.edit_mode == "erase",
        "engram_update_direction": hparams.engram_update_direction == "add",
        "token_scope": hparams.resolved_token_scope() == "all",
        "sequential_edit": bool(hparams.sequential_edit),
        "alpha": abs(float(hparams.resolved_alpha()) - float(alpha)) < 1.0e-12,
        "covariance_device": hparams.resolved_covariance_device() == "cpu",
        "solve_device": hparams.resolved_solve_device() == "cpu",
        "max_cov_dim": int(hparams.max_cov_dim or 0) == 4097,
        "skip_if_dim_larger_than": int(hparams.skip_if_dim_larger_than or 0) == 4097,
        "replacement_mode_disabled": str(hparams.edit_mode).lower() != "replacement",
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _request_dataset(data_file: Path, hparams: EngramMultimodalHparams) -> CaptionDataset:
    return CaptionDataset(str(data_file), config=hparams)


def _verify_selected_modules(model: torch.nn.Module, hparams: EngramMultimodalHparams, out_path: Path) -> List[str]:
    layers = select_linear_layers(model, hparams)
    selected = [layer.name for layer in layers]
    payload = {
        "status": "pass" if set(selected) == set(EXPECTED_MODULES) and len(selected) == len(EXPECTED_MODULES) else "fail",
        "selected_module_names": selected,
        "expected_module_names": EXPECTED_MODULES,
        "selected_layer_depths": sorted(
            {depth for depth in (_extract_layer_depth(name) for name in selected) if depth is not None}
        ),
        "module_policy": {
            "q_proj_k_proj_gate_proj_only": True,
            "sampled_depths": [0, 8, 16, 24],
            "exclude_mm_projector": True,
            "exclude_lm_head": True,
            "exclude_down_proj": True,
            "intentional_exact_module_lock": (
                "The successful non-sequential ablation updated these 8 modules. The sequential gate locks "
                "to the same module names so alpha/2 does not silently change module scope."
            ),
        },
    }
    _json_dump(out_path, payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"Selected modules differ from successful ablation: {payload}")
    return selected


def _read_metadata_only(bank_dir: Path, edit_id: str) -> Dict[str, Any]:
    path = bank_dir / "edits" / edit_id / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    layers = list(metadata.get("layers") or [])
    selected = list(metadata.get("selected_modules") or [])
    target_count = sum(int(row.get("num_target_vectors") or 0) for row in layers)
    reference_count = sum(int(row.get("num_reference_vectors") or 0) for row in layers)
    effective = [float(row.get("effective_update_norm_ratio", row.get("effective_norm_ratio", 0.0)) or 0.0) for row in layers]
    return {
        "selected_modules": selected,
        "selected_layer_depths": sorted(
            {depth for depth in (_extract_layer_depth(name) for name in selected) if depth is not None}
        ),
        "target_activation_count": target_count,
        "reference_activation_count": reference_count,
        "max_effective_update_norm_ratio": max(effective) if effective else 0.0,
        "norm_ratios": {str(row.get("module_name")): row.get("norm_ratio") for row in layers if row.get("module_name")},
        "effective_norm_ratios": {
            str(row.get("module_name")): row.get("effective_update_norm_ratio", row.get("effective_norm_ratio"))
            for row in layers
            if row.get("module_name")
        },
    }


def _param_norms(model: torch.nn.Module, module_names: Iterable[str]) -> Dict[str, Dict[str, Optional[float]]]:
    modules = _module_map(model)
    norms: Dict[str, Dict[str, Optional[float]]] = {}
    for name in module_names:
        module = modules[name]
        norms[name] = {
            "weight_norm": float(module.weight.detach().float().norm().cpu()),
            "bias_norm": float(module.bias.detach().float().norm().cpu()) if module.bias is not None else None,
        }
    return norms


def _evaluate_one(
    model: torch.nn.Module,
    record: Dict[str, Any],
    image_root: Path,
    *,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    target = _target_sample(record, image_root)
    reference = _reference_sample(record, image_root)
    return {
        "target_raw": _answer_metrics(model, dict(target)),
        "reference_raw": _answer_metrics(model, dict(reference)) if reference else None,
        "generation": _maybe_generate(model, record, image_root, max_new_tokens, min_new_tokens, skip_generation),
    }


def _metric_value(raw: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not raw or not raw.get("available") or raw.get(key) is None:
        return None
    return float(raw[key])


def _delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)


def _evaluate_step(
    model: torch.nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    *,
    step: int,
    applied_record_ids: List[str],
    per_record_extract: Dict[str, Dict[str, Any]],
    selected_modules_applied: List[str],
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_index, record in enumerate(records):
        record_id = str(record.get("id"))
        now = _evaluate_one(
            model,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
        )
        base = baselines[record_id]
        target_nll = _metric_value(now["target_raw"], "nll")
        target_logprob = _metric_value(now["target_raw"], "logprob")
        target_nll0 = _metric_value(base["target_raw"], "nll")
        target_logprob0 = _metric_value(base["target_raw"], "logprob")
        ref_nll = _metric_value(now["reference_raw"], "nll")
        ref_logprob = _metric_value(now["reference_raw"], "logprob")
        ref_nll0 = _metric_value(base["reference_raw"], "nll")
        target_delta = _delta(target_nll, target_nll0)
        ref_delta = _delta(ref_nll, ref_nll0)
        generation = now["generation"]
        extract_summary = per_record_extract.get(record_id, {})
        rows.append(
            {
                "step": step,
                "applied_record_ids": list(applied_record_ids),
                "record_id": record_id,
                "case_index": case_index,
                "is_already_edited": case_index < step,
                "is_current_edit": step > 0 and case_index == step - 1,
                "is_future_edit": case_index >= step,
                "target_nll": target_nll,
                "target_logprob": target_logprob,
                "target_nll_delta_vs_step0": target_delta,
                "target_logprob_drop_vs_step0": None if target_logprob is None or target_logprob0 is None else target_logprob0 - target_logprob,
                "reference_nll": ref_nll,
                "reference_logprob": ref_logprob,
                "reference_delta_abs_vs_step0": abs(ref_delta) if ref_delta is not None else None,
                "generation": generation,
                "generation_empty": generation.get("generation_empty") if isinstance(generation, dict) else None,
                "immediate_eos": generation.get("stop_reason") == "immediate_eos" if isinstance(generation, dict) else None,
                "selected_modules": list(selected_modules_applied),
                "target_activation_count": extract_summary.get("target_activation_count") if case_index < step else None,
                "reference_activation_count": extract_summary.get("reference_activation_count") if case_index < step else None,
                "metrics_available": target_delta is not None,
                "nan_inf_detected": not _finite(now),
            }
        )
    return rows


def _aggregate_step(rows: List[Dict[str, Any]], locality_threshold: float, rollback_status: Optional[str] = None) -> Dict[str, Any]:
    step = int(rows[0]["step"]) if rows else 0
    edited = [row for row in rows if row.get("is_already_edited")]
    future = [row for row in rows if row.get("is_future_edit")]
    target_edited = [float(row["target_nll_delta_vs_step0"]) for row in edited if row.get("target_nll_delta_vs_step0") is not None]
    ref_all = [float(row["reference_delta_abs_vs_step0"]) for row in rows if row.get("reference_delta_abs_vs_step0") is not None]
    ref_edited = [float(row["reference_delta_abs_vs_step0"]) for row in edited if row.get("reference_delta_abs_vs_step0") is not None]
    ref_future = [float(row["reference_delta_abs_vs_step0"]) for row in future if row.get("reference_delta_abs_vs_step0") is not None]
    future_target_abs = [abs(float(row["target_nll_delta_vs_step0"])) for row in future if row.get("target_nll_delta_vs_step0") is not None]
    synthetic = next((row for row in rows if row.get("record_id") == "synthetic-5edit-2"), {})
    mean_target = _mean(target_edited)
    mean_ref_all = _mean(ref_all)
    return {
        "step": step,
        "applied_record_ids": rows[0].get("applied_record_ids") if rows else [],
        "mean_target_nll_increase_edited_so_far": mean_target,
        "mean_reference_delta_abs_all_records": mean_ref_all,
        "mean_reference_delta_abs_edited_so_far": _mean(ref_edited),
        "mean_reference_delta_abs_future_records": _mean(ref_future),
        "positive_target_edits_so_far": sum(1 for value in target_edited if value > 0),
        "locality_damage_records": sum(1 for value in ref_all if value > locality_threshold),
        "future_target_drift": _mean(future_target_abs),
        "rollback_status_if_final_step": rollback_status,
        "record_id_match_rate": 1.0,
        "nan_inf_count": sum(1 for row in rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in rows if row.get("generation_empty")),
        "immediate_eos_count": sum(1 for row in rows if row.get("immediate_eos")),
        "synthetic_5edit_2_target_nll_delta_vs_step0": synthetic.get("target_nll_delta_vs_step0"),
        "synthetic_5edit_2_reference_delta_abs_vs_step0": synthetic.get("reference_delta_abs_vs_step0"),
        "score": None if mean_target is None or mean_ref_all is None else mean_target - mean_ref_all,
    }


def _bank_record_id_report(bank_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    bank = EngramBank(bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    rows = []
    for record, edit_id in zip(records, edit_ids):
        meta = _read_metadata_only(bank_dir, edit_id)
        bank_record_id = meta.get("record_id") or meta.get("source_record_id")
        rows.append(
            {
                "raw_record_id": record.get("id"),
                "bank_record_id": bank_record_id,
                "edit_id": edit_id,
                "matching_mode": matching.get("mode"),
                "record_id_match": str(record.get("id")) == str(bank_record_id),
            }
        )
    return {
        "status": "pass" if matching.get("mode") == "record_id" and all(row["record_id_match"] for row in rows) else "fail",
        "matching": matching,
        "edit_ids": edit_ids,
        "rows": rows,
        "record_id_match_rate": _mean([1.0 if row["record_id_match"] else 0.0 for row in rows]),
    }


def _run_sequence(
    *,
    editor: MultimodalEditor,
    hparams: EngramMultimodalHparams,
    records: List[Dict[str, Any]],
    data_file: Path,
    image_root: Path,
    run_dir: Path,
    tag: str,
    alpha: float,
    selected_modules: List[str],
    rollback_tolerance: float,
    metric_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = run_dir / "bank"
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    _configure_hparams(hparams, alpha=alpha, bank_dir=bank_dir, image_root=image_root, device=str(hparams.device))
    ds = _request_dataset(data_file, hparams)
    if len(ds) != len(records):
        raise RuntimeError(f"Dataset length mismatch: CaptionDataset={len(ds)} raw={len(records)}")

    wrapper = editor.model
    snapshots = _snapshot_modules(wrapper, selected_modules)
    param_norm_before = _param_norms(wrapper, selected_modules)
    baselines = {
        str(record.get("id")): _evaluate_one(
            wrapper,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
        )
        for record in records
    }

    matrix_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    applied_record_ids: List[str] = []
    per_record_extract: Dict[str, Dict[str, Any]] = {}
    selected_modules_applied: List[str] = []

    step0_rows = _evaluate_step(
        wrapper,
        records,
        image_root,
        baselines,
        step=0,
        applied_record_ids=applied_record_ids,
        per_record_extract=per_record_extract,
        selected_modules_applied=selected_modules_applied,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        skip_generation=skip_generation,
    )
    matrix_rows.extend(step0_rows)
    summary_rows.append(_aggregate_step(step0_rows, locality_threshold))

    extraction_rows = []
    for idx, request in enumerate(ds):
        record = records[idx]
        record_id = str(record.get("id"))
        request["id"] = record_id
        request["record_id"] = record_id
        request["source_record_id"] = record_id
        hparams.edit_id = f"{tag}__{record_id}"
        hparams.engram_edit_id = hparams.edit_id
        editor.apply_algo(
            wrapper,
            editor.tok,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
            train_ds=None,
        )
        metadata = _read_metadata_only(bank_dir, hparams.edit_id)
        bank_record_id = metadata.get("record_id") or metadata.get("source_record_id")
        summary = _summarize_metadata(metadata)
        if set(summary["selected_modules"]) != set(EXPECTED_MODULES):
            _json_dump(
                run_dir / f"selected_module_mismatch_step{idx + 1}.json",
                {"record_id": record_id, "metadata_selected_modules": summary["selected_modules"], "expected": EXPECTED_MODULES},
            )
            raise RuntimeError(f"Bank selected modules differ for {record_id}: {summary['selected_modules']}")
        if str(bank_record_id) != record_id:
            raise RuntimeError(f"Bank metadata record_id mismatch: raw={record_id} bank={bank_record_id}")
        per_record_extract[record_id] = summary
        applied_record_ids.append(record_id)
        selected_modules_applied = sorted(set(selected_modules_applied).union(summary["selected_modules"]))
        extraction_rows.append(
            {
                "step": idx + 1,
                "record_id": record_id,
                "edit_id": hparams.edit_id,
                "bank_saved": (bank_dir / "edits" / hparams.edit_id / "tensors.pt").exists(),
                "bank_record_id": bank_record_id,
                **summary,
            }
        )
        step_rows = _evaluate_step(
            wrapper,
            records,
            image_root,
            baselines,
            step=idx + 1,
            applied_record_ids=applied_record_ids,
            per_record_extract=per_record_extract,
            selected_modules_applied=selected_modules_applied,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
        )
        matrix_rows.extend(step_rows)
        summary_rows.append(_aggregate_step(step_rows, locality_threshold))

    bank_report = _bank_record_id_report(bank_dir, records)
    param_norm_after_step5 = _param_norms(wrapper, selected_modules)
    max_diff_before_restore = _max_snapshot_diff(wrapper, snapshots)
    _restore_modules(wrapper, snapshots)
    final_rollback_diff = _max_snapshot_diff(wrapper, snapshots)
    param_norm_after_rollback = _param_norms(wrapper, selected_modules)
    rollback_status = "pass" if final_rollback_diff <= rollback_tolerance else "fail"
    summary_rows[-1]["rollback_status_if_final_step"] = rollback_status

    rollback_rows = []
    modules = _module_map(wrapper)
    for module_name in selected_modules:
        before = snapshots[module_name]
        module = modules[module_name]
        weight_diff = float((module.weight.detach().cpu() - before["weight"]).abs().max().item())
        bias_diff = 0.0
        if module.bias is not None and before["bias"] is not None:
            bias_diff = float((module.bias.detach().cpu() - before["bias"]).abs().max().item())
        rollback_rows.append(
            {
                "module_name": module_name,
                "param_norm_before": param_norm_before[module_name],
                "param_norm_after_step5": param_norm_after_step5[module_name],
                "param_norm_after_final_rollback": param_norm_after_rollback[module_name],
                "max_abs_diff_before_vs_after_final_rollback": max(weight_diff, bias_diff),
            }
        )
    rollback_check = {
        "status": rollback_status,
        "max_abs_diff_before_vs_after_step5": max_diff_before_restore,
        "max_abs_diff_before_vs_after_final_rollback": final_rollback_diff,
        "tolerance": rollback_tolerance,
        "modules": rollback_rows,
    }

    payload = {
        "status": "complete",
        "tag": tag,
        "alpha": alpha,
        "bank_dir": str(bank_dir),
        "bank_record_id_report": bank_report,
        "extraction_rows": extraction_rows,
        "sequential_step_matrix": matrix_rows,
        "sequential_step_summary": summary_rows,
        "final_rollback_check": rollback_check,
    }
    _json_dump(run_dir / "sequential_step_matrix.json", matrix_rows)
    _write_csv(run_dir / "sequential_step_matrix.csv", matrix_rows)
    _json_dump(run_dir / "sequential_step_summary.json", summary_rows)
    _write_csv(run_dir / "sequential_step_summary.csv", summary_rows)
    _json_dump(run_dir / "bank_metadata_summary.json", extraction_rows)
    _write_csv(run_dir / "bank_metadata_summary.csv", extraction_rows)
    _json_dump(run_dir / "final_rollback_check.json", rollback_check)
    _json_dump(run_dir / "run_payload.json", payload)

    if abs(alpha) <= 1.0e-12:
        alpha0_checks = _alpha0_checks(
            records=records,
            matrix_rows=matrix_rows,
            summary_rows=summary_rows,
            extraction_rows=extraction_rows,
            bank_report=bank_report,
            rollback_check=rollback_check,
            metric_tolerance=metric_tolerance,
        )
        _json_dump(run_dir / "alpha0_checks.json", alpha0_checks)
        _write_alpha0_report(run_dir / "REPORT_SEQUENTIAL_ALPHA0_GATE.md", alpha0_checks, summary_rows[-1], extraction_rows)
        if alpha0_checks["status"] != "pass":
            raise RuntimeError(f"Sequential alpha=0 gate failed: {alpha0_checks}")

    return payload


def _alpha0_checks(
    *,
    records: List[Dict[str, Any]],
    matrix_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    extraction_rows: List[Dict[str, Any]],
    bank_report: Dict[str, Any],
    rollback_check: Dict[str, Any],
    metric_tolerance: float,
) -> Dict[str, Any]:
    final_rows = [row for row in matrix_rows if int(row.get("step") or -1) == 5]
    target_deltas = [abs(float(row.get("target_nll_delta_vs_step0") or 0.0)) for row in final_rows]
    reference_deltas = [abs(float(row.get("reference_delta_abs_vs_step0") or 0.0)) for row in final_rows]
    checks = {
        "five_edits_run": len(extraction_rows) == 5,
        "target_activation_counts_nonzero": all(int(row.get("target_activation_count") or 0) > 0 for row in extraction_rows),
        "reference_activation_counts_nonzero": all(int(row.get("reference_activation_count") or 0) > 0 for row in extraction_rows),
        "effective_update_norm_zero": all(float(row.get("max_effective_update_norm_ratio") or 0.0) == 0.0 for row in extraction_rows),
        "target_nll_unchanged": all(value <= metric_tolerance for value in target_deltas),
        "reference_nll_unchanged": all(value <= metric_tolerance for value in reference_deltas),
        "final_selected_weights_unchanged": float(rollback_check.get("max_abs_diff_before_vs_after_step5") or 0.0) <= metric_tolerance,
        "final_rollback_pass": rollback_check.get("status") == "pass",
        "record_id_matching_used": bank_report.get("status") == "pass" and bank_report.get("matching", {}).get("mode") == "record_id",
        "no_nan_inf": all(int(row.get("nan_inf_count") or 0) == 0 for row in summary_rows),
        "raw_records_with_record_id": sum(1 for record in records if record.get("id")) == 5,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metric_tolerance": metric_tolerance,
        "max_abs_target_nll_delta_final": max(target_deltas) if target_deltas else None,
        "max_abs_reference_nll_delta_final": max(reference_deltas) if reference_deltas else None,
    }


def _write_alpha0_report(path: Path, checks: Dict[str, Any], final_summary: Dict[str, Any], extraction_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Sequential Alpha=0 Gate",
        "",
        f"- Status: `{checks.get('status')}`",
        f"- Metric tolerance: `{checks.get('metric_tolerance')}`",
        f"- Max target NLL delta at final step: `{_format(checks.get('max_abs_target_nll_delta_final'))}`",
        f"- Max reference NLL delta at final step: `{_format(checks.get('max_abs_reference_nll_delta_final'))}`",
        f"- Final summary: `{final_summary}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.get("checks", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Extracted Edits", ""])
    lines.append("| step | record_id | target vectors | reference vectors | max effective norm | bank saved |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for row in extraction_rows:
        lines.append(
            "| {step} | {record_id} | {target} | {reference} | {norm} | {bank} |".format(
                step=row.get("step"),
                record_id=row.get("record_id"),
                target=row.get("target_activation_count"),
                reference=row.get("reference_activation_count"),
                norm=_format(row.get("max_effective_update_norm_ratio")),
                bank=row.get("bank_saved"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_record_preflight(out_dir: Path, records: List[Dict[str, Any]], alpha0_report: Optional[Dict[str, Any]] = None, alpha_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bank_reports = [report for report in [alpha0_report, alpha_report] if report]
    bank_match_rates = [float(report.get("record_id_match_rate") or 0.0) for report in bank_reports]
    payload = {
        "status": "pass"
        if len(records) == 5
        and all(record.get("id") for record in records)
        and (not bank_reports or all(report.get("status") == "pass" for report in bank_reports))
        else "fail",
        "raw_records_with_record_id": sum(1 for record in records if record.get("id")),
        "record_ids": [record.get("id") for record in records],
        "positional_matching_allowed_by_default": False,
        "record_id_match_rate": _mean(bank_match_rates) if bank_match_rates else 1.0,
        "bank_reports_checked": len(bank_reports),
        "bank_reports": bank_reports,
        "note": "New sequential banks are matched with EngramBank.match_edit_ids_to_records(..., allow_positional_matching=False).",
    }
    _json_dump(out_dir / "record_id_preflight.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"Record-id preflight failed: {payload}")
    return payload


def _make_plots(out_dir: Path, summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(row["step"]) for row in summary_rows]
        for filename, key, ylabel in [
            ("target_nll_by_step.png", "mean_target_nll_increase_edited_so_far", "Mean target NLL increase"),
            ("reference_delta_by_step.png", "mean_reference_delta_abs_all_records", "Mean reference delta"),
            ("edited_vs_future_drift_by_step.png", "future_target_drift", "Future target drift"),
        ]:
            values = [row.get(key) for row in summary_rows]
            plt.figure(figsize=(6, 4))
            plt.plot(steps, [float(value) if value is not None else float("nan") for value in values], marker="o")
            plt.xlabel("step")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=160)
            plt.close()
        return {"status": "pass", "plot_dir": str(plot_dir)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "plot_dir": str(plot_dir)}


def _final_decision(final_summary: Dict[str, Any], rollback_check: Dict[str, Any]) -> str:
    mean_target = final_summary.get("mean_target_nll_increase_edited_so_far")
    mean_ref = final_summary.get("mean_reference_delta_abs_all_records")
    positive = int(final_summary.get("positive_target_edits_so_far") or 0)
    locality = int(final_summary.get("locality_damage_records") or 0)
    rollback_ok = rollback_check.get("status") == "pass"
    synthetic = final_summary.get("synthetic_5edit_2_target_nll_delta_vs_step0")
    if (
        mean_target is not None
        and mean_ref is not None
        and float(mean_target) > 0
        and positive >= 4
        and float(mean_ref) < float(mean_target)
        and locality == 0
        and rollback_ok
        and (synthetic is None or float(synthetic) > 0)
    ):
        return "A. Sequential 5-edit smoke passes. Next gate: small sequential alpha check with alpha 0.025, 0.0375, 0.05."
    if locality > 0 and mean_target is not None and mean_ref is not None and float(mean_ref) >= float(mean_target):
        return "C. Sequential accumulation causes unacceptable locality damage. Do not scale; pivot to Engram-localized replacement/LoRA."
    return "B. Partial sequential signal. Next gate: reduce alpha or run filtered model-known non-empty generation dataset."


def _write_final_report(
    out_dir: Path,
    records: List[Dict[str, Any]],
    selected_modules: List[str],
    alpha0_checks: Dict[str, Any],
    alpha_payload: Dict[str, Any],
    plot_status: Dict[str, Any],
    preflight: Dict[str, Any],
) -> None:
    summary_rows = alpha_payload["sequential_step_summary"]
    final_summary = summary_rows[-1]
    rollback_check = alpha_payload["final_rollback_check"]
    synth_rows = [
        row
        for row in alpha_payload["sequential_step_matrix"]
        if row.get("record_id") == "synthetic-5edit-2"
    ]
    generation_empty = sum(1 for row in alpha_payload["sequential_step_matrix"] if row.get("step") == 5 and row.get("generation_empty"))
    immediate_eos = sum(1 for row in alpha_payload["sequential_step_matrix"] if row.get("step") == 5 and row.get("immediate_eos"))
    decision = _final_decision(final_summary, rollback_check)
    lines = [
        "# Final Sequential 5-Edit Smoke Report",
        "",
        "## Starting Point",
        "",
        "- Best non-sequential config: `token_scope=all`, `module_group=qk_gate_sampled_depths`, `alpha=0.075`.",
        "- Non-sequential result: mean target NLL increase `0.0203598`, mean reference delta `0.00855598`, positive edits `5/5`.",
        "- Sequential alpha uses best alpha / 2: `0.0375`.",
        "",
        "## Data",
        "",
        "- Records: `5` synthetic non-PHI records.",
        f"- Record ids: `{[record.get('id') for record in records]}`",
        "- X_plus / X_minus availability: target and multimodal locality variants were extracted for all 5 edits.",
        f"- Record-id preflight: `{preflight.get('status')}`, match rate `{_format(preflight.get('record_id_match_rate'))}`.",
        "",
        "## Config",
        "",
        "- token_scope: `all`",
        "- module_group: `qk_gate_sampled_depths` with exact successful 8-module lock",
        f"- selected modules: `{selected_modules}`",
        "- alpha: `0.0375`",
        "- update direction: `add`",
        "- sequential_edit: `true`",
        "",
        "## Alpha=0 Gate",
        "",
        f"- Status: `{alpha0_checks.get('status')}`",
        f"- Checks: `{alpha0_checks.get('checks')}`",
        "",
        "## Sequential Step Results",
        "",
        "| step | mean target NLL inc edited | mean ref delta all | positive edited | locality damage | future drift | synthetic-5edit-2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {step} | {target} | {ref} | {pos} | {loc} | {future} | {synth} |".format(
                step=row.get("step"),
                target=_format(row.get("mean_target_nll_increase_edited_so_far")),
                ref=_format(row.get("mean_reference_delta_abs_all_records")),
                pos=row.get("positive_target_edits_so_far"),
                loc=row.get("locality_damage_records"),
                future=_format(row.get("future_target_drift")),
                synth=_format(row.get("synthetic_5edit_2_target_nll_delta_vs_step0")),
            )
        )
    lines.extend(
        [
            "",
            "## Final Step Result",
            "",
            f"- mean_target_nll_increase: `{_format(final_summary.get('mean_target_nll_increase_edited_so_far'))}`",
            f"- mean_reference_delta_abs_all_records: `{_format(final_summary.get('mean_reference_delta_abs_all_records'))}`",
            f"- positive_target_edits: `{final_summary.get('positive_target_edits_so_far')}`",
            f"- locality_damage_records: `{final_summary.get('locality_damage_records')}`",
            f"- record_id_match_rate: `{_format(final_summary.get('record_id_match_rate'))}`",
            f"- bank save status: `{alpha_payload.get('bank_record_id_report', {}).get('status')}`",
            "",
            "## synthetic-5edit-2 Trajectory",
            "",
            "| step | target NLL delta | reference abs delta | generation empty | immediate EOS |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for row in synth_rows:
        lines.append(
            "| {step} | {target} | {ref} | {gen} | {eos} |".format(
                step=row.get("step"),
                target=_format(row.get("target_nll_delta_vs_step0")),
                ref=_format(row.get("reference_delta_abs_vs_step0")),
                gen=row.get("generation_empty"),
                eos=row.get("immediate_eos"),
            )
        )
    lines.extend(
        [
            "",
            "## Generation Diagnostics",
            "",
            f"- Final-step empty generations: `{generation_empty}/5`",
            f"- Final-step immediate EOS count: `{immediate_eos}/5`",
            "- Current evidence remains NLL/logprob-based, not generation-level efficacy.",
            "",
            "## Rollback",
            "",
            f"- Status: `{rollback_check.get('status')}`",
            f"- Final rollback max diff: `{_format(rollback_check.get('max_abs_diff_before_vs_after_final_rollback'))}`",
            "",
            "## Plots",
            "",
            f"- Plot status: `{plot_status.get('status')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
            "No 20-edit run, no replacement mode, and no medical-efficacy claim were made in this task.",
        ]
    )
    (out_dir / "FINAL_SEQUENTIAL_5EDIT_SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict ENGRAM 5-edit sequential smoke gate.")
    parser.add_argument("--alpha0-hparams", required=True)
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="outputs/engram_sequential_5edit_smoke")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.0375)
    parser.add_argument("--alpha0", type=float, default=0.0)
    parser.add_argument("--rollback-tolerance", type=float, default=1e-4)
    parser.add_argument("--metric-tolerance", type=float, default=1e-7)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_git_outputs(out_dir)

    records = _load_records(Path(args.data_file))
    initial_preflight = _write_record_preflight(out_dir, records)
    image_root = Path(args.image_root)
    data_file = Path(args.data_file)

    shutil.copyfile(args.alpha0_hparams, out_dir / "llava_med_5edit_sequential_qkg_sampled_alpha0.used.yaml")
    shutil.copyfile(args.hparams, out_dir / "llava_med_5edit_sequential_qkg_sampled_alpha00375.used.yaml")

    alpha0_hparams = EngramMultimodalHparams.from_hparams(args.alpha0_hparams)
    _configure_hparams(
        alpha0_hparams,
        alpha=args.alpha0,
        bank_dir=out_dir / "alpha0_gate" / "bank",
        image_root=image_root,
        device=args.device,
    )
    alpha0_config_check = _validate_required_hparams(alpha0_hparams, args.alpha0)
    _json_dump(out_dir / "alpha0_gate" / "config_check.json", alpha0_config_check)
    if alpha0_config_check["status"] != "pass":
        raise RuntimeError(f"alpha=0 config check failed: {alpha0_config_check}")

    editor = MultimodalEditor.from_hparams(alpha0_hparams)
    wrapper = editor.model
    wrapper.eval()
    selected_modules = _verify_selected_modules(wrapper, alpha0_hparams, out_dir / "selected_module_metadata.json")

    alpha0_payload = _run_sequence(
        editor=editor,
        hparams=alpha0_hparams,
        records=records,
        data_file=data_file,
        image_root=image_root,
        run_dir=out_dir / "alpha0_gate",
        tag="sequential_alpha0",
        alpha=args.alpha0,
        selected_modules=selected_modules,
        rollback_tolerance=args.rollback_tolerance,
        metric_tolerance=args.metric_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )

    alpha_hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(
        alpha_hparams,
        alpha=args.alpha,
        bank_dir=out_dir / "alpha00375" / "bank",
        image_root=image_root,
        device=args.device,
    )
    alpha_config_check = _validate_required_hparams(alpha_hparams, args.alpha)
    _json_dump(out_dir / "alpha00375" / "config_check.json", alpha_config_check)
    if alpha_config_check["status"] != "pass":
        raise RuntimeError(f"alpha={args.alpha} config check failed: {alpha_config_check}")
    editor.hparams = alpha_hparams

    alpha_payload = _run_sequence(
        editor=editor,
        hparams=alpha_hparams,
        records=records,
        data_file=data_file,
        image_root=image_root,
        run_dir=out_dir / "alpha00375",
        tag="sequential_alpha00375",
        alpha=args.alpha,
        selected_modules=selected_modules,
        rollback_tolerance=args.rollback_tolerance,
        metric_tolerance=args.metric_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )

    preflight = _write_record_preflight(
        out_dir,
        records,
        alpha0_payload["bank_record_id_report"],
        alpha_payload["bank_record_id_report"],
    )
    plot_status = _make_plots(out_dir, alpha_payload["sequential_step_summary"])
    _json_dump(out_dir / "plot_status.json", plot_status)
    _write_final_report(
        out_dir,
        records,
        selected_modules,
        json.loads((out_dir / "alpha0_gate" / "alpha0_checks.json").read_text(encoding="utf-8")),
        alpha_payload,
        plot_status,
        preflight or initial_preflight,
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(out_dir),
                "alpha0_status": json.loads((out_dir / "alpha0_gate" / "alpha0_checks.json").read_text(encoding="utf-8")).get("status"),
                "final_summary": alpha_payload["sequential_step_summary"][-1],
                "rollback_status": alpha_payload["final_rollback_check"].get("status"),
                "plot_status": plot_status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
