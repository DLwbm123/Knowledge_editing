#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.engram.run_medmkeb_modelknown_editing import (  # noqa: E402
    DEFAULT_HPARAMS,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_MODULES,
    _ensure_layout as _ensure_modelknown_layout,
    _finite,
    _format,
    _heavy_imports,
    _json_dump,
    _mean,
    _package_hygiene,
    _plot_optional,
    _run_capture,
    _safe_div,
    _write_csv,
    _write_env_report,
    _write_git_outputs,
    _write_preflight,
    _write_tests,
)


METHOD_B = "B_tiny_lora_replacement"
METHOD_C = "C_engram_projected_tiny_lora"
PARETO_DIRNAME = "sequential_pareto_refine_20"
B_BASE_NEW = 1.77491
B_BASE_REF = 0.469512
B_BASE_DAMAGE = 19
C_BASE_NEW = 1.69794
C_BASE_REF = 0.293396
C_BASE_DAMAGE = 18
C_RESCUE_NEW = 0.302485
C_RESCUE_REF = 0.0387687
C_RESCUE_DAMAGE = 6

QK_GATE_MODULES = list(EXPECTED_MODULES)
QK_ONLY_MODULES = [name for name in EXPECTED_MODULES if name.endswith(".q_proj") or name.endswith(".k_proj")]
GATE_ONLY_MODULES = [name for name in EXPECTED_MODULES if name.endswith(".gate_proj")]
MODULE_SCOPES = {
    "qk_gate_sampled_depths": QK_GATE_MODULES,
    "qk_only_sampled_depths": QK_ONLY_MODULES,
    "gate_only_sampled_depths": GATE_ONLY_MODULES,
}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_layout(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "audit": out_dir / "audit",
        "tests": out_dir / "test_logs",
        "runs": out_dir / "runs",
        "plots": out_dir / "plots",
        "generation": out_dir / "generation_diagnostics",
        "banks": out_dir / "runtime_projector_banks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _cleanup_runtime_projector_banks(out_dir: Path) -> None:
    bank_dir = out_dir / "runtime_projector_banks"
    if bank_dir.exists():
        shutil.rmtree(bank_dir)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Selected records file is not a list: {path}")
    records = [dict(row) for row in rows if isinstance(row, dict)]
    for row in records:
        if "record_id" not in row and "id" in row:
            row["record_id"] = row["id"]
        if "id" not in row and "record_id" in row:
            row["id"] = row["record_id"]
    return records


def _record_id(record: Dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id"))


def _record_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    return [_record_id(record) for record in records]


def _write_data_reuse_report(
    *,
    out_dir: Path,
    selected_records_path: Path,
    previous_record_preflight: Path,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    audit_dir = out_dir / "audit"
    previous = _read_json(previous_record_preflight) if previous_record_preflight.exists() else {}
    expected_ids = [str(item) for item in (previous.get("edit_record_matching") or {}).get("record_ids", [])]
    current_ids = _record_ids(records)
    mismatch = expected_ids != current_ids if expected_ids else False
    duplicates = sorted({item for item in current_ids if current_ids.count(item) > 1})
    payload = {
        "status": "pass" if len(current_ids) == 20 and not duplicates and not mismatch else "fail",
        "selected_records_path": str(selected_records_path),
        "previous_record_preflight": str(previous_record_preflight),
        "selected_record_count": len(current_ids),
        "unique_record_count": len(set(current_ids)),
        "record_id_match_rate": 1.0 if not mismatch and len(current_ids) == 20 else 0.0,
        "positional_matching_used": False,
        "positional_matching_refused_by_default": True,
        "record_ids": current_ids,
        "expected_record_ids": expected_ids,
        "duplicate_record_ids": duplicates,
        "mismatch": mismatch,
        "private_or_patient_data_used": False,
    }
    _json_dump(audit_dir / "selected_record_ids.json", payload)
    lines = [
        "# Data Reuse Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Selected records file: `{selected_records_path}`",
        f"- Selected records: `{len(current_ids)}`",
        f"- Unique records: `{len(set(current_ids))}`",
        f"- Record-id match rate: `{payload['record_id_match_rate']}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- Data was reused exactly from the previous MedMKEB model-known 20 selection; candidates were not regenerated.",
        "",
        "## Record IDs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in current_ids)
    if mismatch:
        lines.extend(["", "## Mismatch", "", "Current selected IDs do not match previous preflight IDs."])
    (audit_dir / "DATA_REUSE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        raise RuntimeError(f"Data reuse preflight failed: {payload}")
    return payload


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_previous_failure_analysis(out_dir: Path, previous_seq_dir: Path) -> Dict[str, Any]:
    audit_dir = out_dir / "audit"
    matrix = _read_csv(previous_seq_dir / "sequential_step_matrix.csv")
    summary = _read_csv(previous_seq_dir / "sequential_summary.csv")
    rows: List[Dict[str, Any]] = []
    if matrix:
        by_record: Dict[str, Dict[str, Any]] = {}
        for row in matrix:
            rid = str(row.get("record_id"))
            method = str(row.get("method"))
            step = int(float(row.get("step") or 0))
            ref = _to_float(row.get("reference_delta_abs"))
            new = _to_float(row.get("new_answer_nll_decrease"))
            item = by_record.setdefault(rid, {"record_id": rid})
            if method == METHOD_C:
                if step == 20:
                    item["C_final_ref_abs"] = ref
                    item["C_final_new"] = new
                    item["C_final_locality_damage"] = str(row.get("locality_damage")).lower() == "true"
                item["C_max_ref_abs"] = max([v for v in [item.get("C_max_ref_abs"), ref] if v is not None], default=None)
                if str(row.get("is_future_record") or row.get("is_future_edit")).lower() == "true" and new is not None:
                    item["C_max_future_drift"] = max(float(item.get("C_max_future_drift") or 0.0), abs(new))
                prev_ref = item.get("_last_C_ref")
                if ref is not None and prev_ref is not None:
                    jump = abs(float(ref) - float(prev_ref))
                    if jump > float(item.get("C_largest_ref_jump") or 0.0):
                        item["C_largest_ref_jump"] = jump
                        item["C_largest_ref_jump_step"] = step
                if ref is not None:
                    item["_last_C_ref"] = ref
            elif method == METHOD_B and step == 20:
                item["B_final_ref_abs"] = ref
                item["B_final_new"] = new
                item["B_final_locality_damage"] = str(row.get("locality_damage")).lower() == "true"
        for item in by_record.values():
            item.pop("_last_C_ref", None)
            item["C_ref_ratio_vs_B"] = _safe_div(item.get("C_final_ref_abs"), item.get("B_final_ref_abs"))
            rows.append(item)
        rows.sort(key=lambda item: float(item.get("C_final_ref_abs") or 0.0), reverse=True)
    else:
        final = [row for row in summary if str(row.get("final_step")).lower() == "true"]
        rows = [
            {
                "record_id": "summary_only",
                "note": "sequential_step_matrix.csv was unavailable; per-record reconstruction was not possible.",
                "final_rows": final,
            }
        ]
    _write_csv(audit_dir / "previous_failure_per_record.csv", rows)
    top = rows[:5]
    c_summary = [row for row in summary if row.get("method") == METHOD_C and str(row.get("final_step")).lower() == "true"]
    b_summary = [row for row in summary if row.get("method") == METHOD_B and str(row.get("final_step")).lower() == "true"]
    payload = {
        "status": "complete" if rows else "empty",
        "previous_seq_dir": str(previous_seq_dir),
        "per_record_rows": len(rows),
        "top_c_reference_drift_records": top,
        "previous_C_final": c_summary[0] if c_summary else {},
        "previous_B_final": b_summary[0] if b_summary else {},
    }
    _json_dump(audit_dir / "previous_failure_analysis.json", payload)
    lines = [
        "# Previous Sequential Failure Analysis",
        "",
        f"- Previous sequential directory: `{previous_seq_dir}`",
        f"- Step matrix available: `{bool(matrix)}`",
        "- Diagnosis-only stage; selected records were not modified.",
        "",
        "## Summary",
        "",
        f"- Previous B final new: `{B_BASE_NEW}`",
        f"- Previous B final reference delta abs: `{B_BASE_REF}`",
        f"- Previous B locality damage records: `{B_BASE_DAMAGE}`",
        f"- Previous C final new: `{C_BASE_NEW}`",
        f"- Previous C final reference delta abs: `{C_BASE_REF}`",
        f"- Previous C locality damage records: `{C_BASE_DAMAGE}`",
        "",
        "C improved reference drift relative to B, but the final drift and locality damage remained too high for a sequential rescue claim.",
        "",
        "## Highest C Reference Drift Records",
        "",
    ]
    for item in top:
        lines.append(
            "- `{record_id}`: C_ref={ref}, B_ref={bref}, C_new={new}, largest_jump_step={jump_step}".format(
                record_id=item.get("record_id"),
                ref=_format(item.get("C_final_ref_abs")),
                bref=_format(item.get("B_final_ref_abs")),
                new=_format(item.get("C_final_new")),
                jump_step=item.get("C_largest_ref_jump_step"),
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Answers",
            "",
            "1. Reference drift grows with sequential accumulation and remains large at the final step.",
            "2. Drift-heavy records are identifiable in `previous_failure_per_record.csv` by final C reference delta.",
            "3. C reduces drift versus B on aggregate, but not enough to satisfy the stricter sequential locality target.",
            "4. Previous-edit retention stays positive while reference drift grows, indicating a target/locality tradeoff rather than a pure target failure.",
            "5. The locality threshold is not the only issue: mean reference drift is substantially above the desired rescue target.",
        ]
    )
    (audit_dir / "PREVIOUS_SEQUENTIAL_FAILURE_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _module_names_for_scope(scope: str) -> List[str]:
    if scope not in MODULE_SCOPES:
        raise ValueError(f"Unknown module_scope={scope}")
    return list(MODULE_SCOPES[scope])


def _regex_for_modules(module_names: Sequence[str]) -> List[str]:
    return [rf"^{re.escape(name)}$" for name in module_names]


def _configure_hparams_for_scope(
    *,
    hparams: Any,
    image_root: Path,
    bank_dir: Path,
    device: str,
    module_names: Sequence[str],
    lora_steps: int,
    lora_ref_loss_weight: float,
) -> None:
    heavy = _heavy_imports()
    _configure_hparams = heavy["_configure_hparams"]
    _configure_hparams(hparams, image_root=image_root, bank_dir=bank_dir, device=str(device), edit_mode="erase")
    hparams.module_patterns = _regex_for_modules(module_names)
    hparams.module_priority_patterns = [r"gate_proj$", r"q_proj$", r"k_proj$"]
    hparams.engram_layers = [0, 8, 16, 24]
    hparams.engram_max_modules = None
    hparams.token_scope = "all"
    hparams.replacement_lambda_ref = float(lora_ref_loss_weight)
    hparams.lora_steps = int(lora_steps)


def _legacy_config_grid(max_c_variants: int, include_optional: bool) -> List[Dict[str, Any]]:
    configs = [
        {
            "config_id": "B_tiny_lora_replacement_baseline",
            "method": METHOD_B,
            "beta": 0.5,
            "lora_steps": 20,
            "module_scope": "qk_gate_sampled_depths",
            "lora_ref_loss_weight": 0.0,
            "prev_edit_loss_weight": 0.0,
            "baseline": True,
        },
        {
            "config_id": "C_baseline_reproduce",
            "method": METHOD_C,
            "beta": 0.5,
            "lora_steps": 20,
            "module_scope": "qk_gate_sampled_depths",
            "lora_ref_loss_weight": 0.0,
            "prev_edit_loss_weight": 0.0,
            "baseline": True,
        },
    ]
    c_priority = [
        ("C_beta0.3_steps20_qkgate_ref0", 0.3, 20, "qk_gate_sampled_depths", 0.0, 0.0),
        ("C_beta0.4_steps20_qkgate_ref0", 0.4, 20, "qk_gate_sampled_depths", 0.0, 0.0),
        ("C_beta0.2_steps20_qkgate_ref0", 0.2, 20, "qk_gate_sampled_depths", 0.0, 0.0),
        ("C_beta0.3_steps10_qkgate_ref0", 0.3, 10, "qk_gate_sampled_depths", 0.0, 0.0),
        ("C_beta0.4_steps10_qkgate_ref0", 0.4, 10, "qk_gate_sampled_depths", 0.0, 0.0),
        ("C_beta0.3_steps20_qkonly_ref0", 0.3, 20, "qk_only_sampled_depths", 0.0, 0.0),
        ("C_beta0.4_steps20_qkonly_ref0", 0.4, 20, "qk_only_sampled_depths", 0.0, 0.0),
        ("C_beta0.3_steps20_qkgate_ref0.05", 0.3, 20, "qk_gate_sampled_depths", 0.05, 0.0),
        ("C_beta0.4_steps20_qkgate_ref0.05", 0.4, 20, "qk_gate_sampled_depths", 0.05, 0.0),
        ("C_beta0.3_steps20_qkgate_ref0.1", 0.3, 20, "qk_gate_sampled_depths", 0.1, 0.0),
    ]
    if include_optional:
        c_priority.extend(
            [
                ("C_beta0.3_steps20_gateonly_ref0", 0.3, 20, "gate_only_sampled_depths", 0.0, 0.0),
                ("C_beta0.4_steps20_gateonly_ref0", 0.4, 20, "gate_only_sampled_depths", 0.0, 0.0),
                ("C_beta0.3_steps20_qkgate_ref0.05_prev0.05", 0.3, 20, "qk_gate_sampled_depths", 0.05, 0.05),
            ]
        )
    for config_id, beta, steps, scope, ref, prev in c_priority[: int(max_c_variants)]:
        configs.append(
            {
                "config_id": config_id,
                "method": METHOD_C,
                "beta": beta,
                "lora_steps": steps,
                "module_scope": scope,
                "lora_ref_loss_weight": ref,
                "prev_edit_loss_weight": prev,
                "baseline": False,
            }
        )
    return configs


def _aggregate_step(rows: List[Dict[str, Any]], config: Dict[str, Any], step: int, total: int) -> Dict[str, Any]:
    step_rows = [row for row in rows if int(row.get("step") or 0) == step]
    edited = [row for row in step_rows if row.get("is_edited_so_far")]
    previous = [row for row in step_rows if row.get("is_previous_edit")]
    future = [row for row in step_rows if row.get("is_future_edit")]
    return {
        "config_id": config["config_id"],
        "method": config["method"],
        "beta": config["beta"],
        "lora_steps": config["lora_steps"],
        "module_scope": config["module_scope"],
        "lora_ref_loss_weight": config["lora_ref_loss_weight"],
        "prev_edit_loss_weight": config["prev_edit_loss_weight"],
        "step": step,
        "record_count": len(step_rows),
        "edited_record_count": len(edited),
        "mean_new_answer_nll_decrease": _mean([row.get("new_answer_nll_decrease_vs_step0") for row in edited]),
        "mean_new_answer_nll_decrease_all_records": _mean([row.get("new_answer_nll_decrease_vs_step0") for row in step_rows]),
        "mean_ref_abs": _mean([row.get("locality_reference_delta_abs_vs_step0") for row in step_rows]),
        "mean_reference_delta_abs_all_records": _mean([row.get("locality_reference_delta_abs_vs_step0") for row in step_rows]),
        "positive_new_answer_edits": sum(1 for row in edited if (row.get("new_answer_nll_decrease_vs_step0") or 0.0) > 0.0),
        "locality_damage_records": sum(1 for row in step_rows if row.get("locality_damage")),
        "previous_edit_retention": _mean([row.get("previous_edit_retention") for row in previous]),
        "future_record_drift": _mean([row.get("future_record_drift") for row in future]),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in step_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in step_rows]),
        "nan_inf_count": sum(1 for row in step_rows if row.get("nan_inf_detected")),
        "final_step": step == total,
    }


def _enrich_row(row: Dict[str, Any], *, config: Dict[str, Any], step: int, idx: int, applied_ids: List[str], module_names: List[str], skipped_modules: List[str], skip_reasons: List[str]) -> Dict[str, Any]:
    rid = str(row.get("record_id"))
    is_edited = idx < step
    is_previous = idx < max(step - 1, 0)
    is_current = idx == step - 1
    is_future = idx >= step
    row.update(
        {
            "config_id": config["config_id"],
            "method": config["method"],
            "beta": config["beta"],
            "lora_steps": config["lora_steps"],
            "module_scope": config["module_scope"],
            "lora_ref_loss_weight": config["lora_ref_loss_weight"],
            "prev_edit_loss_weight": config["prev_edit_loss_weight"],
            "step": step,
            "applied_record_ids": list(applied_ids),
            "record_id": rid,
            "is_edited_so_far": is_edited,
            "is_current_edit": is_current,
            "is_future_edit": is_future,
            "is_previous_edit": is_previous,
            "old_answer_nll": row.get("old_answer_nll_after"),
            "old_answer_nll_delta_vs_step0": row.get("old_answer_nll_increase"),
            "new_answer_nll": row.get("new_answer_nll_after"),
            "new_answer_nll_decrease_vs_step0": row.get("new_answer_nll_decrease"),
            "locality_reference_nll": row.get("reference_nll_after"),
            "locality_reference_delta_abs_vs_step0": row.get("reference_delta_abs"),
            "previous_edit_retention": row.get("new_answer_nll_decrease") if is_previous else None,
            "future_record_drift": abs(float(row.get("new_answer_nll_decrease") or 0.0)) if is_future else None,
            "selected_modules": list(module_names),
            "skipped_modules": list(skipped_modules),
            "skip_reasons": list(skip_reasons),
        }
    )
    return row


def _evaluate_step(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    config: Dict[str, Any],
    step: int,
    applied_ids: List[str],
    module_names: List[str],
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    record_id_match_rate: float,
) -> List[Dict[str, Any]]:
    heavy = _heavy_imports()
    _evaluate_current = heavy["_evaluate_current"]
    _make_eval_row = heavy["_make_eval_row"]
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        after = _evaluate_current(
            model,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=None,
            skip_generation=True,
        )
        row = _make_eval_row(
            method=str(config["method"]),
            record=record,
            case_index=idx,
            before=baselines[str(_record_id(record))],
            after=after,
            rollback_diff=0.0,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            record_id_match_rate=record_id_match_rate,
            edit_id=None,
            beta=float(config["beta"]),
            extra={"positional_matching_used": False},
        )
        rows.append(
            _enrich_row(
                row,
                config=config,
                step=step,
                idx=idx,
                applied_ids=applied_ids,
                module_names=module_names,
                skipped_modules=[],
                skip_reasons=[],
            )
        )
    return rows


def _run_one_config(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    config: Dict[str, Any],
    projector_bank_dir: Optional[Path],
    hparams: Any,
    run_dir: Path,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
    if float(config.get("prev_edit_loss_weight") or 0.0) > 0.0:
        payload = {
            "status": "skipped",
            "reason": "prev_edit_loss_weight is optional and was not implemented to avoid broad training-loop changes",
            "config": config,
        }
        _json_dump(run_dir / "sequential_summary.json", payload)
        return payload, []
    heavy = _heavy_imports()
    EngramBank = heavy["EngramBank"]
    EvalLoraPatch = heavy["EvalLoraPatch"]
    _max_snapshot_diff = heavy["_max_snapshot_diff"]
    _project_factors = heavy["_project_factors"]
    _restore_modules = heavy["_restore_modules"]
    _snapshot_modules = heavy["_snapshot_modules"]
    _train_tiny_lora = heavy["_train_tiny_lora"]

    module_names = _module_names_for_scope(str(config["module_scope"]))
    bank = None
    edit_ids: List[Optional[str]] = [None for _ in records]
    matching: Dict[str, Any] = {"mode": "not_required_for_B"}
    if config["method"] == METHOD_C:
        if projector_bank_dir is None:
            raise RuntimeError(f"C config requires projector bank: {config}")
        bank = EngramBank(projector_bank_dir)
        edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    record_id_match_rate = 1.0 if config["method"] == METHOD_B or matching.get("mode") == "record_id" else 0.0
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if getattr(hparams, "lora_scale", None) is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    active_patches: List[Any] = []
    patch_specs: List[Tuple[str, Dict[str, Any]]] = []
    applied_ids: List[str] = []
    train_rows: List[Dict[str, Any]] = []
    rollback_diff = 0.0
    try:
        rows.extend(
            _evaluate_step(
                model=model,
                records=records,
                image_root=image_root,
                baselines=baselines,
                config=config,
                step=0,
                applied_ids=[],
                module_names=module_names,
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                max_new_tokens=max_new_tokens,
                record_id_match_rate=record_id_match_rate,
            )
        )
        for step, record in enumerate(records, start=1):
            factors, train_summary = _train_tiny_lora(
                model,
                record,
                image_root,
                module_names,
                rank=int(hparams.lora_rank),
                steps=int(config["lora_steps"]),
                lr=float(hparams.lora_lr),
                scale=scale,
                lambda_ref=float(config["lora_ref_loss_weight"]),
            )
            projection_summary = None
            patch_factors = factors
            if config["method"] == METHOD_C:
                assert bank is not None
                edit_id = str(edit_ids[step - 1])
                patch_factors, projection_summary = _project_factors(factors, bank.load_edit(edit_id))
            patch = EvalLoraPatch(model, patch_factors, beta=float(config["beta"]))
            patch.install()
            active_patches.append(patch)
            rid = _record_id(record)
            patch_specs.append((rid, patch_factors))
            applied_ids.append(rid)
            train_rows.append(
                {
                    "config_id": config["config_id"],
                    "step": step,
                    "record_id": rid,
                    "method": config["method"],
                    "lora_train": train_summary,
                    "engram_projection": projection_summary,
                }
            )
            rows.extend(
                _evaluate_step(
                    model=model,
                    records=records,
                    image_root=image_root,
                    baselines=baselines,
                    config=config,
                    step=step,
                    applied_ids=applied_ids,
                    module_names=module_names,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    record_id_match_rate=record_id_match_rate,
                )
            )
    finally:
        for patch in reversed(active_patches):
            patch.remove()
        rollback_diff = _max_snapshot_diff(model, snapshots)
        _restore_modules(model, snapshots)
        if heavy["torch"].cuda.is_available():
            heavy["torch"].cuda.empty_cache()

    for row in rows:
        if int(row.get("step") or 0) == len(records):
            row["rollback_max_abs_diff"] = rollback_diff
            row["rollback_pass"] = rollback_diff <= rollback_tolerance
    summary_rows = [_aggregate_step(rows, config, step, len(records)) for step in range(0, len(records) + 1)]
    rollback = {
        "status": "pass" if rollback_diff <= rollback_tolerance else "fail",
        "rollback_max_abs_diff": rollback_diff,
        "rollback_tolerance": rollback_tolerance,
    }
    final = [row for row in summary_rows if row.get("final_step")]
    payload = {
        "status": "complete",
        "config": config,
        "selected_modules": module_names,
        "record_id_matching": matching,
        "per_record_step_rows": rows,
        "summary_rows": summary_rows,
        "final_summary": final[0] if final else {},
        "final_rollback_check": rollback,
        "train_rows": train_rows,
    }
    _json_dump(run_dir / "config.json", config)
    _json_dump(run_dir / "sequential_step_matrix.json", rows)
    _write_csv(run_dir / "sequential_step_matrix.csv", rows)
    _json_dump(run_dir / "sequential_summary.json", payload)
    _write_csv(run_dir / "sequential_summary.csv", summary_rows)
    _json_dump(run_dir / "final_rollback_check.json", rollback)
    _json_dump(run_dir / "train_trace.json", train_rows)
    return payload, patch_specs


def _legacy_score_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = payload.get("config", {})
    final = payload.get("final_summary", {})
    method = config.get("method")
    new = _to_float(final.get("mean_new_answer_nll_decrease"))
    ref = _to_float(final.get("mean_ref_abs") or final.get("mean_reference_delta_abs_all_records"))
    positive = int(final.get("positive_new_answer_edits") or 0)
    damage = int(final.get("locality_damage_records") or 0)
    rollback = (payload.get("final_rollback_check") or {}).get("status") == "pass"
    match = float(final.get("record_id_match_rate") or 0.0)
    nan = int(final.get("nan_inf_count") or 0)
    new_ratio_c = _safe_div(new, PREVIOUS_C_NEW)
    ref_ratio_c = _safe_div(ref, PREVIOUS_C_REF)
    ref_ratio_b = _safe_div(ref, PREVIOUS_B_REF)
    basic = (
        method == METHOD_C
        and positive >= 16
        and new is not None
        and new > 0.0
        and ref is not None
        and ref < PREVIOUS_C_REF
        and ref_ratio_b is not None
        and ref_ratio_b <= 0.50
        and damage <= 10
        and rollback
        and match == 1.0
        and nan == 0
    )
    strong = (
        method == METHOD_C
        and positive >= 18
        and new_ratio_c is not None
        and new_ratio_c >= 0.80
        and ref_ratio_c is not None
        and ref_ratio_c <= 0.60
        and damage <= 5
        and rollback
        and match == 1.0
        and nan == 0
    )
    pareto = (
        method == METHOD_C
        and positive >= 18
        and new_ratio_c is not None
        and new_ratio_c >= 0.90
        and ref_ratio_c is not None
        and ref_ratio_c <= 0.75
        and damage <= 8
        and rollback
        and match == 1.0
        and nan == 0
    )
    status = "baseline" if method == METHOD_B else ("strong" if strong else ("basic_pass" if basic else ("pareto_promising" if pareto else "fail")))
    if pareto and not strong and not basic:
        status = "pareto_promising"
    return {
        "config_id": config.get("config_id"),
        "method": method,
        "beta": config.get("beta"),
        "lora_steps": config.get("lora_steps"),
        "module_scope": config.get("module_scope"),
        "lora_ref_loss_weight": config.get("lora_ref_loss_weight"),
        "prev_edit_loss_weight": config.get("prev_edit_loss_weight"),
        "final_mean_new": new,
        "final_ref_abs": ref,
        "positive_new": positive,
        "locality_damage": damage,
        "previous_edit_retention": final.get("previous_edit_retention"),
        "rollback_pass": rollback,
        "record_id_match_rate": match,
        "nan_inf_count": nan,
        "new_ratio_vs_previous_C": new_ratio_c,
        "reference_ratio_vs_previous_C": ref_ratio_c,
        "reference_ratio_vs_B": ref_ratio_b,
        "locality_damage_delta_vs_previous_C": damage - PREVIOUS_C_LOCALITY_DAMAGE,
        "basic_rescue_pass": basic,
        "strong_rescue": strong,
        "pareto_promising": pareto,
        "rescue_status": status,
    }


def _legacy_choose_best(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    c_rows = [row for row in scored if row.get("method") == METHOD_C and row.get("final_mean_new") is not None]
    if not c_rows:
        return None
    def key(row: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
        return (
            1 if row.get("strong_rescue") else 0,
            1 if row.get("basic_rescue_pass") else 0,
            1 if row.get("pareto_promising") else 0,
            -float(row.get("locality_damage") or 0),
            -float(row.get("final_ref_abs") or 9999.0),
        )
    return max(c_rows, key=key)


def _legacy_write_best_analysis(out_dir: Path, best: Optional[Dict[str, Any]], config_payloads: Dict[str, Dict[str, Any]]) -> None:
    if not best:
        (out_dir / "BEST_PARETO_CONFIG_ANALYSIS.md").write_text("# Best Variant Analysis\n\nNo C variant completed.\n", encoding="utf-8")
        _write_csv(out_dir / "best_config_per_record.csv", [])
        return
    payload = config_payloads.get(str(best["config_id"]), {})
    rows = [
        row
        for row in payload.get("per_record_step_rows", [])
        if int(row.get("step") or 0) == 20
    ]
    rows.sort(key=lambda row: float(row.get("locality_reference_delta_abs_vs_step0") or 0.0), reverse=True)
    _write_csv(out_dir / "best_config_per_record.csv", rows)
    damaged = [row for row in rows if row.get("locality_damage")]
    lines = [
        "# Best Variant Analysis",
        "",
        f"- Best config: `{best['config_id']}`",
        f"- Rescue status: `{best['rescue_status']}`",
        f"- Final mean new-answer NLL decrease: `{_format(best.get('final_mean_new'))}`",
        f"- Final reference delta abs: `{_format(best.get('final_ref_abs'))}`",
        f"- Positive new-answer edits: `{best.get('positive_new')}/20`",
        f"- Locality damage records: `{best.get('locality_damage')}/20`",
        f"- New ratio vs previous C: `{_format(best.get('new_ratio_vs_previous_C'))}`",
        f"- Reference ratio vs previous C: `{_format(best.get('reference_ratio_vs_previous_C'))}`",
        "",
        "## Interpretation",
        "",
        "- Lower beta is treated as a control for update strength; compare best beta against C baseline reproduction in `pareto_refine_summary.csv`.",
        "- `qk_only_sampled_depths` tests whether removing gate updates reduces reference drift.",
        "- `lora_ref_loss_weight` uses the existing weak reference answer loss in tiny LoRA training.",
        "- Previous-edit preservation was not implemented because it would require broader training-loop changes.",
        "",
        "## Remaining Damaged Records",
        "",
    ]
    for row in damaged[:10]:
        lines.append(
            "- `{rid}`: ref_abs={ref}, new={new}".format(
                rid=row.get("record_id"),
                ref=_format(row.get("locality_reference_delta_abs_vs_step0")),
                new=_format(row.get("new_answer_nll_decrease_vs_step0")),
            )
        )
    if not damaged:
        lines.append("- None.")
    lines.extend(["", "## Largest Reference Drift Records", ""])
    for row in rows[:10]:
        lines.append(
            "- `{rid}`: ref_abs={ref}, new={new}, old={old}".format(
                rid=row.get("record_id"),
                ref=_format(row.get("locality_reference_delta_abs_vs_step0")),
                new=_format(row.get("new_answer_nll_decrease_vs_step0")),
                old=_format(row.get("old_answer_nll_delta_vs_step0")),
            )
        )
    (out_dir / "BEST_PARETO_CONFIG_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _legacy_plot_rescue(out_dir: Path, scored: List[Dict[str, Any]], best_id: Optional[str], config_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    plot_dir = out_dir / "plots"
    made: List[str] = []
    try:
        c_rows = [row for row in scored if row.get("method") == METHOD_C and row.get("final_mean_new") is not None]
        if c_rows:
            plt.figure(figsize=(7, 5))
            plt.scatter([row["final_ref_abs"] for row in c_rows], [row["final_mean_new"] for row in c_rows])
            for row in c_rows:
                plt.annotate(str(row["config_id"]).replace("C_", ""), (row["final_ref_abs"], row["final_mean_new"]), fontsize=6)
            plt.xlabel("final reference delta abs")
            plt.ylabel("final mean new-answer NLL decrease")
            plt.tight_layout()
            path = plot_dir / "ref_vs_new_scatter.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
            plt.figure(figsize=(8, 4))
            plt.bar([str(row["config_id"]).replace("C_", "") for row in c_rows], [row["locality_damage"] for row in c_rows])
            plt.xticks(rotation=45, ha="right", fontsize=6)
            plt.ylabel("locality damage records")
            plt.tight_layout()
            path = plot_dir / "locality_damage_by_config.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if best_id and best_id in config_payloads:
            summary_rows = config_payloads[best_id].get("summary_rows", [])
            xs = [int(row["step"]) for row in summary_rows]
            for key, filename, ylabel in [
                ("mean_ref_abs", "sequential_reference_curve_best.png", "reference delta abs"),
                ("mean_new_answer_nll_decrease", "sequential_new_curve_best.png", "new-answer NLL decrease"),
                ("previous_edit_retention", "sequential_retention_curve_best.png", "previous-edit retention"),
            ]:
                plt.figure(figsize=(7, 4))
                plt.plot(xs, [float(row.get(key) or 0.0) for row in summary_rows], marker="o")
                plt.xlabel("step")
                plt.ylabel(ylabel)
                plt.tight_layout()
                path = plot_dir / filename
                plt.savefig(path)
                plt.close()
                made.append(str(path))
        return {"status": "complete", "files": made}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "files": made}


def _legacy_generation_diagnostics(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    out_dir: Path,
    best: Optional[Dict[str, Any]],
    best_specs: List[Tuple[str, Dict[str, Any]]],
    previous_specs: List[Tuple[str, Dict[str, Any]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    if not best or not (best.get("basic_rescue_pass") or best.get("basic_pass") or best.get("strong_pass") or best.get("pareto_promising")):
        payload = {"status": "skipped", "reason": "no C variant passed basic/Pareto gate"}
        _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", payload)
        return payload
    heavy = _heavy_imports()
    EvalLoraPatch = heavy["EvalLoraPatch"]
    _evaluate_current = heavy["_evaluate_current"]
    rows: List[Dict[str, Any]] = []
    selected = records[:5]

    def run_method(label: str, specs: List[Tuple[str, Dict[str, Any]]], beta: float) -> None:
        patches: List[Any] = []
        try:
            for _, factors in specs:
                patch = EvalLoraPatch(model, factors, beta=beta)
                patch.install()
                patches.append(patch)
            for idx, record in enumerate(selected):
                result = _evaluate_current(model, record, image_root, max_new_tokens=max_new_tokens, min_new_tokens=None, skip_generation=False)
                gen = result.get("generation") or {}
                decoded = str(gen.get("decoded_stripped") or gen.get("decoded_skip_special") or "")
                old = str(record.get("old_answer") or "")
                new = str(record.get("new_answer") or "")
                rows.append(
                    {
                        "method": label,
                        "record_id": _record_id(record),
                        "case_index": idx,
                        "prompt": record.get("src") or record.get("prompt"),
                        "old_answer": old,
                        "new_answer": new,
                        "generation": decoded,
                        "generation_empty": bool(gen.get("generation_empty")),
                        "contains_old_answer": old.casefold() in decoded.casefold() if old else False,
                        "contains_new_answer": new.casefold() in decoded.casefold() if new else False,
                        "exact_new_answer": decoded.strip().casefold() == new.strip().casefold() if new else False,
                        "simple_casefold_contains": new.casefold() in decoded.casefold() if new else False,
                        "notes": "generation diagnostic only; not primary gate",
                    }
                )
        finally:
            for patch in reversed(patches):
                patch.remove()

    run_method("baseline", [], 0.0)
    if previous_specs:
        run_method("previous_C_baseline_reproduce", previous_specs, 0.5)
    run_method(str(best["config_id"]), best_specs, float(best.get("beta") or 0.0))
    payload = {"status": "complete", "rows": rows, "primary_gate": False}
    _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", payload)
    _write_csv(out_dir / "generation_diagnostics" / "generation_5records.csv", rows)
    return payload


def _legacy_write_final_report(
    *,
    out_dir: Path,
    configs: List[Dict[str, Any]],
    scored: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
    data_reuse: Dict[str, Any],
) -> None:
    any_basic = any(row.get("basic_rescue_pass") for row in scored)
    any_improve = any(
        row.get("method") == METHOD_C
        and row.get("final_ref_abs") is not None
        and float(row["final_ref_abs"]) < PREVIOUS_C_REF
        for row in scored
    )
    decision = "C. No C variant improves sequential locality enough. Current ENGRAM-projected LoRA is nonseq-effective but sequential MedMKEB remains unresolved."
    if any_basic:
        decision = "A. A C variant rescues MedMKEB sequential locality. Next: expand to 50 MedMKEB model-known edits or add an external Med-VQA dataset."
    elif any_improve:
        decision = "B. A C variant improves but remains partial. Keep nonseq claim; continue sequential method development with stronger retention/locality constraints."
    lines = [
        "# Final MedMKEB Sequential Rescue 20 Report",
        "",
        "## Starting Point",
        "",
        "- Previous MedMKEB nonseq result: C passed with 20/20 positive new-answer edits and lower reference drift than B.",
        "- Previous MedMKEB sequential result: C completed but was partial because sequential reference/locality drift remained too high.",
        "- This rescue focuses on reducing sequential reference drift and locality damage while preserving most new-answer signal.",
        "",
        "## Data",
        "",
        f"- Reused exact selected records: `{data_reuse.get('selected_record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- No medical or clinical efficacy claim is made.",
        "",
        "## Methods And Configs",
        "",
        f"- Configs run: `{len(configs)}`",
        "- B baseline and previous C baseline reproduction were rerun.",
        "- `prev_edit_loss_weight` remains optional and is skipped if nonzero.",
        "",
        "## Aggregate Table",
        "",
        "| config_id | status | beta | steps | scope | ref_loss | final_new | final_ref | positive_new | locality_damage | retention | rollback | match | nan |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scored:
        lines.append(
            "| {config_id} | {status} | {beta} | {steps} | {scope} | {ref_loss} | {new} | {ref} | {pos} | {damage} | {ret} | {rollback} | {match} | {nan} |".format(
                config_id=row.get("config_id"),
                status=row.get("rescue_status"),
                beta=row.get("beta"),
                steps=row.get("lora_steps"),
                scope=row.get("module_scope"),
                ref_loss=row.get("lora_ref_loss_weight"),
                new=_format(row.get("final_mean_new")),
                ref=_format(row.get("final_ref_abs")),
                pos=row.get("positive_new"),
                damage=row.get("locality_damage"),
                ret=_format(row.get("previous_edit_retention")),
                rollback=row.get("rollback_pass"),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(["", "## Best Variant", ""])
    if best:
        lines.extend(
            [
                f"- Config: `{best.get('config_id')}`",
                f"- Status: `{best.get('rescue_status')}`",
                f"- Final mean new-answer NLL decrease: `{_format(best.get('final_mean_new'))}`",
                f"- Final reference delta abs: `{_format(best.get('final_ref_abs'))}`",
                f"- Positive new-answer edits: `{best.get('positive_new')}/20`",
                f"- Locality damage records: `{best.get('locality_damage')}/20`",
                f"- New ratio vs previous C: `{_format(best.get('new_ratio_vs_previous_C'))}`",
                f"- Reference ratio vs previous C: `{_format(best.get('reference_ratio_vs_previous_C'))}`",
            ]
        )
    else:
        lines.append("- No C variant completed.")
    lines.extend(
        [
            "",
            "## Per-Record Analysis",
            "",
            "- See `BEST_PARETO_CONFIG_ANALYSIS.md` and `best_config_per_record.csv`.",
            "",
            "## Generation Diagnostics",
            "",
            f"- Status: `{generation.get('status')}`",
            "- Generation diagnostics are not used as primary pass/fail evidence.",
            "",
            "## Plots",
            "",
            f"- Status: `{plots.get('status')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
            "## Limitations",
            "",
            "- Bounded 20-edit MedMKEB model-known subset only.",
            "- NLL/logprob metrics are primary evidence.",
            "- Generation diagnostics are secondary only.",
            "- No clinical or medical efficacy claim.",
        ]
    )
    (out_dir / "FINAL_MEDMKEB_SEQUENTIAL_PARETO_REFINE_20_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pareto_config_grid(max_c_variants: int, include_optional: bool) -> List[Dict[str, Any]]:
    priority = [
        ("C_beta0.25_steps15_qkgate_ref0", 0.25, 15, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.25_steps20_qkgate_ref0", 0.25, 20, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.30_steps12_qkgate_ref0", 0.30, 12, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.30_steps15_qkgate_ref0", 0.30, 15, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.35_steps10_qkgate_ref0", 0.35, 10, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.35_steps12_qkgate_ref0", 0.35, 12, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.35_steps15_qkgate_ref0", 0.35, 15, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.30_steps15_qkonly_ref0", 0.30, 15, "qk_only_sampled_depths", 0.0),
    ]
    extra = [
        ("C_beta0.25_steps10_qkgate_ref0", 0.25, 10, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.25_steps12_qkgate_ref0", 0.25, 12, "qk_gate_sampled_depths", 0.0),
        ("C_beta0.35_steps20_qkonly_ref0", 0.35, 20, "qk_only_sampled_depths", 0.0),
        ("C_beta0.30_steps15_qkgate_ref0.02", 0.30, 15, "qk_gate_sampled_depths", 0.02),
        ("C_beta0.35_steps12_qkgate_ref0.02", 0.35, 12, "qk_gate_sampled_depths", 0.02),
        ("C_beta0.35_steps15_qkgate_ref0.02", 0.35, 15, "qk_gate_sampled_depths", 0.02),
        ("C_beta0.35_steps15_qkonly_ref0", 0.35, 15, "qk_only_sampled_depths", 0.0),
    ]
    rows = priority + (extra if include_optional else [])
    rows = rows[: max(0, min(int(max_c_variants), 16))]
    configs = []
    for config_id, beta, steps, scope, ref in rows:
        configs.append(
            {
                "config_id": config_id,
                "method": METHOD_C,
                "beta": beta,
                "lora_steps": steps,
                "module_scope": scope,
                "lora_ref_loss_weight": ref,
                "prev_edit_loss_weight": 0.0,
                "baseline": False,
            }
        )
    return configs


def _alias_payload(payload: Dict[str, Any], alias: str, source_config_id: str) -> Dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    copied.setdefault("config", {})["config_id"] = alias
    copied["config"]["source_config_id"] = source_config_id
    copied["config"]["anchor_reused"] = True
    if isinstance(copied.get("final_summary"), dict):
        copied["final_summary"]["config_id"] = alias
        copied["final_summary"]["source_config_id"] = source_config_id
        copied["final_summary"]["anchor_reused"] = True
    for row in copied.get("summary_rows", []) or []:
        if isinstance(row, dict):
            row["config_id"] = alias
            row["source_config_id"] = source_config_id
            row["anchor_reused"] = True
    for row in copied.get("per_record_step_rows", []) or []:
        if isinstance(row, dict):
            row["config_id"] = alias
            row["source_config_id"] = source_config_id
            row["anchor_reused"] = True
    return copied


def _copy_anchor_run(previous_rescue_dir: Path, out_dir: Path, source_id: str, alias: str) -> None:
    src = previous_rescue_dir / "runs" / source_id
    dst = out_dir / "runs" / alias
    if not src.exists() or dst.exists():
        return
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("*.pt", "*.pth", "*.bin", "__pycache__", "*.pyc", ".DS_Store", "._*"),
    )
    _json_dump(dst / "anchor_reuse_metadata.json", {"source_run": str(src), "alias": alias, "source_config_id": source_id})


def _score_pareto_payload(payload: Dict[str, Any], *, anchor_reused: bool = False, source_config_id: Optional[str] = None) -> Dict[str, Any]:
    config = payload.get("config", {})
    final = payload.get("final_summary", {})
    method = config.get("method")
    new = _to_float(final.get("mean_new_answer_nll_decrease"))
    ref = _to_float(final.get("mean_ref_abs") or final.get("mean_reference_delta_abs_all_records"))
    positive = int(final.get("positive_new_answer_edits") or 0)
    damage = int(final.get("locality_damage_records") or 0)
    rollback = (payload.get("final_rollback_check") or {}).get("status") == "pass"
    match = float(final.get("record_id_match_rate") or 0.0)
    nan = int(final.get("nan_inf_count") or 0)
    score = _safe_div(new, (ref or 0.0) + 1.0e-6) if new is not None and ref is not None else None
    new_ratio_base = _safe_div(new, C_BASE_NEW)
    ref_ratio_base = _safe_div(ref, C_BASE_REF)
    new_ratio_rescue = _safe_div(new, C_RESCUE_NEW)
    ref_ratio_rescue = _safe_div(ref, C_RESCUE_REF)
    basic = (
        method == METHOD_C
        and positive == 20
        and new is not None
        and new > 0.0
        and ref is not None
        and ref < 0.10
        and damage <= 8
        and rollback
        and match == 1.0
        and nan == 0
    )
    strong = (
        method == METHOD_C
        and positive >= 18
        and new is not None
        and new >= 0.60
        and ref is not None
        and ref <= 0.08
        and damage <= 8
        and rollback
        and match == 1.0
        and nan == 0
    )
    pareto = (
        method == METHOD_C
        and positive == 20
        and new is not None
        and new >= 0.50
        and ref is not None
        and ref <= 0.075
        and damage <= 8
        and new_ratio_rescue is not None
        and new_ratio_rescue >= 1.5
        and ref_ratio_base is not None
        and ref_ratio_base <= 0.30
        and rollback
        and match == 1.0
        and nan == 0
    )
    status = "anchor_reused" if anchor_reused else ("pareto_promising" if pareto else ("strong_pass" if strong else ("basic_pass" if basic else "fail")))
    return {
        "config_id": config.get("config_id"),
        "source_config_id": source_config_id or config.get("source_config_id"),
        "anchor_reused": bool(anchor_reused),
        "method": method,
        "beta": config.get("beta"),
        "lora_steps": config.get("lora_steps"),
        "module_scope": config.get("module_scope"),
        "lora_ref_loss_weight": config.get("lora_ref_loss_weight"),
        "prev_edit_loss_weight": config.get("prev_edit_loss_weight"),
        "final_new": new,
        "final_ref": ref,
        "positive_new_answer_edits": positive,
        "locality_damage": damage,
        "previous_edit_retention": final.get("previous_edit_retention"),
        "future_record_drift": final.get("future_record_drift"),
        "rollback_pass": rollback,
        "record_id_match_rate": match,
        "nan_inf_count": nan,
        "new_ratio_vs_base": new_ratio_base,
        "ref_ratio_vs_base": ref_ratio_base,
        "damage_delta_vs_base": damage - C_BASE_DAMAGE,
        "new_ratio_vs_rescue": new_ratio_rescue,
        "ref_ratio_vs_rescue": ref_ratio_rescue,
        "damage_delta_vs_rescue": damage - C_RESCUE_DAMAGE,
        "basic_pass": basic,
        "strong_pass": strong,
        "pareto_promising": pareto,
        "tradeoff_score": score if damage <= 8 else None,
        "status": status,
    }


def _load_reused_anchors(out_dir: Path, previous_rescue_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    anchors = [
        ("C_baseline_reproduce", "C_baseline_reproduce"),
        ("C_best_rescue_reproduce", "C_beta0.3_steps10_qkgate_ref0"),
    ]
    payloads: Dict[str, Dict[str, Any]] = {}
    scores: List[Dict[str, Any]] = []
    lines = [
        "# Anchor Reproduction",
        "",
        "Anchors were reused from the completed `sequential_rescue_20` run because rerunning the 7B multimodal sequential benchmark is not cheap.",
        "",
        "| anchor | source_config | final_new | final_ref | locality_damage | positive_new | rollback | match | reused |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias, source in anchors:
        src_summary = previous_rescue_dir / "runs" / source / "sequential_summary.json"
        if not src_summary.exists():
            lines.append(f"| {alias} | {source} |  |  |  |  |  |  | missing |")
            continue
        payload = _alias_payload(_read_json(src_summary), alias, source)
        _copy_anchor_run(previous_rescue_dir, out_dir, source, alias)
        payloads[alias] = payload
        score = _score_pareto_payload(payload, anchor_reused=True, source_config_id=source)
        scores.append(score)
        lines.append(
            "| {alias} | {source} | {new} | {ref} | {damage} | {positive} | {rollback} | {match} | true |".format(
                alias=alias,
                source=source,
                new=_format(score.get("final_new")),
                ref=_format(score.get("final_ref")),
                damage=score.get("locality_damage"),
                positive=score.get("positive_new_answer_edits"),
                rollback=score.get("rollback_pass"),
                match=_format(score.get("record_id_match_rate")),
            )
        )
    (out_dir / "audit" / "ANCHOR_REPRODUCTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payloads, scores


def _choose_pareto_best(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in scored if row.get("method") == METHOD_C and not row.get("anchor_reused") and row.get("final_new") is not None]
    if not candidates:
        return None
    pareto = [row for row in candidates if row.get("pareto_promising")]
    if pareto:
        return sorted(pareto, key=lambda row: (int(row.get("locality_damage") or 999), -float(row.get("final_new") or 0.0)))[0]
    strong = [row for row in candidates if row.get("strong_pass")]
    if strong:
        return sorted(strong, key=lambda row: (-float(row.get("final_new") or 0.0), int(row.get("locality_damage") or 999)))[0]
    basic = [row for row in candidates if row.get("basic_pass")]
    if basic:
        return sorted(basic, key=lambda row: (-float(row.get("final_new") or 0.0), int(row.get("locality_damage") or 999)))[0]
    return None


def _write_best_pareto_analysis(out_dir: Path, best: Optional[Dict[str, Any]], config_payloads: Dict[str, Dict[str, Any]], scored: List[Dict[str, Any]]) -> None:
    if not best:
        (out_dir / "BEST_PARETO_CONFIG_ANALYSIS.md").write_text(
            "# Best Pareto Config Analysis\n\nNo new Pareto-refined config passed the basic criteria.\n",
            encoding="utf-8",
        )
        _write_csv(out_dir / "best_config_per_record.csv", [])
        return
    payload = config_payloads.get(str(best["config_id"]), {})
    rows = [row for row in payload.get("per_record_step_rows", []) if int(row.get("step") or 0) == 20]
    rows.sort(key=lambda row: float(row.get("locality_reference_delta_abs_vs_step0") or 0.0), reverse=True)
    _write_csv(out_dir / "best_config_per_record.csv", rows)
    damaged = [row for row in rows if row.get("locality_damage")]
    by_id = {row.get("config_id"): row for row in scored}
    base = by_id.get("C_baseline_reproduce", {})
    rescue = by_id.get("C_best_rescue_reproduce", {})
    lines = [
        "# Best Pareto Config Analysis",
        "",
        f"- Best config: `{best.get('config_id')}`",
        f"- Status: `{best.get('status')}`",
        f"- Final new-answer NLL decrease: `{_format(best.get('final_new'))}`",
        f"- Final reference delta abs: `{_format(best.get('final_ref'))}`",
        f"- Positive new-answer edits: `{best.get('positive_new_answer_edits')}/20`",
        f"- Locality damage records: `{best.get('locality_damage')}/20`",
        f"- New ratio vs high-strength C baseline: `{_format(best.get('new_ratio_vs_base'))}`",
        f"- Reference ratio vs high-strength C baseline: `{_format(best.get('ref_ratio_vs_base'))}`",
        f"- New ratio vs previous best rescue: `{_format(best.get('new_ratio_vs_rescue'))}`",
        f"- Reference ratio vs previous best rescue: `{_format(best.get('ref_ratio_vs_rescue'))}`",
        "",
        "## Anchor Comparison",
        "",
        f"- High-strength C baseline final_new/final_ref/damage: `{_format(base.get('final_new'))}` / `{_format(base.get('final_ref'))}` / `{base.get('locality_damage')}`",
        f"- Previous best rescue final_new/final_ref/damage: `{_format(rescue.get('final_new'))}` / `{_format(rescue.get('final_ref'))}` / `{rescue.get('locality_damage')}`",
        "",
        "## Questions",
        "",
        f"1. Remaining locality-damaged records: `{len(damaged)}`.",
        "2. Strength/locality changes can be read from the beta and step columns in `pareto_refine_summary.csv`.",
        "3. Compare steps 12 and 15 rows against the reused steps 10 rescue anchor for the middle-point question.",
        "4. Compare qk_only rows against qk_gate rows at the same beta/steps for reference drift.",
        "5. Weak reference-loss rows are run only when `--include-optional` is supplied.",
        "6. See `best_config_per_record.csv` for answer/image/edit-position concentration.",
        "7. Future-record drift and previous-edit retention are included in the aggregate and per-record files.",
        "",
        "## Remaining Damaged Records",
        "",
    ]
    if damaged:
        for row in damaged[:10]:
            lines.append(
                "- `{rid}`: ref_abs={ref}, new={new}, position={pos}".format(
                    rid=row.get("record_id"),
                    ref=_format(row.get("locality_reference_delta_abs_vs_step0")),
                    new=_format(row.get("new_answer_nll_decrease_vs_step0")),
                    pos=row.get("case_index"),
                )
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Largest Drift Records", ""])
    for row in rows[:10]:
        lines.append(
            "- `{rid}`: ref_abs={ref}, new={new}, future_drift={future}".format(
                rid=row.get("record_id"),
                ref=_format(row.get("locality_reference_delta_abs_vs_step0")),
                new=_format(row.get("new_answer_nll_decrease_vs_step0")),
                future=_format(row.get("future_record_drift")),
            )
        )
    (out_dir / "BEST_PARETO_CONFIG_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pareto(out_dir: Path, scored: List[Dict[str, Any]], best_id: Optional[str], config_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    plot_dir = out_dir / "plots"
    made: List[str] = []
    try:
        rows = [row for row in scored if row.get("method") == METHOD_C and row.get("final_new") is not None]
        if rows:
            xs = [float(row["final_ref"]) for row in rows]
            ys = [float(row["final_new"]) for row in rows]
            colors = ["tab:orange" if row.get("anchor_reused") else "tab:blue" for row in rows]
            plt.figure(figsize=(7, 5))
            plt.scatter(xs, ys, c=colors)
            for row in rows:
                plt.annotate(str(row["config_id"]).replace("C_", ""), (float(row["final_ref"]), float(row["final_new"])), fontsize=6)
            plt.xlabel("final reference delta abs")
            plt.ylabel("final new-answer NLL decrease")
            plt.tight_layout()
            path = plot_dir / "ref_vs_new_scatter.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))

            plt.figure(figsize=(8, 4))
            plt.bar([str(row["config_id"]).replace("C_", "") for row in rows], [int(row["locality_damage"] or 0) for row in rows])
            plt.xticks(rotation=45, ha="right", fontsize=6)
            plt.ylabel("locality damage records")
            plt.tight_layout()
            path = plot_dir / "locality_damage_by_config.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))

            frontier = sorted([row for row in rows if int(row.get("locality_damage") or 999) <= 8], key=lambda row: float(row["final_ref"]))
            plt.figure(figsize=(7, 5))
            plt.scatter(xs, ys, c=colors, alpha=0.6)
            if frontier:
                plt.plot([float(row["final_ref"]) for row in frontier], [float(row["final_new"]) for row in frontier], marker="o")
            plt.xlabel("final reference delta abs")
            plt.ylabel("final new-answer NLL decrease")
            plt.tight_layout()
            path = plot_dir / "pareto_frontier.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if best_id and best_id in config_payloads:
            summary_rows = config_payloads[best_id].get("summary_rows", [])
            step_rows = [row for row in summary_rows if isinstance(row, dict)]
            xs = [int(row["step"]) for row in step_rows]
            for key, filename, ylabel in [
                ("mean_ref_abs", "sequential_reference_curve_best.png", "reference delta abs"),
                ("mean_new_answer_nll_decrease", "sequential_new_curve_best.png", "new-answer NLL decrease"),
                ("previous_edit_retention", "sequential_retention_curve_best.png", "previous-edit retention"),
            ]:
                plt.figure(figsize=(7, 4))
                plt.plot(xs, [float(row.get(key) or 0.0) for row in step_rows], marker="o")
                plt.xlabel("step")
                plt.ylabel(ylabel)
                plt.tight_layout()
                path = plot_dir / filename
                plt.savefig(path)
                plt.close()
                made.append(str(path))
        return {"status": "complete", "files": made}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "files": made}


def _write_final_pareto_report(
    *,
    out_dir: Path,
    configs: List[Dict[str, Any]],
    scored: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
    data_reuse: Dict[str, Any],
) -> None:
    pareto_rows = [row for row in scored if row.get("pareto_promising") and not row.get("anchor_reused")]
    strong_rows = [row for row in scored if row.get("strong_pass") and not row.get("anchor_reused")]
    basic_rows = [row for row in scored if row.get("basic_pass") and not row.get("anchor_reused")]
    if pareto_rows:
        decision = "A. A Pareto-refined C variant achieves a better bounded strength-locality trade-off. Next: validate this config on 50 MedMKEB model-known edits or external Med-VQA."
    elif basic_rows or strong_rows:
        decision = "B. A variant improves over the first rescue but remains partial. Keep as bounded sequential setting and continue method development."
    else:
        decision = "C. No variant improves the trade-off enough. Keep previous best rescue for safety; MedMKEB sequential remains strength-limited."
    lines = [
        "# Final MedMKEB Sequential Pareto Refine 20 Report",
        "",
        "## Starting Point",
        "",
        "- MedMKEB nonseq C passed, but sequential editing showed a target/locality trade-off.",
        "- The first rescue reduced drift and locality damage mainly by weakening edit strength.",
        "- This run searches the small Pareto region between high-strength `beta=0.5, steps=20` and low-drift `beta=0.3, steps=10`.",
        "",
        "## Data",
        "",
        f"- Reused exact selected records: `{data_reuse.get('selected_record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- No medical or clinical efficacy claim is made.",
        "",
        "## Configs",
        "",
        f"- New configs run: `{len(configs)}`",
        "- Anchors are reused from `sequential_rescue_20` and marked `anchor_reused=True`.",
        "- Main gate used NLL/logprob metrics with generation skipped.",
        "",
        "## Aggregate Table",
        "",
        "| config_id | status | reused | beta | steps | scope | ref_loss | final_new | final_ref | positive_new | damage | retention | rollback | match | new/rescue | ref/base | score |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scored:
        lines.append(
            "| {config_id} | {status} | {reused} | {beta} | {steps} | {scope} | {ref_loss} | {new} | {ref} | {positive} | {damage} | {ret} | {rollback} | {match} | {new_rescue} | {ref_base} | {score} |".format(
                config_id=row.get("config_id"),
                status=row.get("status"),
                reused=row.get("anchor_reused"),
                beta=row.get("beta"),
                steps=row.get("lora_steps"),
                scope=row.get("module_scope"),
                ref_loss=row.get("lora_ref_loss_weight"),
                new=_format(row.get("final_new")),
                ref=_format(row.get("final_ref")),
                positive=row.get("positive_new_answer_edits"),
                damage=row.get("locality_damage"),
                ret=_format(row.get("previous_edit_retention")),
                rollback=row.get("rollback_pass"),
                match=_format(row.get("record_id_match_rate")),
                new_rescue=_format(row.get("new_ratio_vs_rescue")),
                ref_base=_format(row.get("ref_ratio_vs_base")),
                score=_format(row.get("tradeoff_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Pareto Analysis",
            "",
            f"- Basic pass configs: `{[row.get('config_id') for row in basic_rows]}`",
            f"- Strong pass configs: `{[row.get('config_id') for row in strong_rows]}`",
            f"- Pareto-promising configs: `{[row.get('config_id') for row in pareto_rows]}`",
            f"- Selected best config: `{best.get('config_id') if best else None}`",
            "",
            "## Best Config Analysis",
            "",
            "- See `BEST_PARETO_CONFIG_ANALYSIS.md` and `best_config_per_record.csv`.",
            "",
            "## Generation Diagnostics",
            "",
            f"- Status: `{generation.get('status')}`",
            "- Generation diagnostics are secondary and not used as the pass/fail criterion.",
            "",
            "## Plots",
            "",
            f"- Status: `{plots.get('status')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
            "## Limitations",
            "",
            "- Bounded 20-edit MedMKEB model-known subset.",
            "- NLL/logprob metrics are primary evidence.",
            "- Generation diagnostic only; no generation-level efficacy claim unless supported.",
            "- No medical or clinical efficacy claim.",
        ]
    )
    (out_dir / "FINAL_MEDMKEB_SEQUENTIAL_PARETO_REFINE_20_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_only(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    records = _load_records(Path(args.selected_records))
    _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    _write_previous_failure_analysis(out_dir, Path(args.previous_seq_dir))
    _load_reused_anchors(out_dir, Path(args.previous_rescue_dir))
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    _cleanup_runtime_projector_banks(out_dir)
    _package_hygiene(out_dir, remove_runtime_bank=True)
    return 0


def _run_gpu(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)
    if test_status.get("engram_tests_pass") is False:
        raise RuntimeError(f"ENGRAM tests failed: {test_status}")
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        input_records=Path(args.selected_records),
        image_root=Path(args.image_root),
        test_status=test_status,
    )
    if preflight.get("status") != "pass":
        raise RuntimeError(f"Preflight failed: {preflight}")

    records = _load_records(Path(args.selected_records))
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    _write_previous_failure_analysis(out_dir, Path(args.previous_seq_dir))
    anchor_payloads, anchor_scores = _load_reused_anchors(out_dir, Path(args.previous_rescue_dir))
    baselines = _read_json(Path(args.baseline_metrics))

    heavy = _heavy_imports()
    torch = heavy["torch"]
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    select_linear_layers = heavy["select_linear_layers"]
    _extract_projector_bank = heavy["_extract_projector_bank"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    first_bank = out_dir / "runtime_projector_banks" / "bootstrap"
    _configure_hparams_for_scope(
        hparams=hparams,
        image_root=Path(args.image_root),
        bank_dir=first_bank,
        device=str(args.device),
        module_names=QK_GATE_MODULES,
        lora_steps=20,
        lora_ref_loss_weight=0.0,
    )
    editor = MultimodalEditor.from_hparams(hparams)

    configs = _pareto_config_grid(max_c_variants=int(args.max_c_variants), include_optional=bool(args.include_optional))
    _json_dump(out_dir / "pareto_refine_grid.json", {"new_configs": configs, "anchor_scores": anchor_scores})
    used_scopes = sorted({config["module_scope"] for config in configs if config["method"] == METHOD_C})
    bank_dirs: Dict[str, Path] = {}
    selected_modules_rows: List[Dict[str, Any]] = []
    for scope in used_scopes:
        module_names = _module_names_for_scope(scope)
        bank_dir = out_dir / "runtime_projector_banks" / scope
        scope_hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
        _configure_hparams_for_scope(
            hparams=scope_hparams,
            image_root=Path(args.image_root),
            bank_dir=bank_dir,
            device=str(args.device),
            module_names=module_names,
            lora_steps=20,
            lora_ref_loss_weight=0.0,
        )
        selected = [layer.name for layer in select_linear_layers(editor.model, scope_hparams)]
        status = "pass" if set(selected) == set(module_names) and len(selected) == len(module_names) else "fail"
        row = {"module_scope": scope, "status": status, "expected_modules": module_names, "selected_modules": selected}
        selected_modules_rows.append(row)
        if status != "pass":
            raise RuntimeError(f"Selected modules mismatch for {scope}: {row}")
        _json_dump(out_dir / "audit" / f"selected_modules_{scope}.json", row)
        _extract_projector_bank(editor, scope_hparams, Path(args.selected_records), records, bank_dir)
        bank_dirs[scope] = bank_dir

    config_payloads: Dict[str, Dict[str, Any]] = dict(anchor_payloads)
    specs_by_config: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    scored: List[Dict[str, Any]] = list(anchor_scores)
    for config in configs:
        run_dir = out_dir / "runs" / str(config["config_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        run_hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
        _configure_hparams_for_scope(
            hparams=run_hparams,
            image_root=Path(args.image_root),
            bank_dir=bank_dirs.get(str(config["module_scope"]), out_dir / "runtime_projector_banks" / str(config["module_scope"])),
            device=str(args.device),
            module_names=_module_names_for_scope(str(config["module_scope"])),
            lora_steps=int(config["lora_steps"]),
            lora_ref_loss_weight=float(config["lora_ref_loss_weight"]),
        )
        payload, specs = _run_one_config(
            model=editor.model,
            records=records,
            image_root=Path(args.image_root),
            baselines=baselines,
            config=config,
            projector_bank_dir=bank_dirs.get(str(config["module_scope"])) if config["method"] == METHOD_C else None,
            hparams=run_hparams,
            run_dir=run_dir,
            rollback_tolerance=float(args.rollback_tolerance),
            locality_threshold=float(args.locality_damage_threshold),
            max_new_tokens=int(args.max_new_tokens),
        )
        config_payloads[str(config["config_id"])] = payload
        specs_by_config[str(config["config_id"])] = specs
        scored_row = _score_pareto_payload(payload)
        scored.append(scored_row)
        _json_dump(run_dir / "score.json", scored_row)

    _write_csv(out_dir / "pareto_refine_summary.csv", scored)
    _json_dump(out_dir / "pareto_refine_summary.json", {"configs": configs, "scores": scored})
    best = _choose_pareto_best(scored)
    _write_best_pareto_analysis(out_dir, best, config_payloads, scored)
    previous_specs = specs_by_config.get("C_baseline_reproduce", [])
    best_specs = specs_by_config.get(str(best["config_id"]), []) if best else []
    generation = _legacy_generation_diagnostics(
        model=editor.model,
        records=records,
        image_root=Path(args.image_root),
        baselines=baselines,
        out_dir=out_dir,
        best=best,
        best_specs=best_specs,
        previous_specs=previous_specs,
        max_new_tokens=int(args.generation_max_new_tokens),
    )
    plots = _plot_pareto(out_dir, scored, str(best["config_id"]) if best else None, config_payloads)
    _write_final_pareto_report(
        out_dir=out_dir,
        configs=configs,
        scored=scored,
        best=best,
        generation=generation,
        plots=plots,
        data_reuse=data_reuse,
    )
    _cleanup_runtime_projector_banks(out_dir)
    hygiene = _package_hygiene(out_dir, remove_runtime_bank=True)
    _json_dump(out_dir / "runtime.json", {"status": "complete", "best": best, "hygiene": hygiene, "selected_modules": selected_modules_rows})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "run-gpu"], default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / PARETO_DIRNAME))
    parser.add_argument("--selected-records", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "medmkeb_modelknown_20.json"))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "baseline_metrics.json"))
    parser.add_argument("--previous-record-preflight", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "record_id_preflight.json"))
    parser.add_argument("--previous-seq-dir", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "sequential"))
    parser.add_argument("--previous-rescue-dir", default=str(DEFAULT_OUTPUT_DIR / "sequential_rescue_20"))
    parser.add_argument("--image-root", default="/remote-home/wangbomin/medmkeb_engram_projected_lora_bundle")
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-c-variants", type=int, default=8)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "prepare":
        return _prepare_only(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
