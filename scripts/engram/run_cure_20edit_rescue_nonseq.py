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
from easyeditor.models.engram.crisp_projection import apply_crisp_projection_to_delta  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_cure_20edit_modelknown import (  # noqa: E402
    _run_pytest,
    _write_env_report,
)
from scripts.engram.run_cure_mededit_5edit import (  # noqa: E402
    EvalMixedDeltaPatch,
    _collect_reference_crisp_cache,
    _dense_delta_from_factor,
    _projection_caches_for_thresholds_from_kfac,
)
from scripts.engram.run_localized_replacement_5edit import (  # noqa: E402
    EXPECTED_MODULES,
    EvalLoraPatch,
    _configure_hparams,
    _evaluate_current,
    _finite,
    _format,
    _json_dump,
    _low_rank_norm,
    _make_eval_row,
    _max_snapshot_diff,
    _mean,
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
METHOD_E = "E_cure_dual_projected_tiny_lora"
GATE_POLICY = "engram_fallback"
CRISP_CACHE_POLICY = "record_local_reference_cache_nonseq"
DIAGNOSTIC_GAMMAS = [0.1, 0.2, 0.3, 0.5]

PRIORITY_CURE_CONFIGS: List[Dict[str, Any]] = [
    {"beta": 0.5, "gamma": 0.5, "lambda": 0.0, "clamp": False},
    {"beta": 0.5, "gamma": 0.5, "lambda": 0.1, "clamp": True},
    {"beta": 0.5, "gamma": 0.5, "lambda": 0.25, "clamp": True},
    {"beta": 0.5, "gamma": 0.5, "lambda": 0.5, "clamp": True},
    {"beta": 0.5, "gamma": 0.3, "lambda": 0.25, "clamp": True},
    {"beta": 0.5, "gamma": 0.2, "lambda": 0.25, "clamp": True},
    {"beta": 0.45, "gamma": 0.5, "lambda": 0.25, "clamp": True},
    {"beta": 0.4, "gamma": 0.5, "lambda": 0.25, "clamp": True},
    {"beta": 0.35, "gamma": 0.5, "lambda": 0.25, "clamp": True},
]


def _run_capture(command: List[str], cwd: Path = PROJECT_ROOT) -> str:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _write_stop_report(out_dir: Path, reason: str, payload: Dict[str, Any]) -> None:
    lines = [
        "# Final CURE 20-Edit Rescue Nonseq Report",
        "",
        "- Status: `stopped`",
        f"- Stop reason: `{reason}`",
        "- No sequential edit was launched.",
        "- No metric was fabricated.",
        "",
        "## Details",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True)[:12000],
        "```",
        "",
        "## Limitations",
        "",
        "- Non-PHI synthetic 20-edit engineering gate only.",
        "- No medical or clinical efficacy claim.",
        "- Delta-space Crisp projection is not original CrispEdit gradient-projected training.",
    ]
    (out_dir / "FINAL_CURE_20EDIT_RESCUE_NONSEQ_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_id(config: Dict[str, Any]) -> str:
    if config.get("method") == METHOD_C:
        return f"C_beta{config['beta']}"
    clamp = "clamptrue" if config.get("clamp") else "clampfalse"
    return f"E_beta{config['beta']}_gamma{config['gamma']}_lambda{config['lambda']}_{clamp}"


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


def _yaml_value(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    return None


def _write_preflight(
    out_dir: Path,
    *,
    hparams_path: Path,
    source_output_dir: Path,
    source_data_path: Path,
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
        "source_data_file_exists": source_data_path.exists(),
        "image_root_ready": image_root.exists(),
        "projector_bank_dir_exists": projector_bank_dir.exists(),
        "projector_bank_index_exists": (projector_bank_dir / "index.json").exists(),
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
            "source_data_file": str(source_data_path),
            "image_root": str(image_root),
            "projector_bank_dir": str(projector_bank_dir),
            "output_dir": str(out_dir),
        },
    }
    lines = [
        "# CURE 20-Edit Rescue Nonseq Preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Python: `{sys.executable}`",
        "- Main gate: non-sequential only; generation skipped when `--skip-generation` is used.",
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


def _materialize_image_root(src: Path, dst: Path) -> str:
    if dst.exists() and not dst.is_symlink() and any(dst.iterdir()):
        return "reuse_existing_nonempty_dir"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return "symlink_to_source_images"
    except OSError:
        shutil.copytree(src, dst, ignore=lambda _src, names: [name for name in names if name.startswith("._") or name == ".DS_Store"])
        return "copy_without_appledouble"


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Expected list records at {path}, got {type(records)}")
    return records


def _reuse_selected_data(source_output_dir: Path, out_dir: Path) -> Tuple[List[Dict[str, Any]], Path, Path, Dict[str, Any]]:
    source_data = source_output_dir / "synthetic_root" / "data" / "medmkeb" / "raw" / "engram_replacement_20edit_modelknown.json"
    source_images = source_output_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    records = _load_records(source_data)
    raw_dir = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw"
    image_root = out_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_path = raw_dir / "engram_replacement_20edit_rescue_nonseq.json"
    data_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    image_materialization = _materialize_image_root(source_images, image_root)

    source_summary_path = source_output_dir / "replacement_data_summary.json"
    source_filter_path = source_output_dir / "data_filter_report.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    source_filter = json.loads(source_filter_path.read_text(encoding="utf-8")) if source_filter_path.exists() else {}
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
        else "fail",
        "source_output_dir": str(source_output_dir),
        "source_data_file": str(source_data),
        "source_replacement_summary": str(source_summary_path),
        "source_data_filter_report": str(source_filter_path),
        "reused_data_file": str(data_path),
        "image_root": str(image_root),
        "image_materialization": image_materialization,
        "record_count": len(records),
        "selected_record_ids": ids,
        "matches_source_replacement_summary_ids": ids == summary_ids,
        "matches_source_data_filter_selected_ids": (ids == filter_ids) if filter_ids else None,
        "record_id_match_rate": 1.0 if len(records) == 20 and ids == summary_ids else 0.0,
        "positional_matching_used": False,
        "private_or_patient_data_used": False,
        "original_data_modified": False,
        "image_rows": image_rows,
    }
    _json_dump(out_dir / "data_reuse_report.json", report)
    return records, data_path, image_root, report


def _load_or_eval_baselines(
    *,
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    source_output_dir: Path,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    source_path = source_output_dir / "baseline_metrics.json"
    baselines: Dict[str, Dict[str, Any]] = {}
    source_used = False
    if source_path.exists():
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and all(str(record["id"]) in loaded for record in records):
            baselines = {str(record["id"]): loaded[str(record["id"])] for record in records}
            source_used = True
    if not baselines:
        for record in records:
            baselines[str(record["id"])] = _evaluate_current(
                model,
                record,
                image_root,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
            )
    report = {
        "source_path": str(source_path),
        "source_baseline_metrics_used": source_used,
        "record_count": len(baselines),
        "record_ids": list(baselines.keys()),
    }
    return baselines, report


def _match_bank_records(records: List[Dict[str, Any]], projector_bank_dir: Path, out_dir: Path) -> Tuple[List[str], Dict[str, Any]]:
    bank = EngramBank(projector_bank_dir)
    try:
        edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
        payload = {
            "status": "pass" if matching.get("mode") == "record_id" else "fail",
            "record_id_match_rate": 1.0 if matching.get("mode") == "record_id" else 0.0,
            "positional_matching_used": False,
            "positional_matching_refused_by_default": True,
            "edit_ids": edit_ids,
            "bank_matching": matching,
        }
    except Exception as exc:
        payload = {
            "status": "fail",
            "record_id_match_rate": 0.0,
            "positional_matching_used": False,
            "positional_matching_refused_by_default": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        edit_ids = []
    _json_dump(out_dir / "record_id_preflight.json", payload)
    return edit_ids, payload


def _factor_norm(factor: Dict[str, torch.Tensor | float]) -> float:
    return _low_rank_norm(
        factor["A"].detach().cpu().float(),  # type: ignore[union-attr]
        factor["B"].detach().cpu().float(),  # type: ignore[union-attr]
        float(factor.get("scale", 1.0)),  # type: ignore[union-attr]
    )


def _norm_total(values: Iterable[Optional[float]]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None
    return math.sqrt(sum(value * value for value in finite_values))


def _mask_counts(projection_cache: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    mask = projection_cache.get("M")
    if isinstance(mask, torch.Tensor):
        active = int(mask.detach().cpu().bool().sum().item())
        total = int(mask.numel())
        return active, total, _safe_div(active, total)
    metadata = projection_cache.get("metadata") or {}
    shape = metadata.get("mask_shape")
    keep_ratio = metadata.get("keep_ratio")
    if isinstance(shape, list) and len(shape) == 2 and keep_ratio is not None:
        total = int(shape[0]) * int(shape[1])
        active = int(round(float(keep_ratio) * total))
        return active, total, _safe_div(active, total)
    return None, None, None


def _lora_entry(factor: Dict[str, torch.Tensor | float]) -> Dict[str, Any]:
    return {
        "kind": "lora",
        "A": factor["A"].detach().cpu().float(),  # type: ignore[union-attr]
        "B": factor["B"].detach().cpu().float(),  # type: ignore[union-attr]
        "scale": float(factor.get("scale", 1.0)),  # type: ignore[union-attr]
    }


def _build_rescue_entries(
    *,
    candidate_factors: Dict[str, Dict[str, torch.Tensor | float]],
    engram_factors: Dict[str, Dict[str, torch.Tensor | float]],
    projection_caches: Dict[str, Dict[str, Any]],
    lambda_mix: float,
    clamp: bool,
    clamp_ratio: float,
    gate_policy: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    module_rows: List[Dict[str, Any]] = []
    lambda_is_zero = abs(float(lambda_mix)) <= 1.0e-12
    for name, engram_factor in engram_factors.items():
        candidate_norm = _factor_norm(candidate_factors[name])
        engram_norm = _factor_norm(engram_factor)
        cache = projection_caches.get(name)
        if lambda_is_zero:
            entries[name] = _lora_entry(engram_factor)
            active, total, keep_ratio = _mask_counts(cache) if cache is not None else (None, None, None)
            module_rows.append(
                {
                    "module_name": name,
                    "crisp_projected": bool(cache is not None),
                    "lambda_zero_lora_fast_path": True,
                    "engram_fallback": bool(cache is None),
                    "fallback_reason": "missing_projection_cache" if cache is None else None,
                    "candidate_delta_norm": candidate_norm,
                    "delta_candidate_norm": candidate_norm,
                    "delta_engram_norm": engram_norm,
                    "delta_crisp_norm": None,
                    "delta_preclamp_norm": engram_norm,
                    "delta_postclamp_norm": engram_norm,
                    "projection_norm_ratio": None,
                    "preclamp_to_engram_ratio": 1.0 if engram_norm != 0.0 else None,
                    "postclamp_to_engram_ratio": 1.0 if engram_norm != 0.0 else None,
                    "clamp_applied": False,
                    "mask_keep_active": active,
                    "mask_keep_total": total,
                    "mask_keep_ratio": keep_ratio,
                }
            )
            continue
        if cache is None:
            if gate_policy != GATE_POLICY:
                raise RuntimeError(f"No projection cache for {name} and unsupported gate_policy={gate_policy}")
            entries[name] = _lora_entry(engram_factor)
            module_rows.append(
                {
                    "module_name": name,
                    "crisp_projected": False,
                    "engram_fallback": True,
                    "fallback_reason": "missing_projection_cache",
                    "candidate_delta_norm": candidate_norm,
                    "delta_candidate_norm": candidate_norm,
                    "delta_engram_norm": engram_norm,
                    "delta_crisp_norm": engram_norm,
                    "delta_preclamp_norm": engram_norm,
                    "delta_postclamp_norm": engram_norm,
                    "projection_norm_ratio": 1.0 if engram_norm != 0.0 else None,
                    "preclamp_to_engram_ratio": 1.0 if engram_norm != 0.0 else None,
                    "postclamp_to_engram_ratio": 1.0 if engram_norm != 0.0 else None,
                    "clamp_applied": False,
                    "mask_keep_active": None,
                    "mask_keep_total": None,
                    "mask_keep_ratio": None,
                }
            )
            continue

        delta_engram = _dense_delta_from_factor(engram_factor)
        delta_crisp = apply_crisp_projection_to_delta(delta_engram, cache).detach().cpu().float()
        preclamp = ((1.0 - float(lambda_mix)) * delta_engram + float(lambda_mix) * delta_crisp).detach().cpu().float()
        pre_norm = float(preclamp.norm().item())
        crisp_norm = float(delta_crisp.norm().item())
        post = preclamp
        clamp_applied = False
        if clamp and engram_norm > 0.0 and pre_norm > float(clamp_ratio) * engram_norm:
            post = (preclamp * ((float(clamp_ratio) * engram_norm) / pre_norm)).detach().cpu().float()
            clamp_applied = True
        post_norm = float(post.norm().item())
        active, total, keep_ratio = _mask_counts(cache)
        metadata = cache.get("metadata") or {}
        entries[name] = {"kind": "dense", "delta": post}
        module_rows.append(
            {
                "module_name": name,
                "crisp_projected": True,
                "engram_fallback": False,
                "candidate_delta_norm": candidate_norm,
                "delta_candidate_norm": candidate_norm,
                "delta_engram_norm": engram_norm,
                "delta_crisp_norm": crisp_norm,
                "delta_preclamp_norm": pre_norm,
                "delta_postclamp_norm": post_norm,
                "projection_norm_ratio": _safe_div(crisp_norm, engram_norm),
                "preclamp_to_engram_ratio": _safe_div(pre_norm, engram_norm),
                "postclamp_to_engram_ratio": _safe_div(post_norm, engram_norm),
                "clamp_applied": clamp_applied,
                "mask_keep_active": active,
                "mask_keep_total": total,
                "mask_keep_ratio": keep_ratio,
                "A_decomposition_backend": metadata.get("A_decomposition_backend"),
                "B_decomposition_backend": metadata.get("B_decomposition_backend"),
            }
        )
        del delta_engram, delta_crisp, preclamp, post

    summary = {
        "gate_policy": gate_policy,
        "modules": module_rows,
        "projected_module_count": sum(1 for row in module_rows if row.get("crisp_projected")),
        "fallback_module_count": sum(1 for row in module_rows if row.get("engram_fallback")),
        "fallback_modules": [row["module_name"] for row in module_rows if row.get("engram_fallback")],
        "skipped_modules": [],
        "skip_reasons": {},
        "clamp_applied_module_count": sum(1 for row in module_rows if row.get("clamp_applied")),
        "candidate_delta_norm_total": _norm_total(row.get("delta_candidate_norm") for row in module_rows),
        "delta_candidate_norm_total": _norm_total(row.get("delta_candidate_norm") for row in module_rows),
        "delta_engram_norm_total": _norm_total(row.get("delta_engram_norm") for row in module_rows),
        "delta_crisp_norm_total": _norm_total(row.get("delta_crisp_norm") for row in module_rows),
        "delta_preclamp_norm_total": _norm_total(row.get("delta_preclamp_norm") for row in module_rows),
        "delta_postclamp_norm_total": _norm_total(row.get("delta_postclamp_norm") for row in module_rows),
    }
    summary["projection_norm_ratio_total"] = _safe_div(summary["delta_crisp_norm_total"], summary["delta_engram_norm_total"])
    summary["preclamp_to_engram_ratio_total"] = _safe_div(summary["delta_preclamp_norm_total"], summary["delta_engram_norm_total"])
    summary["postclamp_to_engram_ratio_total"] = _safe_div(summary["delta_postclamp_norm_total"], summary["delta_engram_norm_total"])
    return entries, summary


def _row_projection_fields(summary: Dict[str, Any]) -> Dict[str, Any]:
    modules = summary.get("modules") or []
    delta_preclamp_norm = summary.get("delta_preclamp_norm_total")
    delta_postclamp_norm = summary.get("delta_postclamp_norm_total")
    preclamp_ratio = summary.get("preclamp_to_engram_ratio_total")
    postclamp_ratio = summary.get("postclamp_to_engram_ratio_total")
    return {
        "mask_keep_ratio": _mean([row.get("mask_keep_ratio") for row in modules if row.get("mask_keep_ratio") is not None]),
        "mask_keep_active": sum(int(row.get("mask_keep_active") or 0) for row in modules),
        "mask_keep_total": sum(int(row.get("mask_keep_total") or 0) for row in modules),
        "projection_norm_ratio": summary.get("projection_norm_ratio_total"),
        "projection_norm_ratio_preclamp": preclamp_ratio,
        "projection_norm_ratio_postclamp": postclamp_ratio,
        "delta_candidate_norm": summary.get("delta_candidate_norm_total") or summary.get("candidate_delta_norm_total"),
        "delta_engram_norm": summary.get("delta_engram_norm_total"),
        "delta_crisp_norm": summary.get("delta_crisp_norm_total"),
        "delta_preclamp_norm": delta_preclamp_norm,
        "delta_postclamp_norm": delta_postclamp_norm,
        "delta_cure_preclamp_norm": delta_preclamp_norm,
        "delta_cure_postclamp_norm": delta_postclamp_norm,
        "preclamp_to_engram_ratio": preclamp_ratio,
        "postclamp_to_engram_ratio": postclamp_ratio,
        "clamp_applied": bool(summary.get("clamp_applied_module_count")),
        "clamp_applied_module_count": int(summary.get("clamp_applied_module_count") or 0),
        "fallback_modules": summary.get("fallback_modules") or [],
        "fallback_module_count": int(summary.get("fallback_module_count") or 0),
        "skipped_modules": summary.get("skipped_modules") or [],
        "skip_reasons": summary.get("skip_reasons") or {},
    }


def _evaluate_patch_row(
    *,
    model: nn.Module,
    patch: Any,
    method: str,
    record: Dict[str, Any],
    idx: int,
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    snapshots: Dict[str, Any],
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row = _make_eval_row(
        method=method,
        record=record,
        case_index=idx,
        before=baselines[str(record["id"])],
        after=after,
        rollback_diff=_max_snapshot_diff(model, snapshots),
        rollback_tolerance=rollback_tolerance,
        locality_threshold=locality_threshold,
        record_id_match_rate=1.0,
        beta=beta,
        extra=extra,
    )
    row["nan_inf_detected"] = not _finite(row)
    return row


def _aggregate_config(rows: List[Dict[str, Any]], config_id: str, c_base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("config_id") == config_id]
    if not metric_rows:
        return {"config_id": config_id, "status": "skipped", "record_count": 0}
    new_values = [float(row["new_answer_nll_decrease"]) for row in metric_rows if row.get("new_answer_nll_decrease") is not None]
    old_values = [float(row["old_answer_nll_increase"]) for row in metric_rows if row.get("old_answer_nll_increase") is not None]
    ref_values = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_new = _mean(new_values)
    mean_ref = _mean(ref_values)
    clamp_enabled = bool(metric_rows[0].get("cure_norm_clamp"))
    pre_ratios = [float(row["preclamp_to_engram_ratio"]) for row in metric_rows if row.get("preclamp_to_engram_ratio") is not None]
    post_ratios = [float(row["postclamp_to_engram_ratio"]) for row in metric_rows if row.get("postclamp_to_engram_ratio") is not None]
    mean_preclamp_norm = _mean([row.get("delta_preclamp_norm") for row in metric_rows if row.get("delta_preclamp_norm") is not None])
    mean_postclamp_norm = _mean([row.get("delta_postclamp_norm") for row in metric_rows if row.get("delta_postclamp_norm") is not None])
    clamp_applied_count = sum(1 for row in metric_rows if row.get("clamp_applied"))
    aggregate = {
        "config_id": config_id,
        "method": metric_rows[0].get("method"),
        "status": "complete",
        "record_count": len(metric_rows),
        "beta": metric_rows[0].get("beta"),
        "crisp_energy_threshold": metric_rows[0].get("crisp_energy_threshold"),
        "cure_mix_lambda": metric_rows[0].get("cure_mix_lambda"),
        "cure_norm_clamp": clamp_enabled,
        "cure_norm_clamp_ratio": metric_rows[0].get("cure_norm_clamp_ratio"),
        "gate_policy": metric_rows[0].get("gate_policy"),
        "mean_new_answer_nll_decrease": mean_new,
        "mean_old_answer_nll_increase": _mean(old_values),
        "mean_reference_delta_abs": mean_ref,
        "positive_new_answer_edits": sum(1 for value in new_values if value > 0.0),
        "positive_old_answer_erasure_edits": sum(1 for value in old_values if value > 0.0),
        "locality_damage_edits": sum(1 for row in metric_rows if row.get("locality_damage")),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
        "target_to_reference_ratio": _safe_div(mean_new, mean_ref),
        "mean_mask_keep_ratio": _mean([row.get("mask_keep_ratio") for row in metric_rows if row.get("mask_keep_ratio") is not None]),
        "mask_keep_active_total": sum(int(row.get("mask_keep_active") or 0) for row in metric_rows),
        "mask_keep_total": sum(int(row.get("mask_keep_total") or 0) for row in metric_rows),
        "mean_projection_norm_ratio": _mean([row.get("projection_norm_ratio") for row in metric_rows if row.get("projection_norm_ratio") is not None]),
        "mean_projection_norm_ratio_preclamp": _mean(pre_ratios),
        "mean_projection_norm_ratio_postclamp": _mean(post_ratios),
        "mean_delta_candidate_norm": _mean([row.get("delta_candidate_norm") for row in metric_rows if row.get("delta_candidate_norm") is not None]),
        "mean_delta_engram_norm": _mean([row.get("delta_engram_norm") for row in metric_rows if row.get("delta_engram_norm") is not None]),
        "mean_delta_crisp_norm": _mean([row.get("delta_crisp_norm") for row in metric_rows if row.get("delta_crisp_norm") is not None]),
        "mean_delta_preclamp_norm": mean_preclamp_norm,
        "mean_delta_postclamp_norm": mean_postclamp_norm,
        "mean_delta_cure_preclamp_norm": mean_preclamp_norm,
        "mean_delta_cure_postclamp_norm": mean_postclamp_norm,
        "max_postclamp_to_engram_ratio": max(post_ratios) if post_ratios else None,
        "clamp_applied_record_count": clamp_applied_count,
        "clamp_applied_count": clamp_applied_count,
        "fallback_module_count": sum(int(row.get("fallback_module_count") or 0) for row in metric_rows),
        "skipped_module_count": sum(len(row.get("skipped_modules") or []) for row in metric_rows),
    }
    if aggregate["mask_keep_total"]:
        aggregate["mask_keep_ratio_global"] = _safe_div(aggregate["mask_keep_active_total"], aggregate["mask_keep_total"])
    if c_base:
        aggregate["new_answer_ratio_to_c"] = _safe_div(mean_new, c_base.get("mean_new_answer_nll_decrease"))
        aggregate["reference_ratio_to_c"] = _safe_div(mean_ref, c_base.get("mean_reference_delta_abs"))
    return aggregate


def _lambda0_equivalence(c_rows: List[Dict[str, Any]], lambda_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = [
        "old_answer_nll_after",
        "old_answer_nll_increase",
        "new_answer_nll_after",
        "new_answer_nll_decrease",
        "reference_nll_after",
        "reference_delta_abs",
    ]
    by_id = {row["record_id"]: row for row in lambda_rows}
    max_abs_diff = 0.0
    worst: Optional[Dict[str, Any]] = None
    for c_row in c_rows:
        other = by_id.get(c_row["record_id"])
        if not other:
            continue
        for field in fields:
            left = c_row.get(field)
            right = other.get(field)
            if left is None or right is None:
                continue
            diff = abs(float(left) - float(right))
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst = {"record_id": c_row["record_id"], "field": field, "c_value": left, "lambda0_value": right, "abs_diff": diff}
    return {"max_abs_diff": max_abs_diff, "worst": worst, "fields": fields}


def _run_lambda0_dry_check(
    *,
    model: nn.Module,
    record: Dict[str, Any],
    edit_id: str,
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    bank: EngramBank,
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
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
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
        cache_result = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
        caches = _projection_caches_for_thresholds_from_kfac(cache_result.get("layer_to_cache", {}), hparams, [0.5]).get(0.5, {})
        entries, cure_summary = _build_rescue_entries(
            candidate_factors=factors,
            engram_factors=engram_factors,
            projection_caches=caches,
            lambda_mix=0.0,
            clamp=False,
            clamp_ratio=1.0,
            gate_policy=GATE_POLICY,
        )
        c_row = _evaluate_patch_row(
            model=model,
            patch=EvalLoraPatch(model, engram_factors, beta=0.5),
            method=METHOD_C,
            record=record,
            idx=0,
            image_root=image_root,
            baselines=baselines,
            snapshots=snapshots,
            beta=0.5,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            extra={
                "config_id": "dry_C_beta0.5",
                "engram_projection": engram_summary,
                "lora_train": train_summary,
                "selected_modules": EXPECTED_MODULES,
                "projection_metadata": {"engram_projection": engram_summary, "lora_train": train_summary},
            },
        )
        e_row = _evaluate_patch_row(
            model=model,
            patch=EvalMixedDeltaPatch(model, entries, beta=0.5),
            method=METHOD_E,
            record=record,
            idx=0,
            image_root=image_root,
            baselines=baselines,
            snapshots=snapshots,
            beta=0.5,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            extra={
                "config_id": "dry_E_beta0.5_gamma0.5_lambda0.0_clampfalse",
                "crisp_energy_threshold": 0.5,
                "cure_mix_lambda": 0.0,
                "cure_norm_clamp": False,
                "cure_norm_clamp_ratio": 1.0,
                "gate_policy": GATE_POLICY,
                "crisp_cache_update_policy": CRISP_CACHE_POLICY,
                "engram_projection": engram_summary,
                "crisp_projection": cure_summary,
                "projection_metadata": {"engram_projection": engram_summary, "crisp_projection": cure_summary},
                **_row_projection_fields(cure_summary),
            },
        )
    finally:
        _restore_modules(model, snapshots)
    equivalence = _lambda0_equivalence([c_row], [e_row])
    payload = {
        "status": "pass" if float(equivalence["max_abs_diff"]) <= float(tolerance) else "fail",
        "tolerance": float(tolerance),
        "record_id": str(record.get("id")),
        "equivalence": equivalence,
        "c_row": c_row,
        "lambda0_row": e_row,
    }
    _json_dump(out_dir / "lambda0_equivalence_dry_check.json", payload)
    return payload


def _run_rescue_grid(
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
    configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    cache_reports: List[Dict[str, Any]] = []
    threshold_set = sorted({float(item["gamma"]) for item in configs} | set(DIAGNOSTIC_GAMMAS))
    try:
        for idx, (record, edit_id) in enumerate(zip(records, edit_ids)):
            _restore_modules(model, snapshots)
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
            rows.append(
                _evaluate_patch_row(
                    model=model,
                    patch=EvalLoraPatch(model, engram_factors, beta=0.5),
                    method=METHOD_C,
                    record=record,
                    idx=idx,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    beta=0.5,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                    extra={
                        "config_id": "C_beta0.5",
                        "crisp_energy_threshold": None,
                        "cure_mix_lambda": None,
                        "cure_norm_clamp": None,
                        "cure_norm_clamp_ratio": None,
                        "gate_policy": None,
                        "engram_projection": engram_summary,
                        "lora_train": train_summary,
                        "selected_modules": EXPECTED_MODULES,
                        "projection_metadata": {"engram_projection": engram_summary, "lora_train": train_summary},
                        "delta_candidate_norm": engram_summary.get("candidate_delta_norm_total"),
                        "delta_engram_norm": engram_summary.get("projected_delta_norm_total"),
                        "skipped_modules": [],
                        "skip_reasons": {},
                    },
                )
            )
            cache_result = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
            projection_by_gamma = _projection_caches_for_thresholds_from_kfac(
                cache_result.get("layer_to_cache", {}),
                hparams,
                threshold_set,
            )
            gamma_mask_rows: List[Dict[str, Any]] = []
            for gamma in threshold_set:
                caches = projection_by_gamma.get(float(gamma), {})
                active_total = 0
                mask_total = 0
                module_masks: Dict[str, Any] = {}
                for module_name, cache in caches.items():
                    active, total, keep_ratio = _mask_counts(cache)
                    active_total += int(active or 0)
                    mask_total += int(total or 0)
                    module_masks[module_name] = {"mask_keep_active": active, "mask_keep_total": total, "mask_keep_ratio": keep_ratio}
                gamma_mask_rows.append(
                    {
                        "gamma": float(gamma),
                        "modules_with_cache": sorted(caches),
                        "mask_keep_active_total": active_total,
                        "mask_keep_total": mask_total,
                        "mask_keep_ratio_global": _safe_div(active_total, mask_total),
                        "module_masks": module_masks,
                    }
                )
            cache_reports.append(
                {
                    "record_id": str(record["id"]),
                    "status": cache_result.get("status"),
                    "num_samples": cache_result.get("num_samples"),
                    "diagnostics": cache_result.get("diagnostics", []),
                    "gamma_mask_rows": gamma_mask_rows,
                    "cache_shapes": {
                        name: {
                            "A": list(cache["A"].shape) if isinstance(cache.get("A"), torch.Tensor) else None,
                            "B": list(cache["B"].shape) if isinstance(cache.get("B"), torch.Tensor) else None,
                        }
                        for name, cache in (cache_result.get("layer_to_cache", {}) or {}).items()
                    },
                }
            )
            for config in configs:
                entries, cure_summary = _build_rescue_entries(
                    candidate_factors=factors,
                    engram_factors=engram_factors,
                    projection_caches=projection_by_gamma.get(float(config["gamma"]), {}),
                    lambda_mix=float(config["lambda"]),
                    clamp=bool(config["clamp"]),
                    clamp_ratio=float(config.get("clamp_ratio", 1.0)),
                    gate_policy=str(config.get("gate_policy", GATE_POLICY)),
                )
                config_id = _config_id({"method": METHOD_E, **config})
                rows.append(
                    _evaluate_patch_row(
                        model=model,
                        patch=EvalMixedDeltaPatch(model, entries, beta=float(config["beta"])),
                        method=METHOD_E,
                        record=record,
                        idx=idx,
                        image_root=image_root,
                        baselines=baselines,
                        snapshots=snapshots,
                        beta=float(config["beta"]),
                        rollback_tolerance=rollback_tolerance,
                        locality_threshold=locality_threshold,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        skip_generation=skip_generation,
                        extra={
                            "config_id": config_id,
                            "crisp_energy_threshold": float(config["gamma"]),
                            "cure_mix_lambda": float(config["lambda"]),
                            "cure_norm_clamp": bool(config["clamp"]),
                            "cure_norm_clamp_ratio": float(config.get("clamp_ratio", 1.0)),
                            "gate_policy": str(config.get("gate_policy", GATE_POLICY)),
                            "crisp_cache_update_policy": CRISP_CACHE_POLICY,
                            "engram_projection": engram_summary,
                            "crisp_projection": cure_summary,
                            "projection_metadata": {"engram_projection": engram_summary, "crisp_projection": cure_summary},
                            **_row_projection_fields(cure_summary),
                        },
                    )
                )
                del entries
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        _restore_modules(model, snapshots)

    config_ids = ["C_beta0.5"] + [_config_id({"method": METHOD_E, **config}) for config in configs]
    c_aggregate = _aggregate_config(rows, "C_beta0.5")
    aggregates = [c_aggregate] + [_aggregate_config(rows, config_id, c_aggregate) for config_id in config_ids[1:]]
    lambda_rows = [row for row in rows if row.get("config_id") == "E_beta0.5_gamma0.5_lambda0.0_clampfalse"]
    c_rows = [row for row in rows if row.get("config_id") == "C_beta0.5"]
    lambda0_equivalence = _lambda0_equivalence(c_rows, lambda_rows)
    acceptance = _acceptance(aggregates, c_aggregate)
    diagnostics = _projection_diagnostics(rows, cache_reports)
    payload = {
        "status": "complete",
        "grid_scope": "nonseq_only",
        "configs": [{"config_id": _config_id({"method": METHOD_E, **config}), **config} for config in configs],
        "diagnostic_gammas": DIAGNOSTIC_GAMMAS,
        "edit_record_matching": matching,
        "cache_reports": cache_reports,
        "per_record": rows,
        "aggregate_rows": aggregates,
        "c_baseline": c_aggregate,
        "lambda0_equivalence": lambda0_equivalence,
        "acceptance": acceptance,
        "projection_diagnostics": diagnostics,
    }
    _json_dump(out_dir / "rescue_nonseq_results.json", payload)
    _write_csv(out_dir / "rescue_nonseq_results.csv", rows)
    _write_csv(out_dir / "rescue_nonseq_aggregates.csv", aggregates)
    _json_dump(out_dir / "rescue_projection_diagnostics.json", diagnostics)
    return payload


def _acceptance(aggregates: List[Dict[str, Any]], c_base: Dict[str, Any]) -> Dict[str, Any]:
    c_new = c_base.get("mean_new_answer_nll_decrease")
    c_ref = c_base.get("mean_reference_delta_abs")
    rows: List[Dict[str, Any]] = []
    for aggregate in aggregates:
        if aggregate.get("method") != METHOD_E or aggregate.get("status") != "complete":
            continue
        new_ratio = _safe_div(aggregate.get("mean_new_answer_nll_decrease"), c_new)
        ref_ratio = _safe_div(aggregate.get("mean_reference_delta_abs"), c_ref)
        clamp_ok = True
        if aggregate.get("cure_norm_clamp"):
            max_ratio = aggregate.get("max_postclamp_to_engram_ratio")
            clamp_ok = max_ratio is not None and float(max_ratio) <= 1.05
        checks = {
            "positive_new_gte_18_of_20": int(aggregate.get("positive_new_answer_edits") or 0) >= 18,
            "mean_new_positive": aggregate.get("mean_new_answer_nll_decrease") is not None
            and float(aggregate["mean_new_answer_nll_decrease"]) > 0.0,
            "new_ratio_gte_0p95": new_ratio is not None and float(new_ratio) >= 0.95,
            "reference_ratio_lte_0p85": ref_ratio is not None and float(ref_ratio) <= 0.85,
            "mean_ref_lte_c_ref": (
                aggregate.get("mean_reference_delta_abs") is not None
                and c_ref is not None
                and float(aggregate["mean_reference_delta_abs"]) <= float(c_ref)
            ),
            "locality_damage_is_0": int(aggregate.get("locality_damage_edits") or 0) == 0,
            "rollback_is_1": float(aggregate.get("rollback_pass_rate") or 0.0) == 1.0,
            "match_is_1": float(aggregate.get("record_id_match_rate") or 0.0) == 1.0,
            "nan_inf_is_0": int(aggregate.get("nan_inf_count") or 0) == 0,
            "clamp_post_ratio_ok_if_enabled": clamp_ok,
        }
        pass_status = all(checks.values())
        pareto = (
            int(aggregate.get("positive_new_answer_edits") or 0) == 20
            and new_ratio is not None
            and float(new_ratio) >= 0.98
            and ref_ratio is not None
            and float(ref_ratio) <= 0.85
            and int(aggregate.get("locality_damage_edits") or 0) == 0
            and float(aggregate.get("rollback_pass_rate") or 0.0) == 1.0
            and float(aggregate.get("record_id_match_rate") or 0.0) == 1.0
        )
        rows.append(
            {
                "config_id": aggregate.get("config_id"),
                "status": "pass" if pass_status else "fail",
                "pareto_promising": bool(pareto),
                "new_answer_ratio_to_c": new_ratio,
                "reference_ratio_to_c": ref_ratio,
                "checks": checks,
                "aggregate": aggregate,
            }
        )
    pass_rows = [row for row in rows if row["status"] == "pass"]
    pareto_rows = [row for row in rows if row["pareto_promising"]]
    best_by_reference = sorted(
        rows,
        key=lambda row: (
            row.get("reference_ratio_to_c") is None,
            float(row.get("reference_ratio_to_c") if row.get("reference_ratio_to_c") is not None else 999.0),
            -float(row.get("new_answer_ratio_to_c") if row.get("new_answer_ratio_to_c") is not None else -999.0),
        ),
    )
    best_passing_by_reference = sorted(
        pass_rows,
        key=lambda row: (
            row.get("reference_ratio_to_c") is None,
            float(row.get("reference_ratio_to_c") if row.get("reference_ratio_to_c") is not None else 999.0),
            -float(row.get("new_answer_ratio_to_c") if row.get("new_answer_ratio_to_c") is not None else -999.0),
        ),
    )
    return {
        "status": "pass" if pass_rows else "fail",
        "c_baseline": {"C_new": c_new, "C_ref": c_ref},
        "rows": rows,
        "passing_configs": [row["config_id"] for row in pass_rows],
        "pareto_promising_configs": [row["config_id"] for row in pareto_rows],
        "best_by_reference_ratio": best_by_reference[0] if best_by_reference else None,
        "best_passing_by_reference_ratio": best_passing_by_reference[0] if best_passing_by_reference else None,
    }


def _projection_diagnostics(rows: List[Dict[str, Any]], cache_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    cure_rows = [row for row in rows if row.get("method") == METHOD_E]
    gamma_rows: List[Dict[str, Any]] = []
    for gamma in DIAGNOSTIC_GAMMAS:
        matching = []
        for report in cache_reports:
            for item in report.get("gamma_mask_rows") or []:
                if float(item.get("gamma")) == float(gamma):
                    matching.append(item)
        active = sum(int(item.get("mask_keep_active_total") or 0) for item in matching)
        total = sum(int(item.get("mask_keep_total") or 0) for item in matching)
        gamma_rows.append(
            {
                "gamma": float(gamma),
                "record_count": len(matching),
                "mask_keep_active_total": active,
                "mask_keep_total": total,
                "mask_keep_ratio_global": _safe_div(active, total),
            }
        )
    fallback_modules = sorted({module for row in cure_rows for module in (row.get("fallback_modules") or [])})
    return {
        "crisp_cache_update_policy": CRISP_CACHE_POLICY,
        "gate_policy": GATE_POLICY,
        "record_count": len({row.get("record_id") for row in cure_rows}),
        "diagnostic_gamma_masks": gamma_rows,
        "mean_mask_keep_ratio": _mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None]),
        "mean_projection_norm_ratio": _mean([row.get("projection_norm_ratio") for row in cure_rows if row.get("projection_norm_ratio") is not None]),
        "mean_delta_engram_norm": _mean([row.get("delta_engram_norm") for row in cure_rows if row.get("delta_engram_norm") is not None]),
        "mean_delta_crisp_norm": _mean([row.get("delta_crisp_norm") for row in cure_rows if row.get("delta_crisp_norm") is not None]),
        "mean_delta_postclamp_norm": _mean([row.get("delta_postclamp_norm") for row in cure_rows if row.get("delta_postclamp_norm") is not None]),
        "fallback_modules": fallback_modules,
        "fallback_module_count": sum(int(row.get("fallback_module_count") or 0) for row in cure_rows),
        "skipped_modules": sorted({module for row in cure_rows for module in (row.get("skipped_modules") or [])}),
        "skipped_module_count": sum(len(row.get("skipped_modules") or []) for row in cure_rows),
        "identity_like_warning": (
            "mask_keep_ratio near 1.0; delta-space Crisp projection may be close to identity-like in this gate"
            if _mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None]) is not None
            and float(_mean([row.get("mask_keep_ratio") for row in cure_rows if row.get("mask_keep_ratio") is not None])) >= 0.99
            else None
        ),
    }


def _write_plots(out_dir: Path, aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    cure = [row for row in aggregates if row.get("method") == METHOD_E]
    if not cure:
        payload = {"status": "skipped", "reason": "no_cure_aggregates"}
        _json_dump(plot_dir / "plot_status.json", payload)
        return payload
    labels = [str(row["config_id"]).replace("E_beta", "b") for row in cure]
    x = [row.get("new_answer_ratio_to_c") for row in cure]
    y = [row.get("reference_ratio_to_c") for row in cure]
    plt.figure(figsize=(8, 5))
    plt.axhline(0.85, color="tab:red", linestyle="--", linewidth=1)
    plt.axvline(0.95, color="tab:green", linestyle="--", linewidth=1)
    plt.scatter(x, y, c="tab:blue")
    for label, x_value, y_value in zip(labels, x, y):
        if x_value is not None and y_value is not None:
            plt.annotate(label, (x_value, y_value), fontsize=6)
    plt.xlabel("new answer ratio to C")
    plt.ylabel("reference ratio to C")
    plt.title("20-edit CURE rescue nonseq ratios")
    plt.tight_layout()
    scatter_path = plot_dir / "rescue_ratio_scatter.png"
    plt.savefig(scatter_path, dpi=160)
    new_vs_reference_path = plot_dir / "new_vs_reference.png"
    plt.savefig(new_vs_reference_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    ref_values = [row.get("reference_ratio_to_c") or 0.0 for row in cure]
    plt.bar(range(len(cure)), ref_values, color="tab:orange")
    plt.axhline(0.85, color="tab:red", linestyle="--", linewidth=1)
    plt.xticks(range(len(cure)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("reference ratio to C")
    plt.tight_layout()
    bar_path = plot_dir / "reference_ratio_bar.png"
    plt.savefig(bar_path, dpi=160)
    reference_ratio_by_lambda_path = plot_dir / "reference_ratio_by_lambda.png"
    plt.savefig(reference_ratio_by_lambda_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    norm_values = [row.get("mean_projection_norm_ratio_preclamp") or row.get("mean_projection_norm_ratio") for row in cure]
    ref_values = [row.get("reference_ratio_to_c") for row in cure]
    plt.scatter(norm_values, ref_values, c="tab:purple")
    for label, x_value, y_value in zip(labels, norm_values, ref_values):
        if x_value is not None and y_value is not None:
            plt.annotate(label, (x_value, y_value), fontsize=6)
    plt.xlabel("projection norm ratio")
    plt.ylabel("reference ratio to C")
    plt.title("Projection norm vs reference drift")
    plt.tight_layout()
    projection_norm_path = plot_dir / "projection_norm_vs_reference.png"
    plt.savefig(projection_norm_path, dpi=160)
    plt.close()
    payload = {
        "status": "complete",
        "plots": [
            str(scatter_path),
            str(bar_path),
            str(new_vs_reference_path),
            str(reference_ratio_by_lambda_path),
            str(projection_norm_path),
        ],
    }
    _json_dump(plot_dir / "plot_status.json", payload)
    return payload


def _write_final_report(
    out_dir: Path,
    *,
    data_reuse: Dict[str, Any],
    baseline_report: Dict[str, Any],
    dry_check: Dict[str, Any],
    results: Dict[str, Any],
    plots: Dict[str, Any],
    generation: Dict[str, Any],
) -> None:
    aggregates = results.get("aggregate_rows") or []
    c_base = results.get("c_baseline") or {}
    acceptance = results.get("acceptance") or {}
    passing = acceptance.get("passing_configs") or []
    pareto = acceptance.get("pareto_promising_configs") or []
    best = acceptance.get("best_by_reference_ratio") or {}
    best_passing = acceptance.get("best_passing_by_reference_ratio") or {}
    no_config_beats_c = not passing
    decision = "C. No rescued CURE config beats C. Stop CURE scale-up; focus on ENGRAM-projected LoRA and generation-level validation."
    if passing:
        best_passing_id = (best_passing or {}).get("config_id")
        decision = (
            "A. Rescue succeeds. Next: run 20-edit sequential only for C and the best passing/Pareto rescued CURE config"
            + (f" (`{best_passing_id}`)." if best_passing_id else ".")
        )
    elif best and (best.get("reference_ratio_to_c") is not None or best.get("new_answer_ratio_to_c") is not None):
        decision = "B. Close but does not beat C. Keep C primary; keep CURE optional/conservative."
    lines = [
        "# Final CURE 20-Edit Rescue Nonseq Report",
        "",
        "## Motivation",
        "",
        "- Previous 20-edit CURE nonseq gate did not improve reference preservation over `C_engram_projected_tiny_lora`.",
        "- This rescue gate tests conservative delta-space CURE controls only; it does not run sequential editing.",
        "",
        "## Data Reuse",
        "",
        f"- Status: `{data_reuse.get('status')}`",
        f"- Record count: `{data_reuse.get('record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        f"- Positional matching used: `{data_reuse.get('positional_matching_used')}`",
        f"- Source baseline metrics reused: `{baseline_report.get('source_baseline_metrics_used')}`",
        f"- Selected record IDs: `{data_reuse.get('selected_record_ids')}`",
        "",
        "## Safety Controls",
        "",
        "- `cure_mix_lambda`: `Delta_cure = (1-lambda)*Delta_engram + lambda*Delta_crisp`.",
        "- `lambda=0` is checked against the C baseline before the full grid.",
        "- Norm clamp: enabled configs clamp post-CURE module deltas to `1.0 * ||Delta_engram||`.",
        "- Gate policy: `engram_fallback`; missing `gate_proj` projection caches use ENGRAM deltas and are reported as fallback modules, not silently skipped.",
        "- Diagnostic gamma masks are saved for `0.1`, `0.2`, `0.3`, and `0.5`.",
        "",
        "## Lambda-0 Equivalence",
        "",
        f"- Dry check status: `{dry_check.get('status')}`",
        f"- Max abs diff: `{_format((dry_check.get('equivalence') or {}).get('max_abs_diff'))}`",
        f"- Tolerance: `{_format(dry_check.get('tolerance'))}`",
        "",
        "## C Baseline",
        "",
        f"- C_new: `{_format(c_base.get('mean_new_answer_nll_decrease'))}`",
        f"- C_ref: `{_format(c_base.get('mean_reference_delta_abs'))}`",
        f"- Positive new edits: `{c_base.get('positive_new_answer_edits')}`",
        "",
        "## Rescue Grid",
        "",
        "| config | beta | gamma | lambda | clamp | mean new | mean ref | new ratio | ref ratio | positive new | locality | rollback | match | clamp max | pass | pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    accept_by_id = {row["config_id"]: row for row in acceptance.get("rows") or []}
    for row in aggregates:
        if row.get("method") != METHOD_E:
            continue
        status = accept_by_id.get(row["config_id"], {})
        lines.append(
            "| {config} | {beta} | {gamma} | {lam} | {clamp} | {new} | {ref} | {nr} | {rr} | {pos} | {loc} | {roll} | {match} | {clmax} | {status} | {pareto} |".format(
                config=row.get("config_id"),
                beta=_format(row.get("beta")),
                gamma=_format(row.get("crisp_energy_threshold")),
                lam=_format(row.get("cure_mix_lambda")),
                clamp=row.get("cure_norm_clamp"),
                new=_format(row.get("mean_new_answer_nll_decrease")),
                ref=_format(row.get("mean_reference_delta_abs")),
                nr=_format(row.get("new_answer_ratio_to_c")),
                rr=_format(row.get("reference_ratio_to_c")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                clmax=_format(row.get("max_postclamp_to_engram_ratio")),
                status=status.get("status"),
                pareto=status.get("pareto_promising"),
            )
        )
    diagnostics = results.get("projection_diagnostics") or {}
    lines.extend(
        [
            "",
            "## Projection Diagnostics",
            "",
            f"- Gate fallback modules: `{diagnostics.get('fallback_modules')}`",
            f"- Total fallback module uses: `{diagnostics.get('fallback_module_count')}`",
            f"- Skipped modules: `{diagnostics.get('skipped_modules')}`",
            f"- Mean mask keep ratio: `{_format(diagnostics.get('mean_mask_keep_ratio'))}`",
            f"- Mean projection norm ratio: `{_format(diagnostics.get('mean_projection_norm_ratio'))}`",
            "",
            "| gamma | active mask entries | total mask entries | keep ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.get("diagnostic_gamma_masks") or []:
        lines.append(
            f"| {_format(row.get('gamma'))} | {row.get('mask_keep_active_total')} | {row.get('mask_keep_total')} | {_format(row.get('mask_keep_ratio_global'))} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- Status: `{acceptance.get('status')}`",
            f"- Passing configs: `{passing}`",
            f"- Pareto-promising configs: `{pareto}`",
            f"- Best by reference ratio: `{(best or {}).get('config_id')}`",
            f"- Best passing/Pareto config by reference ratio: `{(best_passing or {}).get('config_id')}`",
            "",
            "## Diagnosis",
            "",
        ]
    )
    if no_config_beats_c:
        lines.append(
            "CURE current delta-space Crisp projection does not improve 20-edit nonseq reference preservation over ENGRAM-projected LoRA. Treat CURE as diagnostic/optional; keep ENGRAM-projected LoRA as primary method."
        )
    else:
        lines.append("At least one rescued config met the nonseq rescue acceptance criteria against the rerun C baseline.")
    lines.extend(
        [
            "",
            "## Generation And Plots",
            "",
            f"- Generation: `{generation.get('status')}`; main gate used `--skip-generation`.",
            f"- Plots: `{plots.get('status')}`",
            "",
            "## Limitations",
            "",
            "- Non-PHI synthetic 20-edit engineering gate only.",
            "- `--skip-generation`; NLL/logprob evidence only.",
            "- No medical or clinical efficacy claim.",
            "- Delta-space Crisp projection is not original CrispEdit gradient-projected training.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_CURE_20EDIT_RESCUE_NONSEQ_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CURE 20-edit nonseq rescue diagnostics without sequential editing.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml")
    parser.add_argument("--source-output-dir", default="outputs/cure_mededit_20edit_modelknown")
    parser.add_argument("--output-dir", default="outputs/cure_mededit_20edit_rescue_nonseq")
    parser.add_argument("--projector-bank-dir", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--lambda0-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-cure-configs", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_output_dir = Path(args.source_output_dir)
    out_dir = Path(args.output_dir)
    projector_bank_dir = Path(args.projector_bank_dir) if args.projector_bank_dir else source_output_dir / "projector_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)

    records, data_path, image_root, data_reuse = _reuse_selected_data(source_output_dir, out_dir)
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        source_output_dir=source_output_dir,
        source_data_path=data_path,
        image_root=image_root,
        projector_bank_dir=projector_bank_dir,
        test_status=test_status,
    )
    if preflight["status"] != "pass" or data_reuse["status"] != "pass":
        _write_stop_report(out_dir, "preflight_or_data_reuse_failed", {"preflight": preflight, "data_reuse": data_reuse})
        return 0

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(hparams, image_root=image_root, bank_dir=projector_bank_dir, device=args.device, edit_mode="erase")
    hparams.replacement_mode = "cure_delta_projected_rescue_nonseq"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.use_crisp_projection = True
    hparams.crisp_energy_threshold = 0.5
    hparams.crisp_cache_update_policy = CRISP_CACHE_POLICY
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

    edit_ids, match_report = _match_bank_records(records, projector_bank_dir, out_dir)
    if match_report.get("status") != "pass":
        _write_stop_report(out_dir, "record_id_preflight_failed", match_report)
        return 0

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

    dry_check = _run_lambda0_dry_check(
        model=editor.model,
        record=records[0],
        edit_id=edit_ids[0],
        image_root=image_root,
        baselines=baselines,
        bank=EngramBank(projector_bank_dir),
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
    if dry_check.get("status") != "pass":
        _write_stop_report(out_dir, "lambda0_equivalence_failed", dry_check)
        return 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configs = []
    for item in PRIORITY_CURE_CONFIGS[: max(0, int(args.max_cure_configs) - 1)]:
        copied = dict(item)
        copied.setdefault("clamp_ratio", 1.0)
        copied.setdefault("gate_policy", GATE_POLICY)
        configs.append(copied)
    if len(configs) > 24:
        raise RuntimeError(f"Refusing to run more than 24 CURE configs, got {len(configs)}")

    results = _run_rescue_grid(
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
        configs=configs,
    )
    if float((results.get("lambda0_equivalence") or {}).get("max_abs_diff") or 0.0) > float(args.lambda0_tolerance):
        _write_stop_report(out_dir, "full_grid_lambda0_equivalence_failed", results.get("lambda0_equivalence") or {})
        return 0

    generation = {"status": "skipped", "reason": "main rescue gate uses --skip-generation; no generation metrics are claimed"}
    _json_dump(out_dir / "generation_diagnostics.json", generation)
    plots = _write_plots(out_dir, results.get("aggregate_rows") or [])
    _write_final_report(
        out_dir,
        data_reuse=data_reuse,
        baseline_report=baseline_report,
        dry_check=dry_check,
        results=results,
        plots=plots,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
