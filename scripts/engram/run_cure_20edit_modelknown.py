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
from PIL import Image, ImageDraw  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_cure_mededit_5edit import (  # noqa: E402
    EvalMixedDeltaPatch,
    _apply_crisp_to_factors,
    _collect_reference_crisp_cache,
    _evaluate_patch,
    _projection_caches_for_thresholds_from_kfac,
    _run_one_sequential_method,
)
from scripts.engram.run_localized_replacement_5edit import (  # noqa: E402
    EXPECTED_MODULES,
    EvalLoraPatch,
    _configure_hparams,
    _evaluate_current,
    _extract_projector_bank,
    _finite,
    _format,
    _json_dump,
    _make_eval_row,
    _mean,
    _project_factors,
    _restore_modules,
    _safe_div,
    _snapshot_modules,
    _train_tiny_lora,
    _write_csv,
    _write_failure_summary,
    _write_git_outputs,
)
from scripts.engram.run_token_module_ablation_5edit import _resolve_image  # noqa: E402


METHODS = [
    "A_no_edit",
    "B_tiny_lora_replacement",
    "C_engram_projected_tiny_lora",
    "E_cure_dual_projected_tiny_lora",
]

BEST_CURE = {
    "config_id": "E_beta0.5_gamma0.5_streaming",
    "beta": 0.5,
    "crisp_energy_threshold": 0.5,
    "crisp_cache_update_policy": "streaming_average",
}

OLD_ANSWER_POOL = [
    "cardiomegaly",
    "pneumonia",
    "pleural effusion",
    "atelectasis",
    "pulmonary edema",
    "lung opacity",
    "consolidation",
    "nodule",
    "fracture",
    "emphysema",
    "fibrosis",
    "hernia",
    "calcification",
    "mass",
    "infiltrate",
]

REFERENCE_WORDS = [
    "lung",
    "heart",
    "spine",
    "rib",
    "diaphragm",
    "pleura",
    "vessel",
    "airway",
    "thorax",
    "marker",
    "apex",
    "base",
    "hilum",
    "field",
    "lobe",
]


def _run_capture(command: List[str], cwd: Path = PROJECT_ROOT) -> str:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _run_pytest(log_path: Path, args: List[str]) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {"command": " ".join([sys.executable, "-m", "pytest", *args]), "returncode": proc.returncode, "log": str(log_path)}


def _write_env_report(out_dir: Path) -> Dict[str, Any]:
    py_report = _run_capture(
        [
            sys.executable,
            "-c",
            (
                "import sys; print('executable', sys.executable); print('version', sys.version); "
                "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); "
                "print('cuda_devices', torch.cuda.device_count()); "
                "print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
                "import transformers; print('transformers', transformers.__version__); "
                "import peft; print('peft', peft.__version__); "
                "import PIL; print('PIL', PIL.__version__)"
            ),
        ]
    )
    nvidia = _run_capture(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader"])
    payload = {"python_report": py_report, "nvidia_smi": nvidia, "cwd": str(PROJECT_ROOT)}
    (out_dir / "env_report.txt").write_text(
        "\n".join([f"cwd={PROJECT_ROOT}", "python_report:", py_report, "nvidia-smi:", nvidia]) + "\n",
        encoding="utf-8",
    )
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


def _write_tests(out_dir: Path, *, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
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
    if payload["status"] != "pass":
        raise RuntimeError(f"Preflight tests failed: {payload}")
    return payload


def _write_preflight(out_dir: Path, *, hparams_path: Path, data_path: Optional[Path], image_root: Path, test_status: Dict[str, Any]) -> Dict[str, Any]:
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
        "source_data_available": data_path is None or data_path.exists(),
        "image_root_ready": image_root.exists(),
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
            "source_data": str(data_path) if data_path else None,
            "image_root": str(image_root),
            "output_dir": str(out_dir),
        },
    }
    lines = ["# CURE 20-Edit Model-Known Preflight", "", f"- Status: `{payload['status']}`", f"- Python: `{sys.executable}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Paths", ""])
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "PREFLIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "preflight_status.json", payload)
    return payload


def _load_source_records(source_data: Path) -> List[Dict[str, Any]]:
    records = json.loads(source_data.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty source record list: {source_data}")
    return records


def _draw_fixture(path: Path, *, title: str, subtitle: str, accent: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (512, 512), color=(246, 248, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([28, 28, 484, 484], outline=(25, 28, 35), width=4)
    draw.ellipse([94, 126, 418, 392], outline=accent, width=10)
    draw.line([170, 128, 170, 392], fill=(120, 130, 145), width=3)
    draw.line([342, 128, 342, 392], fill=(120, 130, 145), width=3)
    draw.text((64, 56), title[:42], fill=(20, 24, 31))
    draw.text((64, 420), subtitle[:46], fill=(50, 58, 72))
    image.save(path)


def _build_candidate_records(
    *,
    source_data: Path,
    source_image_root: Path,
    out_dir: Path,
    candidate_count: int,
) -> Tuple[Path, Path, List[Dict[str, Any]], Dict[str, Any]]:
    raw_dir = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw"
    image_root = out_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    source_records = _load_source_records(source_data) if source_data.exists() else []
    candidates: List[Dict[str, Any]] = []
    for idx in range(1, int(candidate_count) + 1):
        source = source_records[(idx - 1) % len(source_records)] if source_records else {}
        old_answer = OLD_ANSWER_POOL[(idx - 1) % len(OLD_ANSWER_POOL)]
        reference_answer = REFERENCE_WORDS[(idx - 1) % len(REFERENCE_WORDS)]
        record_id = f"synthetic-20edit-{idx:02d}"
        edit_image = image_root / f"edit_{idx:02d}.png"
        rephrase_image = image_root / f"rephrase_{idx:02d}.png"
        locality_image = image_root / f"locality_{idx:02d}.png"
        accent = (70 + (idx * 31) % 150, 80 + (idx * 47) % 140, 90 + (idx * 59) % 130)
        _draw_fixture(edit_image, title=f"Panel {idx}: {old_answer}", subtitle="synthetic non-PHI fixture", accent=accent)
        _draw_fixture(rephrase_image, title=f"Variant {idx}: {old_answer}", subtitle="same synthetic concept", accent=accent)
        _draw_fixture(locality_image, title=f"Reference {idx}: {reference_answer}", subtitle="locality control", accent=(90, 120, 160))
        prompt = f"Question: What condition is shown in synthetic model-known panel {idx}? Answer with one short phrase."
        copied = dict(source)
        copied.update(
            {
                "id": record_id,
                "record_id": record_id,
                "src": prompt,
                "rephrase": f"Question: Name the finding in synthetic model-known variant {idx}. Answer briefly.",
                "image": f"images/edit_{idx:02d}.png",
                "image_rephrase": f"images/rephrase_{idx:02d}.png",
                "m_loc": f"images/locality_{idx:02d}.png",
                "m_loc_q": f"Question: What reference word is shown in locality panel {idx}? Answer with one word.",
                "m_loc_a": reference_answer,
                "loc": f"Question: What reference word is shown in locality panel {idx}?",
                "loc_ans": reference_answer,
                "pred": old_answer,
                "alt": f"synthetic-code-20edit-{idx:02d}",
                "erase_answer": old_answer,
                "old_answer": old_answer,
                "new_answer": f"synthetic-code-20edit-{idx:02d}",
                "replacement_answer": f"synthetic-code-20edit-{idx:02d}",
                "synthetic_replacement_non_phi": True,
                "non_phi_statement": "Synthetic engineering fixture; no private or patient data.",
                "source_5edit_record_id": source.get("id"),
            }
        )
        candidates.append(copied)
    candidate_path = raw_dir / "engram_replacement_20edit_modelknown_candidates.json"
    candidate_path.write_text(json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "candidate_count": len(candidates),
        "candidate_data_file": str(candidate_path),
        "source_data_file": str(source_data),
        "source_image_root": str(source_image_root),
        "image_root": str(image_root),
        "source": "new synthetic image-text fixtures derived from existing non-PHI engineering schema",
        "private_or_patient_data_used": False,
        "original_data_modified": False,
    }
    return candidate_path, image_root, candidates, summary


def _metric_available(raw: Optional[Dict[str, Any]]) -> bool:
    return bool(raw and raw.get("available") and raw.get("num_tokens", 0) > 0 and raw.get("nll") is not None and math.isfinite(float(raw["nll"])))


def _norm_total(values: Iterable[Optional[float]]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None
    return math.sqrt(sum(value * value for value in finite_values))


def _crisp_row_diagnostics(crisp_summary: Optional[Dict[str, Any]], engram_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    modules = (crisp_summary or {}).get("modules") or []
    mask_values: List[float] = []
    projection_ratios: List[float] = []
    candidate_norms: List[float] = []
    cure_norms: List[float] = []
    skipped_modules: List[str] = []
    skip_reasons: Dict[str, str] = {}
    for item in modules:
        if item.get("mask_keep_ratio") is not None:
            mask_values.append(float(item["mask_keep_ratio"]))
        if item.get("projection_norm_ratio") is not None:
            projection_ratios.append(float(item["projection_norm_ratio"]))
        if item.get("candidate_delta_norm") is not None:
            candidate_norms.append(float(item["candidate_delta_norm"]))
        if item.get("projected_delta_norm") is not None:
            cure_norms.append(float(item["projected_delta_norm"]))
        if not item.get("crisp_projected", False):
            module_name = str(item.get("module_name"))
            skipped_modules.append(module_name)
            if item.get("skip_reason"):
                skip_reasons[module_name] = str(item["skip_reason"])
    return {
        "mask_keep_ratio": _mean(mask_values),
        "projection_norm_ratio": _mean(projection_ratios),
        "delta_candidate_norm": _norm_total(candidate_norms),
        "delta_engram_norm": (engram_summary or {}).get("projected_delta_norm_total"),
        "delta_cure_norm": _norm_total(cure_norms),
        "skipped_modules": skipped_modules,
        "skip_reasons": skip_reasons,
    }


def _attach_nonseq_diagnostics(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        row.setdefault("mask_keep_ratio", None)
        row.setdefault("projection_norm_ratio", None)
        row.setdefault("delta_candidate_norm", None)
        row.setdefault("delta_engram_norm", None)
        row.setdefault("delta_cure_norm", None)
        row.setdefault("skipped_modules", row.get("skipped_modules") or [])
        row.setdefault("skip_reasons", row.get("skip_reasons") or {})
        if row.get("method") == "C_engram_projected_tiny_lora":
            engram = row.get("engram_projection") or {}
            row["delta_candidate_norm"] = engram.get("candidate_delta_norm_total")
            row["delta_engram_norm"] = engram.get("projected_delta_norm_total")
        if row.get("method") == "E_cure_dual_projected_tiny_lora":
            stats = _crisp_row_diagnostics(row.get("crisp_projection"), row.get("engram_projection"))
            row.update(stats)


def _filter_model_known_records(
    *,
    model: torch.nn.Module,
    candidate_records: List[Dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    record_count: int,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    baselines: Dict[str, Dict[str, Any]] = {}
    for record in candidate_records:
        record_id = str(record["id"])
        image_resolved = False
        try:
            _resolve_image(image_root, str(record["image"]))
            _resolve_image(image_root, str(record["m_loc"]))
            image_resolved = True
        except Exception:
            image_resolved = False
        x_minus_nonempty = bool(record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc"))
        metrics = _evaluate_current(
            model,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
        )
        baselines[record_id] = metrics
        old_raw = metrics.get("old_raw")
        new_raw = metrics.get("new_raw")
        ref_raw = metrics.get("reference_raw")
        valid = bool(image_resolved and x_minus_nonempty and _metric_available(old_raw) and _metric_available(new_raw) and _metric_available(ref_raw))
        rows.append(
            {
                "record_id": record_id,
                "valid": valid,
                "old_answer": record.get("old_answer"),
                "new_answer": record.get("new_answer"),
                "reference_answer": record.get("m_loc_a"),
                "image_path_resolved": image_resolved,
                "x_minus_nonempty": x_minus_nonempty,
                "old_answer_nll": old_raw.get("nll") if isinstance(old_raw, dict) else None,
                "old_answer_logprob": old_raw.get("logprob") if isinstance(old_raw, dict) else None,
                "old_answer_token_count": old_raw.get("num_tokens") if isinstance(old_raw, dict) else None,
                "new_answer_nll": new_raw.get("nll") if isinstance(new_raw, dict) else None,
                "new_answer_token_count": new_raw.get("num_tokens") if isinstance(new_raw, dict) else None,
                "reference_nll": ref_raw.get("nll") if isinstance(ref_raw, dict) else None,
                "reference_token_count": ref_raw.get("num_tokens") if isinstance(ref_raw, dict) else None,
                "generation_empty": metrics.get("generation", {}).get("generation_empty") if isinstance(metrics.get("generation"), dict) else None,
                "failure_reason": None if valid else "failed image/new-answer/reference/token/finite pre-edit filter",
            }
        )
    valid_rows = [row for row in rows if row["valid"]]
    old_values = sorted(float(row["old_answer_nll"]) for row in valid_rows)
    old_median = old_values[len(old_values) // 2] if old_values else None
    valid_rows_sorted = sorted(valid_rows, key=lambda row: (float(row["old_answer_nll"]), -float(row["new_answer_nll"])))
    selected_ids = {row["record_id"] for row in valid_rows_sorted[:record_count]}
    selected_records = [record for record in candidate_records if str(record["id"]) in selected_ids]
    selected_records.sort(key=lambda record: sorted(selected_ids).index(str(record["id"])))
    selected_baselines = {str(record["id"]): baselines[str(record["id"])] for record in selected_records}
    report = {
        "status": "pass" if len(selected_records) == int(record_count) else "fail",
        "candidate_count": len(candidate_records),
        "valid_count": len(valid_rows),
        "required_record_count": int(record_count),
        "selected_record_ids": [str(record["id"]) for record in selected_records],
        "selected_rows": [row for row in rows if row["record_id"] in selected_ids],
        "rows": rows,
        "selection_rule": "valid finite old/new/reference NLL records sorted by lowest old_answer_nll, then higher new_answer_nll",
        "old_answer_nll_median_among_valid": old_median,
        "generation_evidence": "skipped_or_weak; not used as pass/fail",
    }
    _json_dump(out_dir / "data_filter_report.json", report)
    if report["status"] != "pass":
        _write_stop_report(out_dir, "data_filter_failed", report)
        return [], {}, report
    final_path = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw" / "engram_replacement_20edit_modelknown.json"
    final_path.write_text(json.dumps(selected_records, indent=2, sort_keys=True), encoding="utf-8")
    replacement_summary = {
        "status": "pass",
        "replacement_data_file": str(final_path),
        "image_root": str(image_root),
        "record_count": len(selected_records),
        "record_ids": [str(record["id"]) for record in selected_records],
        "records": [
            {
                "record_id": str(record["id"]),
                "old_answer": record.get("old_answer"),
                "new_answer": record.get("new_answer"),
                "x_minus_non_empty": bool(record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc")),
                "image_paths_resolve": True,
            }
            for record in selected_records
        ],
        "private_or_patient_data_used": False,
        "original_data_modified": False,
    }
    _json_dump(out_dir / "replacement_data_summary.json", replacement_summary)
    return selected_records, selected_baselines, report


def _write_stop_report(out_dir: Path, reason: str, payload: Dict[str, Any]) -> None:
    lines = [
        "# Final CURE 20-Edit Model-Known Report",
        "",
        f"- Status: `stopped`",
        f"- Stop reason: `{reason}`",
        "- No editing run was launched after this stop condition.",
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
        "- Non-PHI engineering gate only.",
        "- No downstream generation or public benchmark validation was launched.",
        "- No medical or clinical efficacy claim.",
    ]
    (out_dir / "FINAL_CURE_20EDIT_MODELKNOWN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_nonseq(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method]
    new_values = [float(row["new_answer_nll_decrease"]) for row in metric_rows if row.get("new_answer_nll_decrease") is not None]
    old_values = [float(row["old_answer_nll_increase"]) for row in metric_rows if row.get("old_answer_nll_increase") is not None]
    ref_values = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mask_values = [float(row["mask_keep_ratio"]) for row in metric_rows if row.get("mask_keep_ratio") is not None]
    projection_values = [float(row["projection_norm_ratio"]) for row in metric_rows if row.get("projection_norm_ratio") is not None]
    candidate_norms = [float(row["delta_candidate_norm"]) for row in metric_rows if row.get("delta_candidate_norm") is not None]
    engram_norms = [float(row["delta_engram_norm"]) for row in metric_rows if row.get("delta_engram_norm") is not None]
    cure_norms = [float(row["delta_cure_norm"]) for row in metric_rows if row.get("delta_cure_norm") is not None]
    mean_new = _mean(new_values)
    mean_ref = _mean(ref_values)
    return {
        "method": method,
        "record_count": len(metric_rows),
        "beta": BEST_CURE["beta"] if method != "A_no_edit" else 0.0,
        "crisp_energy_threshold": BEST_CURE["crisp_energy_threshold"] if method == "E_cure_dual_projected_tiny_lora" else None,
        "crisp_cache_update_policy": BEST_CURE["crisp_cache_update_policy"] if method == "E_cure_dual_projected_tiny_lora" else None,
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
        "mean_mask_keep_ratio": _mean(mask_values),
        "mean_projection_norm_ratio": _mean(projection_values),
        "mean_delta_candidate_norm": _mean(candidate_norms),
        "mean_delta_engram_norm": _mean(engram_norms),
        "mean_delta_cure_norm": _mean(cure_norms),
        "skipped_module_count": sum(len(row.get("skipped_modules") or []) for row in metric_rows),
    }


def _run_nonseq_20(
    *,
    model: torch.nn.Module,
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
) -> Dict[str, Any]:
    nonseq_dir = out_dir / "nonseq"
    nonseq_dir.mkdir(parents=True, exist_ok=True)
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    record_id_match_rate = 1.0 if matching.get("mode") == "record_id" else 0.0
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    cache_reports: List[Dict[str, Any]] = []
    beta = float(BEST_CURE["beta"])
    threshold = float(BEST_CURE["crisp_energy_threshold"])

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
            cache_result = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
            cache_reports.append(
                {
                    "record_id": str(record["id"]),
                    "status": cache_result.get("status"),
                    "num_samples": cache_result.get("num_samples"),
                    "diagnostics": cache_result.get("diagnostics", []),
                }
            )
            projection_caches = _projection_caches_for_thresholds_from_kfac(
                cache_result.get("layer_to_cache", {}),
                hparams,
                [threshold],
            ).get(threshold, {})
            cure_entries, cure_summary = _apply_crisp_to_factors(engram_factors, projection_caches)
            rows.append(
                _make_eval_row(
                    method="A_no_edit",
                    record=record,
                    case_index=idx,
                    before=baselines[str(record["id"])],
                    after=baselines[str(record["id"])],
                    rollback_diff=0.0,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    record_id_match_rate=1.0,
                    beta=0.0,
                    extra={"selected_modules": EXPECTED_MODULES, "projection_metadata": {}, "skipped_modules": [], "skip_reasons": {}},
                )
            )
            rows.append(
                _evaluate_patch(
                    model=model,
                    patch=EvalLoraPatch(model, factors, beta=beta),
                    method="B_tiny_lora_replacement",
                    record=record,
                    idx=idx,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    beta=beta,
                    threshold=threshold,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                    extra={"lora_train": train_summary},
                )
            )
            rows.append(
                _evaluate_patch(
                    model=model,
                    patch=EvalLoraPatch(model, engram_factors, beta=beta),
                    method="C_engram_projected_tiny_lora",
                    record=record,
                    idx=idx,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    beta=beta,
                    threshold=threshold,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                    extra={"engram_projection": engram_summary},
                )
            )
            rows.append(
                _evaluate_patch(
                    model=model,
                    patch=EvalMixedDeltaPatch(model, cure_entries, beta=beta),
                    method="E_cure_dual_projected_tiny_lora",
                    record=record,
                    idx=idx,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    beta=beta,
                    threshold=threshold,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                    extra={
                        "engram_projection": engram_summary,
                        "crisp_projection": cure_summary,
                        "crisp_cache_update_policy": BEST_CURE["crisp_cache_update_policy"],
                    },
                )
            )
            _restore_modules(model, snapshots)
    finally:
        _restore_modules(model, snapshots)

    _attach_nonseq_diagnostics(rows)
    aggregates = [_aggregate_nonseq(rows, method) for method in METHODS]
    by_method = {row["method"]: row for row in aggregates}
    cure = by_method.get("E_cure_dual_projected_tiny_lora", {})
    c_base = by_method.get("C_engram_projected_tiny_lora", {})
    checks = {
        "positive_new_answer_edits_at_least_16_of_20": int(cure.get("positive_new_answer_edits") or 0) >= 16,
        "mean_new_answer_nll_decrease_positive": cure.get("mean_new_answer_nll_decrease") is not None and float(cure["mean_new_answer_nll_decrease"]) > 0.0,
        "mean_reference_delta_abs_less_than_mean_new_decrease": (
            cure.get("mean_reference_delta_abs") is not None
            and cure.get("mean_new_answer_nll_decrease") is not None
            and float(cure["mean_reference_delta_abs"]) < float(cure["mean_new_answer_nll_decrease"])
        ),
        "rollback_pass_rate_is_1": float(cure.get("rollback_pass_rate") or 0.0) == 1.0,
        "record_id_match_rate_is_1": float(cure.get("record_id_match_rate") or 0.0) == 1.0,
        "no_nan_inf": int(cure.get("nan_inf_count") or 0) == 0,
        "locality_damage_lte_c_baseline": (
            c_base.get("locality_damage_edits") is None
            or int(cure.get("locality_damage_edits") or 0) <= int(c_base.get("locality_damage_edits") or 0)
        ),
        "mean_reference_delta_abs_lte_c_baseline": (
            c_base.get("mean_reference_delta_abs") is None
            or cure.get("mean_reference_delta_abs") is None
            or float(cure["mean_reference_delta_abs"]) <= float(c_base["mean_reference_delta_abs"])
        ),
    }
    acceptance = {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "cure": cure, "c_baseline": c_base}
    payload = {
        "status": "complete",
        "best_cure_config": BEST_CURE,
        "edit_record_matching": matching,
        "cache_reports": cache_reports,
        "per_record": rows,
        "aggregate_rows": aggregates,
        "acceptance": acceptance,
    }
    _json_dump(nonseq_dir / "nonseq_results.json", payload)
    _write_csv(nonseq_dir / "nonseq_results.csv", rows)
    _write_csv(nonseq_dir / "nonseq_aggregates.csv", aggregates)
    _json_dump(nonseq_dir / "crisp_cache_nonseq_summary.json", cache_reports)
    return payload


def _aggregate_sequential_step_20(rows: List[Dict[str, Any]], method: str, step: int) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method and int(row.get("step") or 0) == int(step)]
    edited_rows = [row for row in metric_rows if row.get("is_edited_so_far")]
    previous_rows = [row for row in metric_rows if row.get("previous_edit_retention") is not None]
    future_rows = [row for row in metric_rows if row.get("is_future_edit")]
    new_values = [float(row["new_answer_nll_decrease_vs_step0"]) for row in edited_rows if row.get("new_answer_nll_decrease_vs_step0") is not None]
    ref_values = [float(row["reference_delta_abs_vs_step0"]) for row in metric_rows if row.get("reference_delta_abs_vs_step0") is not None]
    prev_values = [float(row["previous_edit_retention"]) for row in previous_rows if row.get("previous_edit_retention") is not None]
    future_values = [float(row["future_record_drift"]) for row in future_rows if row.get("future_record_drift") is not None]
    return {
        "method": method,
        "step": int(step),
        "record_count": len(metric_rows),
        "edited_record_count": len(edited_rows),
        "mean_new_answer_nll_decrease": _mean(new_values),
        "mean_reference_delta_abs": _mean(ref_values),
        "previous_edit_retention": _mean(prev_values),
        "mean_future_record_drift": _mean(future_values),
        "positive_new_answer_edits": sum(1 for value in new_values if value > 0.0),
        "locality_damage_records": sum(1 for row in metric_rows if row.get("locality_damage")),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows if row.get("rollback_pass") is not None]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
    }


def _projection_diagnostics_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    ratios: List[float] = []
    projection_ratios: List[float] = []
    candidate_norms: List[float] = []
    engram_norms: List[float] = []
    cure_norms: List[float] = []
    skipped = set()
    modules = set()
    skip_reasons: Dict[str, Any] = {}
    accumulated = []
    for row in trace:
        modules.update(row.get("modules_with_cache") or [])
        accumulated.append(int(row.get("accumulated_num_samples") or 0))
        for value in (row.get("mask_keep_ratios") or {}).values():
            if value is not None:
                ratios.append(float(value))
        for module in row.get("skipped_modules") or []:
            if module:
                skipped.add(module)
        skip_reasons.update(row.get("skip_reasons") or {})
        engram = row.get("engram_projection") or {}
        crisp = row.get("crisp_projection") or {}
        if engram.get("candidate_delta_norm_total") is not None:
            candidate_norms.append(float(engram["candidate_delta_norm_total"]))
        if engram.get("projected_delta_norm_total") is not None:
            engram_norms.append(float(engram["projected_delta_norm_total"]))
        if crisp.get("projected_delta_norm_total") is not None:
            cure_norms.append(float(crisp["projected_delta_norm_total"]))
        if crisp.get("projection_norm_ratio_total") is not None:
            projection_ratios.append(float(crisp["projection_norm_ratio_total"]))
        crisp_modules = crisp.get("modules") or []
        module_cure_norm = _norm_total(item.get("projected_delta_norm") for item in crisp_modules)
        if module_cure_norm is not None:
            cure_norms.append(module_cure_norm)
        for item in crisp_modules:
            if item.get("projection_norm_ratio") is not None:
                projection_ratios.append(float(item["projection_norm_ratio"]))
    return {
        "cache_update_policy": BEST_CURE["crisp_cache_update_policy"],
        "average_mask_keep_ratio": _mean(ratios),
        "projection_norm_ratio": _mean(projection_ratios),
        "delta_candidate_norm": _mean(candidate_norms),
        "delta_engram_norm": _mean(engram_norms),
        "delta_cure_norm": _mean(cure_norms),
        "modules_with_cache": sorted(modules),
        "skipped_modules": sorted(skipped),
        "skip_reasons": skip_reasons,
        "accumulated_num_samples": max(accumulated) if accumulated else None,
    }


def _projection_diagnostics_from_nonseq_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == "E_cure_dual_projected_tiny_lora"]
    skipped = sorted({module for row in metric_rows for module in (row.get("skipped_modules") or [])})
    skip_reasons: Dict[str, Any] = {}
    for row in metric_rows:
        skip_reasons.update(row.get("skip_reasons") or {})
    return {
        "cache_update_policy": BEST_CURE["crisp_cache_update_policy"],
        "record_count": len(metric_rows),
        "average_mask_keep_ratio": _mean([float(row["mask_keep_ratio"]) for row in metric_rows if row.get("mask_keep_ratio") is not None]),
        "projection_norm_ratio": _mean([float(row["projection_norm_ratio"]) for row in metric_rows if row.get("projection_norm_ratio") is not None]),
        "delta_candidate_norm": _mean([float(row["delta_candidate_norm"]) for row in metric_rows if row.get("delta_candidate_norm") is not None]),
        "delta_engram_norm": _mean([float(row["delta_engram_norm"]) for row in metric_rows if row.get("delta_engram_norm") is not None]),
        "delta_cure_norm": _mean([float(row["delta_cure_norm"]) for row in metric_rows if row.get("delta_cure_norm") is not None]),
        "skipped_modules": skipped,
        "skipped_module_count": sum(len(row.get("skipped_modules") or []) for row in metric_rows),
        "unique_skipped_module_count": len(skipped),
        "skip_reasons": skip_reasons,
        "identity_like_warning": (
            "mask_keep_ratio near 1.0; curvature projection may be close to identity-like on this gate"
            if metric_rows
            and _mean([float(row["mask_keep_ratio"]) for row in metric_rows if row.get("mask_keep_ratio") is not None]) is not None
            and float(_mean([float(row["mask_keep_ratio"]) for row in metric_rows if row.get("mask_keep_ratio") is not None])) >= 0.99
            else None
        ),
    }


def _write_projection_diagnostics(out_dir: Path, nonseq: Optional[Dict[str, Any]], sequential: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    nonseq_diag = _projection_diagnostics_from_nonseq_rows((nonseq or {}).get("per_record", []))
    sequential_diag = (sequential or {}).get("projection_diagnostics") or {}
    payload = {
        "best_cure_config": BEST_CURE,
        "nonseq": nonseq_diag,
        "sequential": sequential_diag,
        "curvature_overclaim_guard": {
            "mask_keep_ratio_near_one_nonseq": nonseq_diag.get("average_mask_keep_ratio") is not None and float(nonseq_diag["average_mask_keep_ratio"]) >= 0.99,
            "mask_keep_ratio_near_one_sequential": sequential_diag.get("average_mask_keep_ratio") is not None
            and float(sequential_diag["average_mask_keep_ratio"]) >= 0.99,
            "projection_norm_ratio_available_nonseq": nonseq_diag.get("projection_norm_ratio") is not None,
            "projection_norm_ratio_available_sequential": sequential_diag.get("projection_norm_ratio") is not None,
            "skipped_modules_nonseq": nonseq_diag.get("skipped_modules"),
            "skipped_modules_sequential": sequential_diag.get("skipped_modules"),
        },
    }
    _json_dump(out_dir / "projection_diagnostics.json", payload)
    return payload


def _run_sequential_20(
    *,
    model: torch.nn.Module,
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
) -> Dict[str, Any]:
    seq_dir = out_dir / "sequential"
    seq_dir.mkdir(parents=True, exist_ok=True)
    bank = EngramBank(projector_bank_dir)
    _, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    record_id_match_rate = 1.0 if matching.get("mode") == "record_id" else 0.0
    rows: List[Dict[str, Any]] = []
    rollback_checks: List[Dict[str, Any]] = []
    cache_trace: List[Dict[str, Any]] = []
    for method in ["C_engram_projected_tiny_lora", "E_cure_dual_projected_tiny_lora"]:
        method_rows, rollback_payload, method_cache_trace = _run_one_sequential_method(
            model=model,
            method=method,
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=projector_bank_dir,
            module_names=module_names,
            hparams=hparams,
            beta=float(BEST_CURE["beta"]),
            threshold=float(BEST_CURE["crisp_energy_threshold"]),
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            record_id_match_rate=record_id_match_rate,
            crisp_cache_update_policy=str(BEST_CURE["crisp_cache_update_policy"]),
        )
        for row in method_rows:
            row["crisp_cache_update_policy"] = BEST_CURE["crisp_cache_update_policy"] if method.startswith("E_") else None
            row["old_answer_nll_delta_vs_step0"] = row.get("old_answer_nll_delta_vs_step0")
        rows.extend(method_rows)
        rollback_checks.append(rollback_payload)
        cache_trace.extend(method_cache_trace)

    summary_rows = [
        _aggregate_sequential_step_20(rows, method, step)
        for method in ["C_engram_projected_tiny_lora", "E_cure_dual_projected_tiny_lora"]
        for step in range(0, len(records) + 1)
    ]
    final_rows = [row for row in summary_rows if int(row.get("step") or 0) == len(records)]
    by_method = {row["method"]: row for row in final_rows}
    cure = by_method.get("E_cure_dual_projected_tiny_lora", {})
    c_base = by_method.get("C_engram_projected_tiny_lora", {})
    new_ratio = _safe_div(cure.get("mean_new_answer_nll_decrease"), c_base.get("mean_new_answer_nll_decrease"))
    ref_ratio = _safe_div(cure.get("mean_reference_delta_abs"), c_base.get("mean_reference_delta_abs"))
    retention_ratio = _safe_div(cure.get("previous_edit_retention"), c_base.get("previous_edit_retention"))
    rollback_payload = {
        "status": "pass" if all(item.get("rollback_pass") for item in rollback_checks) else "fail",
        "rollback_tolerance": rollback_tolerance,
        "methods": rollback_checks,
    }
    basic_checks = {
        "positive_new_answer_edits_at_least_16_of_20": int(cure.get("positive_new_answer_edits") or 0) >= 16,
        "mean_new_answer_nll_decrease_positive": cure.get("mean_new_answer_nll_decrease") is not None and float(cure["mean_new_answer_nll_decrease"]) > 0.0,
        "mean_reference_delta_abs_less_than_mean_new_decrease": (
            cure.get("mean_reference_delta_abs") is not None
            and cure.get("mean_new_answer_nll_decrease") is not None
            and float(cure["mean_reference_delta_abs"]) < float(cure["mean_new_answer_nll_decrease"])
        ),
        "locality_damage_records_lte_2": int(cure.get("locality_damage_records") or 0) <= 2,
        "rollback_pass": rollback_payload["status"] == "pass",
        "record_id_match_rate_is_1": float(cure.get("record_id_match_rate") or 0.0) == 1.0,
        "no_nan_inf": int(cure.get("nan_inf_count") or 0) == 0,
    }
    relative_checks = {
        "new_answer_ratio_gte_0p95": new_ratio is not None and float(new_ratio) >= 0.95,
        "retention_ratio_gte_0p95": retention_ratio is not None and float(retention_ratio) >= 0.95,
        "reference_ratio_lte_0p85": ref_ratio is not None and float(ref_ratio) <= 0.85,
    }
    if all(basic_checks.values()) and all(relative_checks.values()):
        status = "pass"
    elif all(basic_checks.values()):
        status = "partial"
    else:
        status = "fail"
    diagnostics = _projection_diagnostics_from_trace([row for row in cache_trace if row.get("method") == "E_cure_dual_projected_tiny_lora"])
    payload = {
        "status": status,
        "best_cure_config": BEST_CURE,
        "edit_record_matching": matching,
        "per_record_step_rows": rows,
        "summary_rows": summary_rows,
        "final_rows": final_rows,
        "relative_to_c": {
            "new_answer_ratio": new_ratio,
            "reference_ratio": ref_ratio,
            "retention_ratio": retention_ratio,
        },
        "acceptance": {
            "status": status,
            "basic_checks": basic_checks,
            "relative_checks": relative_checks,
            "cure_final": cure,
            "c_baseline_final": c_base,
        },
        "final_rollback_check": rollback_payload,
        "crisp_cache_update_trace": cache_trace,
        "projection_diagnostics": diagnostics,
    }
    _json_dump(seq_dir / "sequential_step_matrix.json", rows)
    _write_csv(seq_dir / "sequential_step_matrix.csv", rows)
    _json_dump(seq_dir / "sequential_summary.json", payload)
    _write_csv(seq_dir / "sequential_summary.csv", summary_rows)
    _json_dump(seq_dir / "final_rollback_check.json", rollback_payload)
    _json_dump(seq_dir / "crisp_cache_update_trace.json", cache_trace)
    _json_dump(seq_dir / "projection_diagnostics.json", diagnostics)
    return payload


def _write_record_id_preflight(out_dir: Path, records: List[Dict[str, Any]], projector_bank_dir: Path) -> Dict[str, Any]:
    bank = EngramBank(projector_bank_dir)
    try:
        _, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
        match_rate = 1.0 if matching.get("mode") == "record_id" else 0.0
        positional_refused_by_default = True
        status = "pass" if match_rate == 1.0 else "fail"
    except Exception as exc:
        matching = {"error": f"{type(exc).__name__}: {exc}"}
        match_rate = 0.0
        positional_refused_by_default = True
        status = "fail"
    payload = {
        "status": status,
        "raw_records_have_record_id": sum(1 for record in records if record.get("id") or record.get("record_id")),
        "record_count": len(records),
        "record_id_match_rate": match_rate,
        "positional_matching_used": False,
        "positional_matching_refused_by_default": positional_refused_by_default,
        "bank_matching": matching,
    }
    _json_dump(out_dir / "record_id_preflight.json", payload)
    return payload


def _write_final_report(
    out_dir: Path,
    *,
    data_report: Dict[str, Any],
    nonseq: Optional[Dict[str, Any]],
    sequential: Optional[Dict[str, Any]],
    generation: Dict[str, Any],
    projection_diagnostics: Dict[str, Any],
) -> None:
    nonseq_rows = (nonseq or {}).get("aggregate_rows", [])
    seq_final = (sequential or {}).get("final_rows", [])
    nonseq_status = (nonseq or {}).get("acceptance", {}).get("status", "not_run")
    sequential_status = (sequential or {}).get("status", "not_run")
    decision = "C. CURE fails 20-edit. Do not scale; inspect data filtering, beta/gamma/cache policy, or projection order."
    if nonseq_status == "pass" and sequential_status == "pass":
        decision = "A. CURE passes 20-edit nonseq and sequential gates and remains Pareto-promising. Next: non-empty generation / public medical benchmark validation."
    elif nonseq_status == "pass" and sequential_status in {"partial", "pass"}:
        decision = "B. CURE passes 20-edit but no longer clearly improves over ENGRAM-projected LoRA. Keep both C and E for 20-edit; treat CURE as conservative variant."
    lines = [
        "# Final CURE 20-Edit Model-Known Report",
        "",
        "## Motivation",
        "",
        "- Starting point: 5-edit CURE Pareto-promising config `E_beta0.5_gamma0.5_streaming`.",
        "- 10-edit model-known non-PHI gate passed with record-id matching `1.0` and positive CURE edits `10/10`.",
        "- This gate tests whether the same configuration survives a 20-edit model-known non-PHI validation.",
        "",
        "## Data",
        "",
        f"- Candidate count: `{data_report.get('candidate_count')}`",
        f"- Valid count: `{data_report.get('valid_count')}`",
        f"- Selected record IDs: `{data_report.get('selected_record_ids')}`",
        "- Data source: synthetic non-PHI engineering fixtures; no private or patient data.",
        "- Filtering used finite old-answer, new-answer, and reference NLL plus resolved image/reference paths.",
        "",
        "| record_id | old_answer | new_answer | reference_answer |",
        "|---|---|---|---|",
    ]
    for row in data_report.get("selected_rows") or []:
        lines.append(
            f"| {row.get('record_id')} | {row.get('old_answer')} | {row.get('new_answer')} | {row.get('reference_answer')} |"
        )
    lines.extend(
        [
            "",
            "## Methods",
            "",
            "- `A_no_edit`",
            "- `B_tiny_lora_replacement`",
            "- `C_engram_projected_tiny_lora`",
            "- `E_cure_dual_projected_tiny_lora`",
            "- Direct ENGRAM erase is used only as a prior recorded failure baseline; it was not rerun.",
            "",
            "## Non-Sequential Results",
            "",
            f"- Acceptance: `{nonseq_status}`",
            "",
            "| method | mean new decrease | mean old increase | mean reference delta | positive new | positive old erase | locality damage | rollback | match | nan/inf | mean mask | mean proj ratio | skipped modules |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nonseq_rows:
        lines.append(
            "| {method} | {new} | {old} | {ref} | {pos_new} | {pos_old} | {loc} | {roll} | {match} | {nan} | {mask} | {proj} | {skip} |".format(
                method=row.get("method"),
                new=_format(row.get("mean_new_answer_nll_decrease")),
                old=_format(row.get("mean_old_answer_nll_increase")),
                ref=_format(row.get("mean_reference_delta_abs")),
                pos_new=row.get("positive_new_answer_edits"),
                pos_old=row.get("positive_old_answer_erasure_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
                mask=_format(row.get("mean_mask_keep_ratio")),
                proj=_format(row.get("mean_projection_norm_ratio")),
                skip=row.get("skipped_module_count"),
            )
        )
    lines.extend(["", "## Sequential Results", "", f"- Status: `{sequential_status}`", ""])
    if seq_final:
        lines.extend(
            [
                "| method | mean new decrease | mean reference delta | retention | positive new | locality damage | rollback | match | nan/inf |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in seq_final:
            lines.append(
                "| {method} | {new} | {ref} | {ret} | {pos} | {loc} | {roll} | {match} | {nan} |".format(
                    method=row.get("method"),
                    new=_format(row.get("mean_new_answer_nll_decrease")),
                    ref=_format(row.get("mean_reference_delta_abs")),
                    ret=_format(row.get("previous_edit_retention")),
                    pos=row.get("positive_new_answer_edits"),
                    loc=row.get("locality_damage_records"),
                    roll=_format(row.get("rollback_pass_rate")),
                    match=_format(row.get("record_id_match_rate")),
                    nan=row.get("nan_inf_count"),
                )
            )
        rel = (sequential or {}).get("relative_to_c", {})
        lines.extend(
            [
                "",
                "## Relative Comparison",
                "",
                f"- new_answer_ratio: `{_format(rel.get('new_answer_ratio'))}`",
                f"- reference_ratio: `{_format(rel.get('reference_ratio'))}`",
                f"- retention_ratio: `{_format(rel.get('retention_ratio'))}`",
            ]
        )
    else:
        lines.append("- Sequential stage was skipped because nonseq CURE did not pass.")
    diag = (sequential or {}).get("projection_diagnostics", {})
    nonseq_diag = (projection_diagnostics or {}).get("nonseq", {})
    guard = (projection_diagnostics or {}).get("curvature_overclaim_guard", {})
    curvature_note = "projection diagnostics do not support a strong curvature contribution claim"
    if not (guard.get("mask_keep_ratio_near_one_nonseq") or guard.get("mask_keep_ratio_near_one_sequential")):
        curvature_note = "mask diagnostics are not near identity-like in this gate"
    lines.extend(
        [
            "",
            "## Crisp Diagnostics",
            "",
            f"- cache policy: `{BEST_CURE['crisp_cache_update_policy']}`",
            f"- nonseq average mask keep ratio: `{_format(nonseq_diag.get('average_mask_keep_ratio'))}`",
            f"- nonseq projection norm ratio: `{_format(nonseq_diag.get('projection_norm_ratio'))}`",
            f"- nonseq skipped modules: `{nonseq_diag.get('skipped_modules')}`",
            f"- sequential average mask keep ratio: `{_format(diag.get('average_mask_keep_ratio'))}`",
            f"- sequential projection norm ratio: `{_format(diag.get('projection_norm_ratio'))}`",
            f"- skipped modules: `{diag.get('skipped_modules')}`",
            f"- curvature interpretation: `{curvature_note}`",
            "",
            "## Generation Diagnostics",
            "",
            f"- Status: `{generation.get('status')}`",
            "- Main gate used `--skip-generation`; evidence is NLL/logprob-based.",
            "",
            "## Limitations",
            "",
            "- Non-PHI 20-edit only.",
            "- No clinical or medical efficacy claim.",
            "- Delta-space Crisp projection is not original CrispEdit gradient-projected training.",
            "- Generation-level efficacy is not established by this skipped-generation gate.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_CURE_20EDIT_MODELKNOWN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 20-edit model-known non-PHI CURE-MedEdit gate.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml")
    parser.add_argument("--source-data", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json")
    parser.add_argument("--source-image-root", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/images")
    parser.add_argument("--output-dir", default="outputs/cure_mededit_20edit_modelknown")
    parser.add_argument("--sequential-report", default="outputs/engram_sequential_5edit_smoke/FINAL_SEQUENTIAL_5EDIT_SMOKE_REPORT.md")
    parser.add_argument("--best-direct-config", default="outputs/engram_token_module_ablation_5edit/best_overall_config.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--record-count", type=int, default=20)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
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
    _write_failure_summary(out_dir, Path(args.sequential_report), Path(args.best_direct_config))
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)

    candidate_path, image_root, candidate_records, candidate_summary = _build_candidate_records(
        source_data=Path(args.source_data),
        source_image_root=Path(args.source_image_root),
        out_dir=out_dir,
        candidate_count=args.candidate_count,
    )
    preflight = _write_preflight(out_dir, hparams_path=Path(args.hparams), data_path=candidate_path, image_root=image_root, test_status=test_status)
    if preflight["status"] != "pass":
        _write_stop_report(out_dir, "preflight_failed", preflight)
        return 0

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(hparams, image_root=image_root, bank_dir=out_dir / "projector_bank", device=args.device, edit_mode="erase")
    hparams.replacement_mode = "cure_delta_projected"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.use_crisp_projection = True
    hparams.crisp_energy_threshold = float(BEST_CURE["crisp_energy_threshold"])
    hparams.crisp_cache_update_policy = str(BEST_CURE["crisp_cache_update_policy"])
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

    records, baselines, data_report = _filter_model_known_records(
        model=editor.model,
        candidate_records=candidate_records,
        image_root=image_root,
        out_dir=out_dir,
        record_count=args.record_count,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
    )
    data_report.update(candidate_summary)
    _json_dump(out_dir / "data_filter_report.json", data_report)
    if not records:
        return 0
    _json_dump(out_dir / "baseline_metrics.json", baselines)

    data_file = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw" / "engram_replacement_20edit_modelknown.json"
    projector_extract = _extract_projector_bank(editor, hparams, data_file, records, out_dir / "projector_bank")
    _json_dump(out_dir / "projector_extraction_summary.json", projector_extract)
    record_id_preflight = _write_record_id_preflight(out_dir, records, out_dir / "projector_bank")
    if record_id_preflight["status"] != "pass":
        _write_stop_report(out_dir, "record_id_preflight_failed", record_id_preflight)
        return 0

    nonseq = _run_nonseq_20(
        model=editor.model,
        records=records,
        image_root=image_root,
        baselines=baselines,
        projector_bank_dir=out_dir / "projector_bank",
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        out_dir=out_dir,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=bool(args.skip_generation),
    )

    sequential: Optional[Dict[str, Any]] = None
    if nonseq.get("acceptance", {}).get("status") == "pass":
        sequential = _run_sequential_20(
            model=editor.model,
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=out_dir / "projector_bank",
            module_names=EXPECTED_MODULES,
            hparams=hparams,
            out_dir=out_dir,
            rollback_tolerance=args.rollback_tolerance,
            locality_threshold=args.locality_damage_threshold,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=bool(args.skip_generation),
        )
    else:
        seq_dir = out_dir / "sequential"
        seq_dir.mkdir(parents=True, exist_ok=True)
        sequential = {"status": "skipped", "reason": "nonseq CURE failed", "nonseq_acceptance": nonseq.get("acceptance")}
        _json_dump(seq_dir / "sequential_skipped.json", sequential)

    generation = {"status": "skipped", "reason": "main gate uses --skip-generation; optional generation diagnostics not launched"}
    _json_dump(out_dir / "generation_diagnostics.json", generation)
    projection_diagnostics = _write_projection_diagnostics(out_dir, nonseq, sequential)
    _write_final_report(
        out_dir,
        data_report=data_report,
        nonseq=nonseq,
        sequential=sequential,
        generation=generation,
        projection_diagnostics=projection_diagnostics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
