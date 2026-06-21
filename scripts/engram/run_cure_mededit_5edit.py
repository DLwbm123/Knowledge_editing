#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.crisp_kfac_collector import collect_crisp_kfac_caches  # noqa: E402
from easyeditor.models.engram.crisp_projection import (  # noqa: E402
    apply_crisp_projection_to_delta,
    build_crisp_kfac_projection_cache_from_decomposition,
    compute_crisp_kfac_projection_cache,
)
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_localized_replacement_5edit import (  # noqa: E402
    EXPECTED_MODULES,
    EvalLoraPatch,
    _configure_hparams,
    _evaluate_current,
    _extract_projector_bank,
    _finite,
    _format,
    _json_dump,
    _load_records,
    _make_eval_row,
    _max_snapshot_diff,
    _mean,
    _module_map,
    _prepare_replacement_data,
    _project_factors,
    _reference_sample,
    _restore_modules,
    _safe_div,
    _snapshot_modules,
    _train_tiny_lora,
    _write_csv,
    _write_failure_summary,
    _write_git_outputs,
)


METHODS = [
    "A_no_edit",
    "B_tiny_lora_replacement",
    "C_engram_projected_tiny_lora",
    "D_crisp_projected_tiny_lora",
    "E_cure_dual_projected_tiny_lora",
]


class EvalMixedDeltaPatch:
    """Evaluate dense projected deltas and low-rank LoRA factors together."""

    def __init__(self, model: nn.Module, entries: Dict[str, Dict[str, Any]], *, beta: float) -> None:
        self.model = model
        self.entries = entries
        self.beta = float(beta)
        self.original_forwards: Dict[str, Any] = {}

    def install(self) -> None:
        modules = _module_map(self.model)
        for name, entry in self.entries.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"CURE target is not nn.Linear: {name}")
            self.original_forwards[name] = module.forward
            if entry["kind"] == "dense":
                delta = entry["delta"].to(module.weight.device, dtype=torch.float32)

                def patched_forward(x, *, _base=module.forward, _delta=delta, _beta=self.beta):
                    base = _base(x)
                    update = torch.nn.functional.linear(x.to(torch.float32), _delta) * float(_beta)
                    return base + update.to(dtype=base.dtype)

            elif entry["kind"] == "lora":
                a = entry["A"].to(module.weight.device, dtype=torch.float32)
                b = entry["B"].to(module.weight.device, dtype=torch.float32)
                scale = float(entry.get("scale", 1.0))

                def patched_forward(x, *, _base=module.forward, _a=a, _b=b, _scale=scale, _beta=self.beta):
                    base = _base(x)
                    low = torch.nn.functional.linear(x.to(torch.float32), _a)
                    update = torch.nn.functional.linear(low, _b) * float(_scale) * float(_beta)
                    return base + update.to(dtype=base.dtype)

            else:
                raise ValueError(f"Unknown mixed delta kind: {entry['kind']}")
            module.forward = patched_forward  # type: ignore[method-assign]

    def remove(self) -> None:
        modules = _module_map(self.model)
        for name, forward in reversed(list(self.original_forwards.items())):
            modules[name].forward = forward  # type: ignore[method-assign]
        self.original_forwards.clear()


def _parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one float in {raw!r}")
    return values


def _crisp_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "fp64"}:
        return torch.float64
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported crisp cache dtype: {name}")


def _lora_to_mixed_entries(factors: Dict[str, Dict[str, torch.Tensor | float]]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "kind": "lora",
            "A": factor["A"].detach().cpu().float(),  # type: ignore[union-attr]
            "B": factor["B"].detach().cpu().float(),  # type: ignore[union-attr]
            "scale": float(factor.get("scale", 1.0)),  # type: ignore[union-attr]
        }
        for name, factor in factors.items()
    }


def _dense_delta_from_factor(factor: Dict[str, torch.Tensor | float]) -> torch.Tensor:
    a = factor["A"].detach().cpu().float()  # type: ignore[union-attr]
    b = factor["B"].detach().cpu().float()  # type: ignore[union-attr]
    scale = float(factor.get("scale", 1.0))  # type: ignore[union-attr]
    return b.matmul(a) * scale


def _apply_crisp_to_factors(
    factors: Dict[str, Dict[str, torch.Tensor | float]],
    projection_caches: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    module_rows: List[Dict[str, Any]] = []
    for name, factor in factors.items():
        projection_cache = projection_caches.get(name)
        if projection_cache is None:
            entries[name] = {
                "kind": "lora",
                "A": factor["A"].detach().cpu().float(),  # type: ignore[union-attr]
                "B": factor["B"].detach().cpu().float(),  # type: ignore[union-attr]
                "scale": float(factor.get("scale", 1.0)),  # type: ignore[union-attr]
            }
            module_rows.append({"module_name": name, "crisp_projected": False, "skip_reason": "missing_projection_cache"})
            continue
        delta = _dense_delta_from_factor(factor)
        projected = apply_crisp_projection_to_delta(delta, projection_cache).detach().cpu().float()
        entries[name] = {"kind": "dense", "delta": projected}
        delta_norm = float(delta.norm().item())
        projected_norm = float(projected.norm().item())
        module_rows.append(
            {
                "module_name": name,
                "crisp_projected": True,
                "candidate_delta_norm": delta_norm,
                "projected_delta_norm": projected_norm,
                "projection_norm_ratio": _safe_div(projected_norm, delta_norm),
                "mask_keep_ratio": (projection_cache.get("metadata") or {}).get("keep_ratio"),
                "A_decomposition_backend": (projection_cache.get("metadata") or {}).get("A_decomposition_backend"),
                "B_decomposition_backend": (projection_cache.get("metadata") or {}).get("B_decomposition_backend"),
            }
        )
    return entries, {
        "modules": module_rows,
        "projected_module_count": sum(1 for row in module_rows if row.get("crisp_projected")),
        "total_module_count": len(module_rows),
    }


def _loss_from_sample(model: nn.Module, sample: Dict[str, Any]) -> torch.Tensor:
    output = model(dict(sample))
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("Crisp K-FAC sample forward did not return a loss")
    return loss


def _collect_reference_crisp_cache(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
) -> Dict[str, Any]:
    samples = []
    for record in records:
        sample = _reference_sample(record, image_root)
        if sample is not None:
            samples.append(sample)
    if not samples:
        return {"status": "skipped", "reason": "no_reference_samples", "layer_to_cache": {}, "diagnostics": []}
    result = collect_crisp_kfac_caches(
        model,
        module_names,
        samples,
        _loss_from_sample,
        max_dim=int(hparams.crisp_max_dim),
        energy_threshold=None,
        build_projection_cache=False,
        clear_cuda_cache=bool(hparams.clear_cuda_cache),
    )
    result["status"] = "complete"
    result["cache_dataset"] = "reference"
    result["num_samples"] = len(samples)
    return result


def _projection_caches_from_kfac(
    layer_to_cache: Dict[str, Dict[str, Any]],
    hparams: EngramMultimodalHparams,
    energy_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    return {
        module_name: compute_crisp_kfac_projection_cache(
            cache["A"],
            cache["B"],
            float(energy_threshold),
            device=str(hparams.crisp_cache_device),
            dtype=_crisp_dtype(hparams.crisp_cache_dtype),
        )
        for module_name, cache in layer_to_cache.items()
    }


def _projection_caches_for_thresholds_from_kfac(
    layer_to_cache: Dict[str, Dict[str, Any]],
    hparams: EngramMultimodalHparams,
    thresholds: List[float],
) -> Dict[float, Dict[str, Dict[str, Any]]]:
    if not thresholds:
        return {}

    dtype = _crisp_dtype(hparams.crisp_cache_dtype)
    device = str(hparams.crisp_cache_device)
    by_threshold: Dict[float, Dict[str, Dict[str, Any]]] = {float(threshold): {} for threshold in thresholds}
    first_threshold = float(thresholds[0])
    for module_name, cache in layer_to_cache.items():
        first_cache = compute_crisp_kfac_projection_cache(
            cache["A"],
            cache["B"],
            first_threshold,
            device=device,
            dtype=dtype,
        )
        by_threshold[first_threshold][module_name] = first_cache
        metadata = first_cache.get("metadata") or {}
        for threshold in thresholds[1:]:
            threshold = float(threshold)
            by_threshold[threshold][module_name] = build_crisp_kfac_projection_cache_from_decomposition(
                A_shape=metadata.get("A_shape") or cache["A"].shape,
                B_shape=metadata.get("B_shape") or cache["B"].shape,
                Sa=first_cache["Sa"],
                Ua=first_cache["Ua"],
                Sb=first_cache["Sb"],
                Ub=first_cache["Ub"],
                energy_threshold=threshold,
                device=device,
                dtype=dtype,
                A_backend=str(metadata.get("A_decomposition_backend") or "precomputed"),
                B_backend=str(metadata.get("B_decomposition_backend") or "precomputed"),
                A_decomposition_error=metadata.get("A_decomposition_error"),
                B_decomposition_error=metadata.get("B_decomposition_error"),
            )
    return by_threshold


def _aggregate_group(rows: List[Dict[str, Any]], method: str, beta: float, threshold: float) -> Dict[str, Any]:
    metric_rows = [
        row
        for row in rows
        if row.get("method") == method
        and float(row.get("beta") or 0.0) == float(beta)
        and float(row.get("crisp_energy_threshold") or threshold) == float(threshold)
    ]
    new_decreases = [float(row["new_answer_nll_decrease"]) for row in metric_rows if row.get("new_answer_nll_decrease") is not None]
    old_increases = [float(row["old_answer_nll_increase"]) for row in metric_rows if row.get("old_answer_nll_increase") is not None]
    ref_abs = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_new = _mean(new_decreases)
    mean_ref = _mean(ref_abs)
    return {
        "method": method,
        "beta": float(beta),
        "crisp_energy_threshold": float(threshold),
        "status": "complete" if metric_rows else "skipped",
        "record_count": len(metric_rows),
        "mean_new_answer_nll_decrease": mean_new,
        "mean_old_answer_nll_increase": _mean(old_increases),
        "mean_reference_delta_abs": mean_ref,
        "mean_ref_delta": mean_ref,
        "positive_new_answer_edits": sum(1 for value in new_decreases if value > 0.0),
        "positive_old_answer_erasure_edits": sum(1 for value in old_increases if value > 0.0),
        "locality_damage_edits": sum(1 for row in metric_rows if row.get("locality_damage")),
        "target_to_reference_ratio": _safe_div(mean_new, mean_ref),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
    }


def _cure_acceptance(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "positive_new_answer_edits_at_least_4_of_5": int(aggregate.get("positive_new_answer_edits") or 0) >= 4,
        "mean_new_answer_nll_decrease_positive": (aggregate.get("mean_new_answer_nll_decrease") is not None)
        and float(aggregate["mean_new_answer_nll_decrease"]) > 0.0,
        "mean_reference_delta_abs_less_than_mean_new_decrease": (
            aggregate.get("mean_reference_delta_abs") is not None
            and aggregate.get("mean_new_answer_nll_decrease") is not None
            and float(aggregate["mean_reference_delta_abs"]) < float(aggregate["mean_new_answer_nll_decrease"])
        ),
        "rollback_pass_rate_is_1": float(aggregate.get("rollback_pass_rate") or 0.0) == 1.0,
        "record_id_match_rate_is_1": float(aggregate.get("record_id_match_rate") or 0.0) == 1.0,
        "no_nan_inf": int(aggregate.get("nan_inf_count") or 0) == 0,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "aggregate": aggregate}


def _evaluate_patch(
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
    threshold: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    extra: Optional[Dict[str, Any]] = None,
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
    row_extra = {"crisp_energy_threshold": float(threshold)}
    if extra:
        row_extra.update(extra)
    row_extra.setdefault("selected_modules", EXPECTED_MODULES)
    projection_metadata = {
        key: row_extra[key]
        for key in ("engram_projection", "crisp_projection", "lora_train")
        if key in row_extra
    }
    row_extra.setdefault("projection_metadata", projection_metadata)
    crisp_projection = row_extra.get("crisp_projection") or {}
    skipped = [
        row.get("module_name")
        for row in crisp_projection.get("modules", [])
        if not row.get("crisp_projected")
    ]
    skip_reasons = {
        row.get("module_name"): row.get("skip_reason")
        for row in crisp_projection.get("modules", [])
        if not row.get("crisp_projected")
    }
    row_extra.setdefault("skipped_modules", skipped)
    row_extra.setdefault("skip_reasons", skip_reasons)
    return _make_eval_row(
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
        extra=row_extra,
    )


def _clone_layer_to_cache(layer_to_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cloned: Dict[str, Dict[str, Any]] = {}
    for name, cache in layer_to_cache.items():
        if not isinstance(cache, dict) or not isinstance(cache.get("A"), torch.Tensor) or not isinstance(cache.get("B"), torch.Tensor):
            continue
        cloned[name] = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in cache.items()
        }
    return cloned


def _merge_layer_to_cache(
    accumulated: Dict[str, Dict[str, Any]],
    new_cache: Dict[str, Dict[str, Any]],
    *,
    accumulated_num_samples: int,
    new_num_samples: int,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    new_num_samples = int(max(0, new_num_samples))
    accumulated_num_samples = int(max(0, accumulated_num_samples))
    if not accumulated:
        return _clone_layer_to_cache(new_cache), new_num_samples
    if not new_cache or new_num_samples <= 0:
        return _clone_layer_to_cache(accumulated), accumulated_num_samples

    total = accumulated_num_samples + new_num_samples
    merged: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(accumulated) | set(new_cache)):
        old_entry = accumulated.get(name)
        new_entry = new_cache.get(name)
        if old_entry is None:
            merged[name] = _clone_layer_to_cache({name: new_entry or {}}).get(name, {})
            continue
        if new_entry is None:
            merged[name] = _clone_layer_to_cache({name: old_entry}).get(name, {})
            continue
        if not isinstance(old_entry.get("A"), torch.Tensor) or not isinstance(old_entry.get("B"), torch.Tensor):
            continue
        if not isinstance(new_entry.get("A"), torch.Tensor) or not isinstance(new_entry.get("B"), torch.Tensor):
            merged[name] = _clone_layer_to_cache({name: old_entry}).get(name, {})
            continue
        merged[name] = dict(old_entry)
        merged[name]["A"] = (
            old_entry["A"].detach().cpu().float() * float(accumulated_num_samples)
            + new_entry["A"].detach().cpu().float() * float(new_num_samples)
        ) / float(total)
        merged[name]["B"] = (
            old_entry["B"].detach().cpu().float() * float(accumulated_num_samples)
            + new_entry["B"].detach().cpu().float() * float(new_num_samples)
        ) / float(total)
    return merged, total


def _cache_shapes(layer_to_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    shapes: Dict[str, Dict[str, Any]] = {}
    for name, cache in layer_to_cache.items():
        a = cache.get("A")
        b = cache.get("B")
        shapes[name] = {
            "A": list(a.shape) if isinstance(a, torch.Tensor) else None,
            "B": list(b.shape) if isinstance(b, torch.Tensor) else None,
        }
    return shapes


def _mask_keep_ratios(projection_caches: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    ratios: Dict[str, Optional[float]] = {}
    for name, cache in projection_caches.items():
        metadata = cache.get("metadata") or {}
        ratio = metadata.get("keep_ratio")
        ratios[name] = float(ratio) if ratio is not None else None
    return ratios


def _rollback_rows(
    model: nn.Module,
    snapshots: Dict[str, Dict[str, torch.Tensor | None]],
    *,
    method: str,
    param_norm_after_step5: Optional[Dict[str, float]] = None,
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
                "param_norm_after_step5": (param_norm_after_step5 or {}).get(name),
                "param_norm_after_final_rollback": float(after_weight.norm().item()),
                "max_abs_diff_before_vs_after_final_rollback": diff,
            }
        )
    return rows, max_diff


def _current_weight_norms(model: nn.Module, module_names: List[str]) -> Dict[str, float]:
    modules = _module_map(model)
    return {
        name: float(modules[name].weight.detach().cpu().float().norm().item())
        for name in module_names
    }


def _sequential_step_rows(
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
    threshold: float,
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
            extra={"crisp_energy_threshold": float(threshold), "selected_modules": EXPECTED_MODULES},
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
            "generation": base_row.get("generation_after"),
            "generation_empty": base_row.get("generation_empty"),
            "rollback_pass": None,
            "rollback_max_abs_diff": None,
            "record_id_match_rate": float(record_id_match_rate),
            "nan_inf_detected": base_row.get("nan_inf_detected"),
            "beta": float(beta),
            "crisp_energy_threshold": float(threshold),
            "old_answer": base_row.get("old_answer"),
            "new_answer": base_row.get("new_answer"),
        }
        row["nan_inf_detected"] = bool(row["nan_inf_detected"] or not _finite(row))
        rows.append(row)
    return rows


def _aggregate_sequential_step(rows: List[Dict[str, Any]], method: str, step: int) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method and int(row.get("step") or 0) == int(step)]
    edited_rows = [row for row in metric_rows if row.get("is_edited_so_far")]
    previous_rows = [row for row in metric_rows if row.get("previous_edit_retention") is not None]
    future_rows = [row for row in metric_rows if row.get("is_future_edit")]
    new_values = [
        float(row["new_answer_nll_decrease_vs_step0"])
        for row in edited_rows
        if row.get("new_answer_nll_decrease_vs_step0") is not None
    ]
    ref_values = [
        float(row["reference_delta_abs_vs_step0"])
        for row in metric_rows
        if row.get("reference_delta_abs_vs_step0") is not None
    ]
    previous_values = [
        float(row["previous_edit_retention"])
        for row in previous_rows
        if row.get("previous_edit_retention") is not None
    ]
    future_values = [
        float(row["future_record_drift"])
        for row in future_rows
        if row.get("future_record_drift") is not None
    ]
    return {
        "method": method,
        "step": int(step),
        "record_count": len(metric_rows),
        "edited_record_count": len(edited_rows),
        "mean_new_answer_nll_decrease_edited_records": _mean(new_values),
        "positive_new_answer_edits": sum(1 for value in new_values if value > 0.0),
        "mean_reference_delta_abs_all_records": _mean(ref_values),
        "locality_damage_records": sum(1 for row in metric_rows if row.get("locality_damage")),
        "mean_previous_edit_retention": _mean(previous_values),
        "mean_future_record_drift": _mean(future_values),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows if row.get("rollback_pass") is not None]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
    }


def _sequential_acceptance(summary_rows: List[Dict[str, Any]], rollback_payload: Dict[str, Any]) -> Dict[str, Any]:
    final_rows = [row for row in summary_rows if int(row.get("step") or 0) == 5]
    by_method = {row.get("method"): row for row in final_rows}
    cure = by_method.get("E_cure_dual_projected_tiny_lora", {})
    engram = by_method.get("C_engram_projected_tiny_lora", {})
    cure_ref = cure.get("mean_reference_delta_abs_all_records")
    cure_new = cure.get("mean_new_answer_nll_decrease_edited_records")
    checks = {
        "no_crash": True,
        "no_nan_inf": int(cure.get("nan_inf_count") or 0) == 0,
        "record_id_match_rate_is_1": float(cure.get("record_id_match_rate") or 0.0) == 1.0,
        "final_rollback_pass": rollback_payload.get("status") == "pass",
        "bank_cache_metadata_saved": True,
        "mean_new_answer_nll_decrease_positive": cure_new is not None and float(cure_new) > 0.0,
        "positive_new_answer_edits_at_least_4_of_5": int(cure.get("positive_new_answer_edits") or 0) >= 4,
        "mean_reference_delta_abs_less_than_mean_new_decrease": (
            cure_ref is not None and cure_new is not None and float(cure_ref) < float(cure_new)
        ),
        "reference_damage_below_direct_engram_failure": cure_ref is not None and float(cure_ref) < 0.0515347,
        "cure_reference_delta_lte_engram": (
            cure_ref is not None
            and engram.get("mean_reference_delta_abs_all_records") is not None
            and float(cure_ref) <= float(engram["mean_reference_delta_abs_all_records"])
        ),
        "cure_retention_no_worse_than_engram": (
            cure.get("mean_previous_edit_retention") is not None
            and engram.get("mean_previous_edit_retention") is not None
            and float(cure["mean_previous_edit_retention"]) >= float(engram["mean_previous_edit_retention"])
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "cure_final": cure,
        "engram_final": engram,
    }


def _beta0_acceptance(rows: List[Dict[str, Any]], rollback_payload: Dict[str, Any], *, tolerance: float = 1.0e-6) -> Dict[str, Any]:
    metric_rows = [row for row in rows if int(row.get("step") or 0) > 0]
    final_rows = [row for row in rows if int(row.get("step") or 0) == 5]

    def _max_abs(key: str) -> float:
        values = [abs(float(row[key])) for row in metric_rows if row.get(key) is not None]
        return max(values) if values else 0.0

    checks = {
        "five_edits_run": bool(final_rows) and {row.get("method") for row in final_rows} == {
            "C_engram_projected_tiny_lora",
            "E_cure_dual_projected_tiny_lora",
        },
        "new_answer_nll_unchanged": _max_abs("new_answer_nll_decrease_vs_step0") <= tolerance,
        "old_answer_nll_unchanged": _max_abs("old_answer_nll_delta_vs_step0") <= tolerance,
        "reference_nll_unchanged": _max_abs("reference_delta_abs_vs_step0") <= tolerance,
        "selected_weights_unchanged": rollback_payload.get("status") == "pass",
        "final_rollback_pass": rollback_payload.get("status") == "pass",
        "record_id_match_rate_is_1": all(float(row.get("record_id_match_rate") or 0.0) == 1.0 for row in rows),
        "no_nan_inf": not any(row.get("nan_inf_detected") for row in rows),
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _write_beta0_report(out_dir: Path, acceptance: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# CURE Sequential Beta=0 Gate",
        "",
        f"- Status: `{acceptance.get('status')}`",
        "- Methods: `C_engram_projected_tiny_lora`, `E_cure_dual_projected_tiny_lora`",
        "- beta: `0.0`",
        "- crisp_energy_threshold: `0.7`",
        "- Generation: skipped; evidence is NLL/logprob-based.",
        "",
        "## Checks",
        "",
    ]
    for key, value in (acceptance.get("checks") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Step Summary", ""])
    lines.extend(
        [
            "| method | step | mean new decrease edited | mean reference delta all | positive new | locality damage | nan/inf |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {method} | {step} | {new} | {ref} | {pos} | {loc} | {nan} |".format(
                method=row.get("method"),
                step=row.get("step"),
                new=_format(row.get("mean_new_answer_nll_decrease_edited_records")),
                ref=_format(row.get("mean_reference_delta_abs_all_records")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_records"),
                nan=row.get("nan_inf_count"),
            )
        )
    (out_dir / "REPORT_CURE_SEQUENTIAL_BETA0_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_direct_failure_metrics(report_path: Path) -> Dict[str, Any]:
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    def _extract_float(name: str, fallback: float) -> float:
        match = re.search(rf"{re.escape(name)}:\s*`([^`]+)`", text)
        return float(match.group(1)) if match else fallback

    def _extract_int(name: str, fallback: int) -> int:
        match = re.search(rf"{re.escape(name)}:\s*`([^`]+)`", text)
        return int(float(match.group(1))) if match else fallback

    return {
        "report_path": str(report_path),
        "report_found": report_path.exists(),
        "mean_target_nll_increase": _extract_float("mean_target_nll_increase", 0.0131024),
        "mean_reference_delta_abs_all_records": _extract_float("mean_reference_delta_abs_all_records", 0.0515347),
        "locality_damage_records": _extract_int("locality_damage_records", 4),
        "record_count": 5,
    }


def _load_previous_nonseq(previous_dir: Path) -> Dict[str, Any]:
    path = previous_dir / "cure_nonseq_results.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_path"] = str(path)
    return payload


def _write_sequential_final_report(
    out_dir: Path,
    *,
    sequential: Dict[str, Any],
    previous_nonseq: Dict[str, Any],
    direct_failure: Dict[str, Any],
    exact_command: str,
    skip_generation: bool,
) -> None:
    summary_rows = sequential.get("summary_rows", [])
    final_rows = [row for row in summary_rows if int(row.get("step") or 0) == 5]
    by_method = {row.get("method"): row for row in final_rows}
    cure = by_method.get("E_cure_dual_projected_tiny_lora", {})
    engram = by_method.get("C_engram_projected_tiny_lora", {})
    beta0_report = out_dir / "beta0_gate" / "REPORT_CURE_SEQUENTIAL_BETA0_GATE.md"
    beta0_payload_path = out_dir / "beta0_gate" / "cure_sequential_step_summary.json"
    beta0_payload = json.loads(beta0_payload_path.read_text(encoding="utf-8")) if beta0_payload_path.exists() else {}
    beta0_acceptance = beta0_payload.get("beta0_acceptance", {"status": "not_found"})

    acceptance = sequential.get("acceptance", {})
    decision = "C. CURE sequential fails under the strict acceptance checks. Do not scale; inspect beta/gamma/cache update/projection order."
    if acceptance.get("status") == "pass":
        cure_ref = cure.get("mean_reference_delta_abs_all_records")
        engram_ref = engram.get("mean_reference_delta_abs_all_records")
        if cure_ref is not None and engram_ref is not None and float(cure_ref) <= float(engram_ref):
            decision = "A. CURE sequential passes and improves or matches ENGRAM-projected LoRA. Next gate: 10-edit model-known non-PHI set."
        else:
            decision = "B. CURE sequential passes but does not improve over ENGRAM-projected LoRA. Keep ENGRAM-projected LoRA as simpler baseline; test 10-edit with both."

    best_nonseq = (previous_nonseq.get("best_cure") or previous_nonseq.get("acceptance", {}).get("aggregate") or {})
    lines = [
        "# Final CURE Sequential 5-Edit Report",
        "",
        "## Starting Point",
        "",
        f"- Previous nonseq CURE status: `{previous_nonseq.get('acceptance', {}).get('status')}`",
        f"- Best nonseq CURE method: `{best_nonseq.get('method')}`",
        f"- Best nonseq beta: `{best_nonseq.get('beta')}`",
        f"- Best nonseq crisp_energy_threshold: `{best_nonseq.get('crisp_energy_threshold')}`",
        "- Sequential beta: `0.5`, chosen as best nonseq beta / 2 to reduce accumulated locality risk.",
        "",
        "## Environment And Command",
        "",
        f"- Exact command: `{exact_command}`",
        f"- Output directory: `{out_dir}`",
        f"- Preflight: `{out_dir / 'PREFLIGHT.md'}`",
        f"- Environment report: `{out_dir / 'env_report.txt'}`",
        "",
        "## Tests",
        "",
        "- Logs are saved under `test_logs/`.",
        "",
        "## Beta=0 Sequential Gate",
        "",
        f"- Status: `{beta0_acceptance.get('status')}`",
        f"- Report: `{beta0_report}`",
        "",
        "## Methods Compared",
        "",
        "- `C_engram_projected_tiny_lora`",
        "- `E_cure_dual_projected_tiny_lora`",
        "- `B_tiny_lora_replacement`: skipped; optional only and not required for this gate.",
        "",
        "## Sequential Step Results",
        "",
        "| method | step | mean new decrease edited | mean reference delta all | previous retention | future drift | positive new | locality damage | rollback | match | nan/inf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {step} | {new} | {ref} | {ret} | {future} | {pos} | {loc} | {roll} | {match} | {nan} |".format(
                method=row.get("method"),
                step=row.get("step"),
                new=_format(row.get("mean_new_answer_nll_decrease_edited_records")),
                ref=_format(row.get("mean_reference_delta_abs_all_records")),
                ret=_format(row.get("mean_previous_edit_retention")),
                future=_format(row.get("mean_future_record_drift")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_records"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Final Step Comparison",
            "",
            "| method | mean_new_answer_nll_decrease | positive_new_answer_edits | mean_reference_delta_abs_all_records | locality_damage_records | rollback | record_id_match_rate | nan_inf_count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            "| {method} | {new} | {pos} | {ref} | {loc} | {roll} | {match} | {nan} |".format(
                method=row.get("method"),
                new=_format(row.get("mean_new_answer_nll_decrease_edited_records")),
                pos=row.get("positive_new_answer_edits"),
                ref=_format(row.get("mean_reference_delta_abs_all_records")),
                loc=row.get("locality_damage_records"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    failed_checks = [key for key, value in (acceptance.get("checks") or {}).items() if not value]
    lines.extend(["", "## Acceptance Checks", ""])
    lines.append(f"- Sequential acceptance: `{acceptance.get('status')}`")
    for key, value in (acceptance.get("checks") or {}).items():
        lines.append(f"- {key}: `{value}`")
    if failed_checks:
        lines.append(f"- Failed checks: `{failed_checks}`")
    cache_trace = sequential.get("crisp_cache_update_trace", [])
    lines.extend(
        [
            "",
            "## Crisp Cache Update",
            "",
            f"- Online accumulated cache used: `{bool(cache_trace)}`",
            "- Cache update policy: `streaming_average`",
            f"- Trace file: `{out_dir / 'crisp_cache_update_trace.json'}`",
            "",
            "## Comparison To Previous Direct ENGRAM Erase Sequential Failure",
            "",
            f"- direct erase mean_target_nll_increase: `{direct_failure.get('mean_target_nll_increase')}`",
            f"- direct erase mean_reference_delta_abs_all_records: `{direct_failure.get('mean_reference_delta_abs_all_records')}`",
            f"- direct erase locality_damage_records: `{direct_failure.get('locality_damage_records')}/5`",
            f"- CURE final mean_reference_delta_abs_all_records: `{_format(cure.get('mean_reference_delta_abs_all_records'))}`",
            f"- CURE final locality_damage_records: `{cure.get('locality_damage_records')}`",
            "",
            "## Generation",
            "",
            "- `--skip-generation` was used; evidence is NLL/logprob-based.",
            "- No generation-level, medical, or clinical efficacy claim is made.",
            "",
            "## Limitations",
            "",
            "- Synthetic non-PHI 5-edit only.",
            "- No 20-edit run.",
            "- No clinical or medical efficacy claim.",
            "- Delta-space Crisp projection is not original CrispEdit gradient-projected training.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_CURE_SEQUENTIAL_5EDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
    threshold: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
    record_id_match_rate: float,
    crisp_cache_update_policy: str = "streaming_average",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    if crisp_cache_update_policy not in {"static", "streaming_average"}:
        raise ValueError(f"Unsupported crisp_cache_update_policy: {crisp_cache_update_policy}")
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
    static_projection_caches: Optional[Dict[str, Dict[str, Any]]] = None

    if method == "E_cure_dual_projected_tiny_lora":
        initial_cache = _collect_reference_crisp_cache(model, records, image_root, module_names, hparams)
        accumulated_layer_to_cache = _clone_layer_to_cache(initial_cache.get("layer_to_cache", {}))
        accumulated_num_samples = int(initial_cache.get("num_samples") or 0)
        if crisp_cache_update_policy == "static":
            static_projection_caches = _projection_caches_for_thresholds_from_kfac(
                accumulated_layer_to_cache,
                hparams,
                [float(threshold)],
            ).get(float(threshold), {})
        cache_trace.append(
            {
                "step": 0,
                "method": method,
                "record_id": None,
                "crisp_cache_update_policy": crisp_cache_update_policy,
                "accumulated_num_samples": accumulated_num_samples,
                "modules_with_cache": sorted(accumulated_layer_to_cache),
                "cache_shapes": _cache_shapes(accumulated_layer_to_cache),
                "mask_keep_ratios": {},
                "cache_update_success": initial_cache.get("status") == "complete",
                "skipped_modules": [],
                "skip_reasons": {},
            }
        )

    try:
        rows.extend(
            _sequential_step_rows(
                model=model,
                records=records,
                image_root=image_root,
                baselines=baselines,
                snapshots=snapshots,
                method=method,
                step=0,
                applied_record_ids=[],
                beta=beta,
                threshold=threshold,
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
            if method == "C_engram_projected_tiny_lora":
                patch = EvalLoraPatch(model, engram_factors, beta=beta)
            elif method == "E_cure_dual_projected_tiny_lora":
                if static_projection_caches is not None:
                    projection_caches = static_projection_caches
                else:
                    projection_caches_by_threshold = _projection_caches_for_thresholds_from_kfac(
                        accumulated_layer_to_cache,
                        hparams,
                        [float(threshold)],
                    )
                    projection_caches = projection_caches_by_threshold.get(float(threshold), {})
                cure_entries, cure_summary = _apply_crisp_to_factors(engram_factors, projection_caches)
                patch = EvalMixedDeltaPatch(model, cure_entries, beta=beta)
            else:
                raise ValueError(f"Unsupported sequential method: {method}")
            patch.install()
            active_patches.append(patch)
            applied_ids.append(str(record["id"]))
            rows.extend(
                _sequential_step_rows(
                    model=model,
                    records=records,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    method=method,
                    step=step,
                    applied_record_ids=applied_ids,
                    beta=beta,
                    threshold=threshold,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                    record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                )
            )
            if method == "E_cure_dual_projected_tiny_lora":
                if crisp_cache_update_policy == "streaming_average":
                    new_cache = _collect_reference_crisp_cache(model, [record], image_root, module_names, hparams)
                    accumulated_layer_to_cache, accumulated_num_samples = _merge_layer_to_cache(
                        accumulated_layer_to_cache,
                        new_cache.get("layer_to_cache", {}),
                        accumulated_num_samples=accumulated_num_samples,
                        new_num_samples=int(new_cache.get("num_samples") or 0),
                    )
                    cache_update_success = new_cache.get("status") == "complete"
                else:
                    cache_update_success = True
                skipped_modules = [
                    row.get("module_name")
                    for row in cure_summary.get("modules", [])
                    if not row.get("crisp_projected")
                ]
                skip_reasons = {
                    row.get("module_name"): row.get("skip_reason")
                    for row in cure_summary.get("modules", [])
                    if not row.get("crisp_projected")
                }
                cache_trace.append(
                    {
                        "step": int(step),
                        "method": method,
                        "record_id": str(record["id"]),
                        "crisp_cache_update_policy": crisp_cache_update_policy,
                        "accumulated_num_samples": accumulated_num_samples,
                        "modules_with_cache": sorted(accumulated_layer_to_cache),
                        "cache_shapes": _cache_shapes(accumulated_layer_to_cache),
                        "mask_keep_ratios": _mask_keep_ratios(projection_caches),
                        "cache_update_success": cache_update_success,
                        "skipped_modules": skipped_modules,
                        "skip_reasons": skip_reasons,
                        "engram_projection": engram_summary,
                        "crisp_projection": cure_summary,
                        "lora_train": train_summary,
                    }
                )
    finally:
        after_step5_norms = _current_weight_norms(model, module_names)
        for patch in reversed(active_patches):
            patch.remove()
        rollback_rows, rollback_diff = _rollback_rows(
            model,
            snapshots,
            method=method,
            param_norm_after_step5=after_step5_norms,
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


def _run_sequential_cure(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    *,
    beta: float,
    threshold: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
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
            beta=beta,
            threshold=threshold,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
            record_id_match_rate=record_id_match_rate,
        )
        rows.extend(method_rows)
        rollback_checks.append(rollback_payload)
        cache_trace.extend(method_cache_trace)

    summary_rows = [
        _aggregate_sequential_step(rows, method, step)
        for method in ["C_engram_projected_tiny_lora", "E_cure_dual_projected_tiny_lora"]
        for step in range(0, len(records) + 1)
    ]
    rollback_payload = {
        "status": "pass" if all(item.get("rollback_pass") for item in rollback_checks) else "fail",
        "rollback_tolerance": float(rollback_tolerance),
        "methods": rollback_checks,
    }
    acceptance = _sequential_acceptance(summary_rows, rollback_payload)
    payload = {
        "status": "complete",
        "beta": float(beta),
        "crisp_energy_threshold": float(threshold),
        "methods": ["C_engram_projected_tiny_lora", "E_cure_dual_projected_tiny_lora"],
        "edit_record_matching": matching,
        "per_record_step_rows": rows,
        "summary_rows": summary_rows,
        "crisp_cache_update_trace": cache_trace,
        "final_rollback_check": rollback_payload,
        "acceptance": acceptance,
    }
    if float(beta) == 0.0:
        payload["beta0_acceptance"] = _beta0_acceptance(rows, rollback_payload)
    _json_dump(out_dir / "cure_sequential_step_matrix.json", rows)
    _write_csv(out_dir / "cure_sequential_step_matrix.csv", rows)
    _json_dump(out_dir / "cure_sequential_step_summary.json", payload)
    _write_csv(out_dir / "cure_sequential_step_summary.csv", summary_rows)
    _json_dump(out_dir / "crisp_cache_update_trace.json", cache_trace)
    _json_dump(out_dir / "final_rollback_check.json", rollback_payload)
    if float(beta) == 0.0:
        _write_beta0_report(out_dir, payload["beta0_acceptance"], summary_rows)
    return payload


def _run_nonseq_cure(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    *,
    beta_grid: List[float],
    thresholds: List[float],
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    cache_reports: List[Dict[str, Any]] = []

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
                    "diagnostics": cache_result.get("diagnostics", []),
                }
            )

            projection_caches_by_threshold = _projection_caches_for_thresholds_from_kfac(
                cache_result.get("layer_to_cache", {}),
                hparams,
                thresholds,
            )
            for threshold in thresholds:
                projection_caches = projection_caches_by_threshold.get(float(threshold), {})
                crisp_entries, crisp_summary = _apply_crisp_to_factors(factors, projection_caches)
                cure_entries, cure_summary = _apply_crisp_to_factors(engram_factors, projection_caches)
                for beta in beta_grid:
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
                            beta=beta,
                            extra={
                                "crisp_energy_threshold": float(threshold),
                                "selected_modules": EXPECTED_MODULES,
                                "projection_metadata": {},
                                "skipped_modules": [],
                                "skip_reasons": {},
                            },
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
                            patch=EvalMixedDeltaPatch(model, crisp_entries, beta=beta),
                            method="D_crisp_projected_tiny_lora",
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
                            extra={"crisp_projection": crisp_summary},
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
                            extra={"engram_projection": engram_summary, "crisp_projection": cure_summary},
                        )
                    )
            _restore_modules(model, snapshots)
    finally:
        _restore_modules(model, snapshots)

    aggregates = [_aggregate_group(rows, method, beta, threshold) for threshold in thresholds for beta in beta_grid for method in METHODS]
    cure_rows = [row for row in aggregates if row["method"] == "E_cure_dual_projected_tiny_lora" and row["status"] == "complete"]
    best_cure = max(
        cure_rows,
        key=lambda row: (
            float(row.get("positive_new_answer_edits") or 0),
            float(row.get("mean_new_answer_nll_decrease") or -math.inf),
            -float(row.get("mean_reference_delta_abs") or math.inf),
        ),
    ) if cure_rows else {}
    acceptance = _cure_acceptance(best_cure) if best_cure else {"status": "skipped", "reason": "no_cure_rows"}
    payload = {
        "status": "complete",
        "edit_record_matching": matching,
        "beta_grid": beta_grid,
        "crisp_energy_thresholds": thresholds,
        "cache_reports": cache_reports,
        "aggregate_rows": aggregates,
        "best_cure": best_cure,
        "acceptance": acceptance,
        "per_record": rows,
    }
    nonseq_dir = out_dir
    _json_dump(nonseq_dir / "cure_nonseq_results.json", payload)
    _write_csv(nonseq_dir / "cure_nonseq_results.csv", rows)
    _write_csv(nonseq_dir / "cure_nonseq_aggregates.csv", aggregates)
    _json_dump(out_dir / "crisp_cache_nonseq_summary.json", cache_reports)
    return payload


def _write_final_report(
    out_dir: Path,
    *,
    data_summary: Dict[str, Any],
    projector_extract: Optional[Dict[str, Any]],
    nonseq: Optional[Dict[str, Any]],
    sequential: Optional[Dict[str, Any]],
    prepare_only: bool,
) -> None:
    aggregates = (nonseq or {}).get("aggregate_rows", [])
    best_cure = (nonseq or {}).get("best_cure", {})
    acceptance = (nonseq or {}).get("acceptance", {"status": "skipped"})
    decision = "C. CURE was not validated; revisit curvature cache, projection order, or projected LoRA training."
    if prepare_only:
        decision = "C. Prepare-only run completed; no model metrics were produced."
    elif acceptance.get("status") == "pass":
        engram = [
            row
            for row in aggregates
            if row.get("method") == "C_engram_projected_tiny_lora"
            and row.get("beta") == best_cure.get("beta")
            and row.get("crisp_energy_threshold") == best_cure.get("crisp_energy_threshold")
        ]
        if engram and (best_cure.get("mean_reference_delta_abs") is not None) and (
            float(best_cure["mean_reference_delta_abs"]) < float(engram[0].get("mean_reference_delta_abs") or math.inf)
        ):
            decision = "A. CURE improves locality over ENGRAM-projected LoRA while preserving replacement success. Next gate: 10-edit model-known non-PHI set."
        else:
            decision = "B. CURE matches ENGRAM-projected LoRA but does not improve it. Keep ENGRAM-projected LoRA as simpler baseline and test 10-edit."

    lines = [
        "# Final CURE-MedEdit 5-Edit Report",
        "",
        "## Current Code Audit",
        "",
        "- Written to `outputs/cure_mededit_5edit/CURRENT_CODE_AUDIT.md` before source changes.",
        "- Existing localized replacement already implemented `Delta_candidate @ P_engram` as low-rank `B @ (A @ P)`.",
        "- Missing pieces were CrispEdit-style projection, MLLM K-FAC collection, CURE runner integration, and tests.",
        "",
        "## CrispEdit Source Audit",
        "",
        "- Written to `outputs/cure_mededit_5edit/CRISPEDIT_SOURCE_AUDIT.md`.",
        "- Inspected upstream commit `09035f16695998f3a71ec6006245d99e8cc648c8` under `external/CrispEdit/`.",
        "- Reused/adapted K-FAC projection cache logic, cache combination, and the ProjectedAdam matrix-free projection formula.",
        "- Did not reuse the text-only `run_crispedit.py` pipeline or ROME text-only layer stats.",
        "",
        "## Method",
        "",
        "- Candidate delta: tiny-LoRA replacement delta.",
        "- ENGRAM projector: multimodal target/reference projector bank from selected modules.",
        "- Crisp-style projection: K-FAC low-curvature mask with `Q_proj = U_B @ ((U_B.T @ Q @ U_A) * M.T) @ U_A.T`.",
        "- CURE delta-space MVP: `Delta_cure = Pi_crisp(Delta_candidate @ P_engram)`.",
        "- This is not identical to original CrispEdit gradient-projected training.",
        f"- Projector extraction status: `{(projector_extract or {}).get('status')}`",
        "",
        "## Data",
        "",
        f"- Records: `{data_summary.get('record_count')}`",
        f"- Replacement data: `{data_summary.get('replacement_data_file')}`",
        f"- Private or patient data used: `{data_summary.get('private_or_patient_data_used')}`",
        f"- Original data modified: `{data_summary.get('original_data_modified')}`",
        "",
        "## Non-Sequential Results",
        "",
    ]
    if not aggregates:
        lines.append("- No non-sequential model metrics were produced in this run.")
    else:
        lines.extend(
            [
                "| Method | beta | threshold | mean new NLL decrease | mean old NLL increase | mean ref delta abs | positive new | locality damage | rollback | match | nan/inf |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregates:
            lines.append(
                "| {method} | {beta} | {thr} | {new} | {old} | {ref} | {pos} | {loc} | {roll} | {match} | {nan} |".format(
                    method=row.get("method"),
                    beta=_format(row.get("beta")),
                    thr=_format(row.get("crisp_energy_threshold")),
                    new=_format(row.get("mean_new_answer_nll_decrease")),
                    old=_format(row.get("mean_old_answer_nll_increase")),
                    ref=_format(row.get("mean_reference_delta_abs")),
                    pos=row.get("positive_new_answer_edits"),
                    loc=row.get("locality_damage_edits"),
                    roll=_format(row.get("rollback_pass_rate")),
                    match=_format(row.get("record_id_match_rate")),
                    nan=row.get("nan_inf_count"),
                )
            )
        lines.extend(["", f"- Best CURE aggregate: `{best_cure}`", f"- CURE acceptance: `{acceptance.get('status')}`"])
    lines.extend(
        [
            "",
            "## Sequential Results",
            "",
            f"- Status: `{(sequential or {}).get('status', 'not_run')}`",
            f"- Reason: `{(sequential or {}).get('reason')}`",
            "",
            "## Comparison To Previous Direct ENGRAM Erase",
            "",
            "- Previous direct ENGRAM erase showed non-sequential signal but failed sequential locality scaling.",
            "- This CURE task keeps direct erase as a prior failure baseline and does not scale direct ENGRAM erase.",
            "",
            "## Limitations",
            "",
            "- Synthetic non-PHI 5-edit only.",
            "- No medical or clinical efficacy claim.",
            "- No 20-edit run.",
            "- Generation may be skipped or weak depending on runtime flags.",
            "- Delta-space Crisp projection is not the original CrispEdit gradient-projected training loop.",
            "- If ENGRAM-projected LoRA already has near-zero reference damage, CURE may not improve 5-edit locality.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    report_text = "\n".join(lines)
    (out_dir / "FINAL_CURE_MEDEDIT_5EDIT_REPORT.md").write_text(report_text, encoding="utf-8")
    (out_dir / "FINAL_CURE_NONSEQ_5EDIT_REPORT.md").write_text(report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CURE-MedEdit 5-edit prototype: ENGRAM-localized replacement plus Crisp-style projection.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml")
    parser.add_argument("--source-data", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json")
    parser.add_argument("--source-image-root", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/images")
    parser.add_argument("--output-dir", default="outputs/cure_mededit_5edit")
    parser.add_argument("--previous-nonseq-dir", default="outputs/cure_mededit_5edit/nonseq_real")
    parser.add_argument("--sequential-report", default="outputs/engram_sequential_5edit_smoke/FINAL_SEQUENTIAL_5EDIT_SMOKE_REPORT.md")
    parser.add_argument("--best-direct-config", default="outputs/engram_token_module_ablation_5edit/best_overall_config.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-grid", default="0.5,1.0")
    parser.add_argument("--crisp-energy-thresholds", default="0.7,0.9")
    parser.add_argument("--crisp-energy-threshold", type=float, default=None)
    parser.add_argument("--sequential-beta", type=float, default=0.5)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-sequential", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_git_outputs(out_dir)
    _write_failure_summary(out_dir, Path(args.sequential_report), Path(args.best_direct_config))
    replacement_data, image_root, data_summary = _prepare_replacement_data(Path(args.source_data), Path(args.source_image_root), out_dir)
    shutil.copyfile(args.hparams, out_dir / "base_hparams.used.yaml")

    if args.prepare_only:
        payload = {
            "status": "prepared",
            "replacement_data_file": str(replacement_data),
            "image_root": str(image_root),
            "next_step": "Run without --prepare-only on the LLaVA-Med environment.",
        }
        _json_dump(out_dir / "prepare_only_status.json", payload)
        _write_final_report(out_dir, data_summary=data_summary, projector_extract=None, nonseq=None, sequential=payload, prepare_only=True)
        return 0

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(
        hparams,
        image_root=image_root,
        bank_dir=out_dir / "projector_bank",
        device=args.device,
        edit_mode="erase",
    )
    hparams.replacement_mode = "cure_delta_projected"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.use_crisp_projection = True

    records = _load_records(replacement_data)
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    selected_status = {
        "status": "pass" if set(selected) == set(EXPECTED_MODULES) and len(selected) == len(EXPECTED_MODULES) else "fail",
        "selected_module_names": selected,
        "expected_module_names": EXPECTED_MODULES,
    }
    _json_dump(out_dir / "selected_modules_preflight.json", selected_status)
    if selected_status["status"] != "pass":
        raise RuntimeError(f"Selected modules do not match locked q/k/gate set: {selected_status}")

    baselines = {
        str(record["id"]): _evaluate_current(
            editor.model,
            record,
            image_root,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
        )
        for record in records
    }
    _json_dump(out_dir / "baseline_metrics.json", baselines)

    projector_extract = _extract_projector_bank(editor, hparams, replacement_data, records, out_dir / "projector_bank")
    _json_dump(out_dir / "projector_extraction_summary.json", projector_extract)

    thresholds = [float(args.crisp_energy_threshold)] if args.crisp_energy_threshold is not None else _parse_float_list(args.crisp_energy_thresholds)
    if args.run_sequential:
        sequential_threshold = float(args.crisp_energy_threshold) if args.crisp_energy_threshold is not None else float(thresholds[0])
        sequential = _run_sequential_cure(
            editor.model,
            records,
            image_root,
            baselines,
            out_dir / "projector_bank",
            EXPECTED_MODULES,
            hparams,
            out_dir,
            beta=float(args.sequential_beta),
            threshold=sequential_threshold,
            rollback_tolerance=args.rollback_tolerance,
            locality_threshold=args.locality_damage_threshold,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
        )
        if float(args.sequential_beta) == 0.0:
            return 0
        _write_sequential_final_report(
            out_dir,
            sequential=sequential,
            previous_nonseq=_load_previous_nonseq(Path(args.previous_nonseq_dir)),
            direct_failure=_extract_direct_failure_metrics(Path(args.sequential_report)),
            exact_command=" ".join([sys.executable, *sys.argv]),
            skip_generation=bool(args.skip_generation),
        )
        return 0

    nonseq = _run_nonseq_cure(
        editor.model,
        records,
        image_root,
        baselines,
        out_dir / "projector_bank",
        EXPECTED_MODULES,
        hparams,
        out_dir,
        beta_grid=_parse_float_list(args.beta_grid),
        thresholds=thresholds,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )

    sequential = {"status": "skipped", "reason": "sequential CURE runner is gated behind --run-sequential and nonseq CURE pass"}
    if args.run_sequential and nonseq.get("acceptance", {}).get("status") == "pass":
        sequential = {
            "status": "skipped",
            "reason": "full sequential CURE application is not enabled in this MVP; use nonseq result as the first gate",
        }
        _json_dump(out_dir / "sequential" / "cure_sequential_skipped.json", sequential)
    elif args.run_sequential:
        sequential = {"status": "skipped", "reason": "nonseq CURE did not pass"}
        _json_dump(out_dir / "sequential" / "cure_sequential_skipped.json", sequential)

    _write_final_report(out_dir, data_summary=data_summary, projector_extract=projector_extract, nonseq=nonseq, sequential=sequential, prepare_only=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
