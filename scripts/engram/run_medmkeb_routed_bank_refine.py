#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.engram.run_medmkeb_modelknown_editing import (  # noqa: E402
    DEFAULT_HPARAMS,
    DEFAULT_OUTPUT_DIR,
    _finite,
    _format,
    _heavy_imports,
    _json_dump,
    _mean,
    _run_pytest,
    _safe_div,
    _write_csv,
    _write_env_report,
    _write_git_outputs,
    _write_preflight,
)
from scripts.engram.run_medmkeb_sequential_pareto_refine import (  # noqa: E402
    _configure_hparams_for_scope,
    _load_records,
    _read_json,
)
from scripts.engram.run_medmkeb_routed_edit_bank import (  # noqa: E402
    RoutedLoraPatch,
    _bank_configs,
    _build_edit_bank,
    _build_query_cache,
    _cosine,
    _delta,
    _ensure_layout,
    _existing_projector_bank_matches,
    _extract_prototype,
    _module_scope_names,
    _raw_nll,
    _sample_for_kind,
    _tensor_summary,
    _write_package_hygiene_report,
)


REFINE_DIRNAME = "routed_bank_refine_20"
PREVIOUS_ROUTED_DIR = DEFAULT_OUTPUT_DIR / "routed_bank_20"
ORACLE_FINAL_NEW = 0.775974
ORACLE_FINAL_REF = 0.0
ORACLE_SELF_HIT = 1.0


def _record_id(record: Dict[str, Any]) -> str:
    return str(record.get("id") or record.get("record_id"))


def _write_tests(out_dir: Path, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    if not run_tests:
        payload = {"status": "skipped", "reason": "--skip-tests"}
        _json_dump(test_dir / "test_status.json", payload)
        return payload
    engram_tests = sorted(str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.glob("tests/test_engram_*.py"))
    runs = [
        _run_pytest(test_dir / "test_routed_bank.log", ["tests/test_engram_routed_bank.py", "-q"]),
        _run_pytest(test_dir / "test_engram_all.log", [*engram_tests, "-q"]),
        _run_pytest(test_dir / "test_cure_crisp_projection.log", ["tests/test_cure_crisp_projection.py", "-q"]),
        _run_pytest(test_dir / "test_cure_kfac_collector_tiny_mllm.log", ["tests/test_cure_kfac_collector_tiny_mllm.py", "-q"]),
    ]
    payload = {
        "status": "pass" if runs[0]["returncode"] == 0 and runs[1]["returncode"] == 0 else "fail",
        "routed_bank_tests_pass": runs[0]["returncode"] == 0,
        "engram_tests_pass": runs[1]["returncode"] == 0,
        "cure_projection_tests_pass": runs[2]["returncode"] == 0,
        "cure_kfac_tests_pass": runs[3]["returncode"] == 0,
        "cure_tests_blocking": False,
        "runs": runs,
    }
    _json_dump(test_dir / "test_status.json", payload)
    return payload


def _refine_configs(compact: bool = True) -> List[Dict[str, Any]]:
    configs = [
        {
            "config_id": "R_oracle_self_high",
            "bank_config_id": "bank_high_strength",
            "route_policy": "oracle_self",
            "prototype_type": "target",
            "threshold": None,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top1_target_adaptive_high",
            "bank_config_id": "bank_high_strength",
            "route_policy": "top1_threshold",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top3_target_adaptive_high",
            "bank_config_id": "bank_high_strength",
            "route_policy": "top3_threshold",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 3,
        },
        {
            "config_id": "R_target_margin_top1_p90_m005",
            "bank_config_id": "bank_high_strength",
            "route_policy": "target_margin_top1",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "margin": 0.05,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_target_ref_reject_lam05_p90",
            "bank_config_id": "bank_high_strength",
            "route_policy": "target_ref_reject",
            "prototype_type": "target_plus_reference",
            "threshold": "adaptive_p90_reference",
            "target_threshold": "adaptive_p90_reference",
            "lambda_ref": 0.5,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_contrastive_margin_p90_m005",
            "bank_config_id": "bank_high_strength",
            "route_policy": "contrastive_margin_top1",
            "prototype_type": "contrastive",
            "threshold": "adaptive_p90_reference",
            "margin": 0.05,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_zscore_refreject_z15",
            "bank_config_id": "bank_high_strength",
            "route_policy": "zscore_reference_reject",
            "prototype_type": "target",
            "z_threshold": 1.5,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_two_stage_p90_m005_refreject",
            "bank_config_id": "bank_high_strength",
            "route_policy": "two_stage_router",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "margin": 0.05,
            "lambda_ref": 0.5,
            "z_threshold": 1.5,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_oracle_reference_reject_top1",
            "bank_config_id": "bank_high_strength",
            "route_policy": "oracle_reference_reject_upper_bound",
            "base_policy": "top1_threshold",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 1,
        },
    ]
    if not compact:
        configs[4:4] = [
            {
                "config_id": "R_target_margin_top1_p95_m005",
                "bank_config_id": "bank_high_strength",
                "route_policy": "target_margin_top1",
                "prototype_type": "target",
                "threshold": "adaptive_p95_reference",
                "margin": 0.05,
                "max_active_edits": 1,
            },
            {
                "config_id": "R_target_margin_top1_p90_m010",
                "bank_config_id": "bank_high_strength",
                "route_policy": "target_margin_top1",
                "prototype_type": "target",
                "threshold": "adaptive_p90_reference",
                "margin": 0.10,
                "max_active_edits": 1,
            },
            {
                "config_id": "R_target_ref_reject_lam10_p90",
                "bank_config_id": "bank_high_strength",
                "route_policy": "target_ref_reject",
                "prototype_type": "target_plus_reference",
                "threshold": "adaptive_p90_reference",
                "target_threshold": "adaptive_p90_reference",
                "lambda_ref": 1.0,
                "max_active_edits": 1,
            },
        ]
    return configs


def _threshold_value(name_or_value: Any, stats: Dict[str, Any], prototype_type: str) -> Optional[float]:
    if name_or_value is None:
        return None
    if isinstance(name_or_value, (int, float)):
        return float(name_or_value)
    key = str(name_or_value)
    if key.startswith("adaptive_"):
        return float(stats["adaptive_thresholds"][prototype_type][key])
    return float(key)


def _prototype_key(prototype_type: str) -> str:
    if prototype_type == "target_plus_reference":
        return "target"
    return prototype_type


def _entry_similarity(query: Any, entry: Dict[str, Any], prototype_type: str) -> float:
    return _cosine(query, entry[_prototype_key(prototype_type)])


def _calibration_stats(
    *,
    records: List[Dict[str, Any]],
    entries: Sequence[Dict[str, Any]],
    query_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    import torch

    rows: List[Dict[str, Any]] = []
    adaptive: Dict[str, Dict[str, float]] = {}
    ref_by_edit: Dict[str, List[float]] = {str(entry["edit_id"]): [] for entry in entries}
    for prototype_type in ["target", "contrastive"]:
        target_self: List[float] = []
        target_nonself: List[float] = []
        ref_max: List[float] = []
        ref_top1: List[float] = []
        future_max: List[float] = []
        for step in range(0, len(records) + 1):
            active = list(entries[:step])
            if not active:
                continue
            for idx, record in enumerate(records):
                rid = _record_id(record)
                target_q = query_cache[(rid, "new")]["query"]
                ref_q = query_cache[(rid, "reference")]["query"]
                if target_q is not None:
                    sims = [(entry, _entry_similarity(target_q, entry, prototype_type)) for entry in active]
                    top = max([sim for _entry, sim in sims], default=None)
                    for entry, sim in sims:
                        if str(entry["record_id"]) == rid:
                            target_self.append(sim)
                        else:
                            target_nonself.append(sim)
                    if idx >= step and top is not None:
                        future_max.append(top)
                if ref_q is not None:
                    sims = [(entry, _entry_similarity(ref_q, entry, prototype_type)) for entry in active]
                    if sims:
                        sorted_sims = sorted([sim for _entry, sim in sims], reverse=True)
                        ref_top1.append(sorted_sims[0])
                        ref_max.append(sorted_sims[0])
                        for entry, sim in sims:
                            if prototype_type == "target":
                                ref_by_edit[str(entry["edit_id"])].append(sim)
        def percentile(values: List[float], q: float) -> float:
            if not values:
                return 1.0
            return float(torch.quantile(torch.tensor(values, dtype=torch.float32), q).item())
        adaptive[prototype_type] = {
            "adaptive_p90_reference": percentile(ref_max, 0.90),
            "adaptive_p95_reference": percentile(ref_max, 0.95),
        }
        rows.append(
            {
                "prototype_type": prototype_type,
                "target_self_mean": _mean(target_self),
                "target_nonself_mean": _mean(target_nonself),
                "reference_max_mean": _mean(ref_max),
                "reference_top1_mean": _mean(ref_top1),
                "future_max_mean": _mean(future_max),
                "reference_max_p90": adaptive[prototype_type]["adaptive_p90_reference"],
                "reference_max_p95": adaptive[prototype_type]["adaptive_p95_reference"],
                "target_self_count": len(target_self),
                "reference_count": len(ref_max),
            }
        )
    z_stats = {}
    for entry in entries:
        values = ref_by_edit[str(entry["edit_id"])]
        mu = float(sum(values) / len(values)) if values else 0.0
        var = float(sum((value - mu) ** 2 for value in values) / len(values)) if values else 0.0
        z_stats[str(entry["edit_id"])] = {"mu_ref": mu, "sigma_ref": math.sqrt(var)}
    return {"rows": rows, "adaptive_thresholds": adaptive, "z_reference_stats": z_stats}


def _route_refined(
    *,
    query: Any,
    entries: Sequence[Dict[str, Any]],
    query_record_id: str,
    query_kind: str,
    config: Dict[str, Any],
    threshold: Optional[float],
    target_threshold: Optional[float],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    if not entries or query is None:
        return _empty_route(query_kind, threshold)
    policy = str(config["route_policy"])
    prototype_type = str(config.get("prototype_type") or "target")
    max_active = int(config.get("max_active_edits") or 1)
    margin_threshold = float(config.get("margin") or 0.0)
    lambda_ref = float(config.get("lambda_ref") or 0.0)
    z_threshold = config.get("z_threshold")
    scored: List[Dict[str, Any]] = []
    for entry in entries:
        target_sim = _cosine(query, entry["target"])
        ref_proto = entry.get("reference")
        reference_sim = _cosine(query, ref_proto) if ref_proto is not None else 0.0
        contrastive_sim = _cosine(query, entry["contrastive"])
        z_info = calibration.get("z_reference_stats", {}).get(str(entry["edit_id"]), {})
        z_score = (target_sim - float(z_info.get("mu_ref") or 0.0)) / (float(z_info.get("sigma_ref") or 0.0) + 1.0e-6)
        if prototype_type == "contrastive":
            score = contrastive_sim
            proto_sim = contrastive_sim
        elif policy == "target_ref_reject":
            score = target_sim - lambda_ref * max(0.0, reference_sim)
            proto_sim = target_sim
        elif policy == "zscore_reference_reject":
            score = z_score
            proto_sim = target_sim
        else:
            score = target_sim
            proto_sim = target_sim
        scored.append(
            {
                "entry": entry,
                "score": float(score),
                "proto_sim": float(proto_sim),
                "target_sim": float(target_sim),
                "reference_sim": float(reference_sim),
                "contrastive_sim": float(contrastive_sim),
                "z_score": float(z_score),
            }
        )
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    second = ranked[1]["score"] if len(ranked) > 1 else None
    top = ranked[0]
    margin_value = None if second is None else float(top["score"]) - float(second)
    active: List[Dict[str, Any]] = []
    reject_reason = "none"

    if policy == "oracle_self":
        active = [item for item in ranked if str(item["entry"]["record_id"]) == str(query_record_id) and query_kind in {"new", "old", "target"}][:1]
        reject_reason = "oracle_reference_or_no_self" if not active else "none"
    elif policy == "top1_threshold":
        if threshold is not None and top["score"] >= float(threshold):
            active = [top]
        else:
            reject_reason = "below_threshold"
    elif policy == "top3_threshold":
        if threshold is not None:
            active = [item for item in ranked[:max_active] if item["score"] >= float(threshold)]
            reject_reason = "below_threshold" if not active else "none"
    elif policy == "target_margin_top1":
        if threshold is None or top["target_sim"] < float(threshold):
            reject_reason = "below_threshold"
        elif margin_value is None or margin_value < margin_threshold:
            reject_reason = "below_margin"
        else:
            active = [top]
    elif policy == "target_ref_reject":
        if threshold is None or top["score"] < float(threshold):
            reject_reason = "below_score_threshold"
        elif target_threshold is None or top["target_sim"] < float(target_threshold):
            reject_reason = "below_target_threshold"
        else:
            active = [top]
    elif policy == "contrastive_margin_top1":
        if threshold is None or top["score"] < float(threshold):
            reject_reason = "below_threshold"
        elif margin_value is None or margin_value < margin_threshold:
            reject_reason = "below_margin"
        else:
            active = [top]
    elif policy == "zscore_reference_reject":
        if z_threshold is not None and top["z_score"] >= float(z_threshold):
            active = [top]
        else:
            reject_reason = "below_z_threshold"
    elif policy == "two_stage_router":
        target_ranked = sorted(scored, key=lambda item: item["target_sim"], reverse=True)
        top = target_ranked[0]
        second_target = target_ranked[1]["target_sim"] if len(target_ranked) > 1 else None
        margin_value = None if second_target is None else float(top["target_sim"]) - float(second_target)
        ref_reject_threshold = threshold
        if threshold is None or top["target_sim"] < float(threshold):
            reject_reason = "below_threshold"
        elif margin_value is None or margin_value < margin_threshold:
            reject_reason = "below_margin"
        elif ref_reject_threshold is not None and top["reference_sim"] > float(ref_reject_threshold):
            reject_reason = "reference_too_high"
        elif z_threshold is not None and top["z_score"] < float(z_threshold):
            reject_reason = "below_z_threshold"
        else:
            active = [top]
    elif policy == "oracle_reference_reject_upper_bound":
        if query_kind == "reference":
            reject_reason = "oracle_reference_reject"
        elif threshold is not None and top["target_sim"] >= float(threshold):
            active = [top]
        else:
            reject_reason = "below_threshold"
    else:
        raise ValueError(f"unsupported route_policy: {policy}")

    active_entries = [item["entry"] for item in active]
    top_entry = top["entry"]
    active_ids = [str(entry["edit_id"]) for entry in active_entries]
    active_record_ids = [str(entry["record_id"]) for entry in active_entries]
    return {
        "query_kind": query_kind,
        "active_edit_ids": active_ids,
        "active_record_ids": active_record_ids,
        "active_edit_similarities": [float(item["score"]) for item in active],
        "active_edit_count": len(active_entries),
        "self_edit_active": str(query_record_id) in active_record_ids,
        "top1_edit_id": str(top_entry["edit_id"]),
        "top1_record_id": str(top_entry["record_id"]),
        "top1_similarity": float(top["score"]),
        "second_similarity": second,
        "margin_value": margin_value,
        "top1_margin": margin_value,
        "max_similarity": float(top["score"]),
        "target_sim": float(top["target_sim"]),
        "reference_sim": float(top["reference_sim"]),
        "z_score": float(top["z_score"]),
        "threshold_value": threshold,
        "target_threshold_value": target_threshold,
        "reject_reason": reject_reason,
    }


def _empty_route(query_kind: str, threshold: Optional[float]) -> Dict[str, Any]:
    return {
        "query_kind": query_kind,
        "active_edit_ids": [],
        "active_record_ids": [],
        "active_edit_similarities": [],
        "active_edit_count": 0,
        "self_edit_active": False,
        "top1_edit_id": None,
        "top1_record_id": None,
        "top1_similarity": None,
        "second_similarity": None,
        "margin_value": None,
        "top1_margin": None,
        "max_similarity": None,
        "target_sim": None,
        "reference_sim": None,
        "z_score": None,
        "threshold_value": threshold,
        "target_threshold_value": None,
        "reject_reason": "empty_bank_or_query",
        "temporary_rollback_pass": True,
        "temporary_rollback_max_abs_diff": 0.0,
    }


def _evaluate_one(
    *,
    model: Any,
    record: Dict[str, Any],
    kind: str,
    entries: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    threshold: Optional[float],
    target_threshold: Optional[float],
    module_names: Sequence[str],
    calibration: Dict[str, Any],
    query_cache: Dict[Tuple[str, str], Dict[str, Any]],
    rollback_tolerance: float,
    eval_cache: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[Optional[Dict[str, Any]], bool, float]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    import torch
    from scripts.engram.run_token_module_ablation_5edit import _answer_metrics, _max_snapshot_diff, _snapshot_modules

    rid = _record_id(record)
    cached = query_cache[(rid, kind)]
    sample = cached["sample"]
    query = cached["query"]
    if sample is None or query is None:
        return None, _empty_route(kind, threshold)
    route = _route_refined(
        query=query,
        entries=entries,
        query_record_id=rid,
        query_kind=kind,
        config=config,
        threshold=threshold,
        target_threshold=target_threshold,
        calibration=calibration,
    )
    active_by_id = {entry["edit_id"]: entry for entry in entries}
    active_entries = [active_by_id[edit_id] for edit_id in route["active_edit_ids"]]
    cache_key = (str(config["config_id"]), f"{rid}::{kind}", tuple(route["active_edit_ids"]))
    if not active_entries:
        raw = cached["baseline_raw"]
        rollback_diff = 0.0
        rollback_pass = True
    elif cache_key in eval_cache:
        raw, rollback_pass, rollback_diff = eval_cache[cache_key]
    else:
        snapshots = _snapshot_modules(model, list(module_names))
        patch = RoutedLoraPatch(model, active_entries, [1.0 for _ in active_entries])
        patch.install()
        try:
            raw = _answer_metrics(model, dict(sample))
        finally:
            patch.remove()
        rollback_diff = _max_snapshot_diff(model, snapshots)
        rollback_pass = rollback_diff <= float(rollback_tolerance)
        eval_cache[cache_key] = (raw, rollback_pass, rollback_diff)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    route.update({"temporary_rollback_pass": rollback_pass, "temporary_rollback_max_abs_diff": rollback_diff})
    return raw, route


def _aggregate_step(rows: List[Dict[str, Any]], config: Dict[str, Any], step: int, total: int) -> Dict[str, Any]:
    step_rows = [row for row in rows if int(row.get("step") or -1) == int(step)]
    edited = [row for row in step_rows if row.get("is_edited_so_far")]
    previous = [row for row in step_rows if row.get("is_previous_edit")]
    future = [row for row in step_rows if row.get("is_future_edit")]
    return {
        "config_id": config["config_id"],
        "bank_config_id": config["bank_config_id"],
        "route_policy": config["route_policy"],
        "prototype_type": config.get("prototype_type"),
        "threshold": config.get("threshold"),
        "margin": config.get("margin"),
        "lambda_ref": config.get("lambda_ref"),
        "z_threshold": config.get("z_threshold"),
        "step": step,
        "record_count": len(step_rows),
        "edited_record_count": len(edited),
        "mean_new_answer_nll_decrease": _mean([row.get("new_answer_nll_decrease_vs_step0") for row in edited]),
        "mean_ref_abs": _mean([row.get("locality_reference_delta_abs_vs_step0") for row in step_rows]),
        "positive_new_answer_edits": sum(1 for row in edited if (row.get("new_answer_nll_decrease_vs_step0") or 0.0) > 0.0),
        "locality_damage_records": sum(1 for row in step_rows if row.get("locality_damage")),
        "previous_edit_retention": _mean([row.get("previous_edit_retention") for row in previous]),
        "mean_previous_edit_forgetting": _mean([row.get("previous_edit_forgetting") for row in previous]),
        "retention_ratio": _mean([row.get("retention_ratio") for row in previous]),
        "future_record_drift": _mean([row.get("future_record_drift") for row in future]),
        "temporary_rollback_pass_rate": _mean([1.0 if row.get("temporary_rollback_pass") else 0.0 for row in step_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in step_rows]),
        "nan_inf_count": sum(1 for row in step_rows if row.get("nan_inf_detected")),
        "final_step": step == total,
    }


def _routing_metrics(routing_rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]], config: Dict[str, Any], total: int) -> Dict[str, Any]:
    final_targets = [row for row in routing_rows if int(row.get("step") or -1) == total and row.get("query_kind") == "new" and row.get("is_edited_so_far")]
    final_refs = [row for row in routing_rows if int(row.get("step") or -1) == total and row.get("query_kind") == "reference"]
    future_targets = [row for row in routing_rows if row.get("query_kind") == "new" and row.get("is_future_edit")]
    return {
        "config_id": config["config_id"],
        "self_hit_rate": _mean([1.0 if row.get("self_edit_active") else 0.0 for row in final_targets]),
        "top1_self_rate": _mean([1.0 if str(row.get("top1_record_id")) == str(row.get("record_id")) else 0.0 for row in final_targets]),
        "mean_self_similarity": _mean([row.get("top1_similarity") for row in final_targets if str(row.get("top1_record_id")) == str(row.get("record_id"))]),
        "mean_top1_margin": _mean([row.get("margin_value") for row in final_targets]),
        "edited_target_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in final_targets]),
        "reference_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in final_refs]),
        "reference_false_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in final_refs]),
        "mean_reference_active_count": _mean([row.get("active_edit_count") for row in final_refs]),
        "mean_reference_max_similarity": _mean([row.get("max_similarity") for row in final_refs]),
        "reference_rejection_rate": _mean([1.0 if (row.get("active_edit_count") or 0) == 0 else 0.0 for row in final_refs]),
        "future_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in future_targets]),
        "future_false_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in future_targets]),
        "mean_future_max_similarity": _mean([row.get("max_similarity") for row in future_targets]),
        "target_self_active": sum(1 for row in final_targets if row.get("self_edit_active")),
        "target_wrong_edit_active": sum(1 for row in final_targets if (row.get("active_edit_count") or 0) > 0 and not row.get("self_edit_active")),
        "target_no_edit_active": sum(1 for row in final_targets if (row.get("active_edit_count") or 0) == 0),
        "reference_any_edit_active": sum(1 for row in final_refs if (row.get("active_edit_count") or 0) > 0),
        "reference_no_edit_active": sum(1 for row in final_refs if (row.get("active_edit_count") or 0) == 0),
        "matrix_row_count": len(matrix_rows),
    }


def _evaluate_config(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    baselines: Dict[str, Any],
    entries: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    run_dir: Path,
    module_names: Sequence[str],
    query_cache: Dict[Tuple[str, str], Dict[str, Any]],
    calibration: Dict[str, Any],
    rollback_tolerance: float,
    locality_threshold: float,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    threshold = _threshold_value(config.get("threshold"), calibration, _prototype_key(str(config.get("prototype_type") or "target")))
    target_threshold = _threshold_value(config.get("target_threshold"), calibration, "target")
    rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    record_ids = [_record_id(record) for record in records]
    record_id_match_rate = 1.0 if len(record_ids) == len(set(record_ids)) == len(records) else 0.0
    eval_cache: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[Optional[Dict[str, Any]], bool, float]] = {}
    for step in range(0, len(records) + 1):
        active_entries = list(entries[:step])
        applied_ids = [str(entry["record_id"]) for entry in active_entries]
        for idx, record in enumerate(records):
            rid = _record_id(record)
            old_raw, old_route = _evaluate_one(model=model, record=record, kind="old", entries=active_entries, config=config, threshold=threshold, target_threshold=target_threshold, module_names=module_names, calibration=calibration, query_cache=query_cache, rollback_tolerance=rollback_tolerance, eval_cache=eval_cache)
            new_raw, new_route = _evaluate_one(model=model, record=record, kind="new", entries=active_entries, config=config, threshold=threshold, target_threshold=target_threshold, module_names=module_names, calibration=calibration, query_cache=query_cache, rollback_tolerance=rollback_tolerance, eval_cache=eval_cache)
            ref_raw, ref_route = _evaluate_one(model=model, record=record, kind="reference", entries=active_entries, config=config, threshold=threshold, target_threshold=target_threshold, module_names=module_names, calibration=calibration, query_cache=query_cache, rollback_tolerance=rollback_tolerance, eval_cache=eval_cache)
            base = baselines[str(rid)]
            old_delta = _delta(_raw_nll(old_raw), _raw_nll(base.get("old_raw")))
            new_delta = _delta(_raw_nll(new_raw), _raw_nll(base.get("new_raw")))
            ref_delta = _delta(_raw_nll(ref_raw), _raw_nll(base.get("reference_raw")))
            new_decrease = None if new_delta is None else -new_delta
            ref_abs = None if ref_delta is None else abs(ref_delta)
            is_edited = idx < step
            is_previous = idx < max(step - 1, 0)
            is_future = idx >= step
            rollback_pass = bool(old_route["temporary_rollback_pass"] and new_route["temporary_rollback_pass"] and ref_route["temporary_rollback_pass"])
            rollback_max = max(float(old_route["temporary_rollback_max_abs_diff"] or 0.0), float(new_route["temporary_rollback_max_abs_diff"] or 0.0), float(ref_route["temporary_rollback_max_abs_diff"] or 0.0))
            row = {
                "config_id": config["config_id"],
                "bank_config_id": config["bank_config_id"],
                "route_policy": config["route_policy"],
                "prototype_type": config.get("prototype_type"),
                "threshold": config.get("threshold"),
                "threshold_value": threshold,
                "margin": config.get("margin"),
                "lambda_ref": config.get("lambda_ref"),
                "z_threshold": config.get("z_threshold"),
                "step": step,
                "applied_record_ids": applied_ids,
                "record_id": rid,
                "is_edited_so_far": is_edited,
                "is_current_edit": bool(step > 0 and idx == step - 1),
                "is_previous_edit": is_previous,
                "is_future_edit": is_future,
                **{key: new_route.get(key) for key in ["active_edit_ids", "active_edit_count", "active_edit_similarities", "self_edit_active", "top1_edit_id", "top1_similarity", "second_similarity", "margin_value", "max_similarity", "target_sim", "reference_sim", "z_score", "reject_reason"]},
                "reference_active_edit_ids": ref_route.get("active_edit_ids"),
                "reference_active_edit_count": ref_route.get("active_edit_count"),
                "reference_self_edit_active": ref_route.get("self_edit_active"),
                "old_answer_nll": _raw_nll(old_raw),
                "old_answer_nll_delta_vs_step0": old_delta,
                "new_answer_nll": _raw_nll(new_raw),
                "new_answer_nll_decrease_vs_step0": new_decrease,
                "locality_reference_nll": _raw_nll(ref_raw),
                "locality_reference_delta_abs_vs_step0": ref_abs,
                "previous_edit_retention": new_decrease if is_previous else None,
                "previous_edit_forgetting": None if not is_previous or new_decrease is None else max(0.0, -float(new_decrease)),
                "retention_ratio": _safe_div(new_decrease, ORACLE_FINAL_NEW) if is_previous else None,
                "future_record_drift": abs(float(new_decrease or 0.0)) if is_future else None,
                "locality_damage": bool(ref_abs is not None and ref_abs > float(locality_threshold)),
                "temporary_rollback_pass": rollback_pass,
                "temporary_rollback_max_abs_diff": rollback_max,
                "record_id_match_rate": record_id_match_rate,
                "nan_inf_detected": not _finite({"old": old_raw, "new": new_raw, "ref": ref_raw, "rollback": rollback_max}),
            }
            rows.append(row)
            for kind, route in [("old", old_route), ("new", new_route), ("reference", ref_route)]:
                routing_rows.append({"config_id": config["config_id"], "step": step, "record_id": rid, "query_kind": kind, "is_edited_so_far": is_edited, "is_future_edit": is_future, **route})
        _json_dump(run_dir / "progress.json", {"status": "running", "config_id": config["config_id"], "step_completed": step, "steps_total": len(records), "eval_cache_entries": len(eval_cache), "latest_step_summary": _aggregate_step(rows, config, step, len(records))})
        _write_csv(run_dir / "routed_step_matrix.partial.csv", rows)
    summary_rows = [_aggregate_step(rows, config, step, len(records)) for step in range(0, len(records) + 1)]
    metrics = _routing_metrics(routing_rows, rows, config, len(records))
    rollback = {"status": "pass" if _mean([1.0 if row.get("temporary_rollback_pass") else 0.0 for row in rows]) == 1.0 else "fail", "temporary_rollback_max_abs_diff": max(float(row.get("temporary_rollback_max_abs_diff") or 0.0) for row in rows) if rows else 0.0}
    payload = {"status": "complete", "config": config, "threshold_value": threshold, "target_threshold_value": target_threshold, "per_record_step_rows": rows, "summary_rows": summary_rows, "final_summary": summary_rows[-1], "routing_metrics": metrics, "rollback_check": rollback}
    _json_dump(run_dir / "config.json", config)
    _json_dump(run_dir / "routed_step_matrix.json", rows)
    _write_csv(run_dir / "routed_step_matrix.csv", rows)
    _json_dump(run_dir / "routed_summary.json", payload)
    _write_csv(run_dir / "routed_summary.csv", summary_rows)
    _write_csv(run_dir / "routing_trace.csv", routing_rows)
    _json_dump(run_dir / "routing_metrics.json", metrics)
    _json_dump(run_dir / "rollback_check.json", rollback)
    return payload


def _score(payload: Dict[str, Any]) -> Dict[str, Any]:
    final = payload["final_summary"]
    metrics = payload["routing_metrics"]
    rollback = payload["rollback_check"]
    config = payload["config"]
    new = final.get("mean_new_answer_nll_decrease")
    ref = final.get("mean_ref_abs")
    positive = int(final.get("positive_new_answer_edits") or 0)
    damage = int(final.get("locality_damage_records") or 0)
    rollback_rate = final.get("temporary_rollback_pass_rate")
    match = final.get("record_id_match_rate")
    nan = int(final.get("nan_inf_count") or 0)
    self_hit = metrics.get("self_hit_rate")
    ref_false = metrics.get("reference_false_activation_rate")
    future = metrics.get("future_activation_rate")
    basic = positive >= 18 and new is not None and float(new) > 0.0 and ref is not None and float(ref) <= 0.10 and damage <= 8 and rollback_rate == 1.0 and match == 1.0 and nan == 0
    strong = basic and float(new) >= 0.60 and float(ref) <= 0.08 and self_hit is not None and float(self_hit) >= 0.70 and ref_false is not None and float(ref_false) <= 0.30
    breakthrough = strong and float(new) >= 0.80 and damage <= 5 and float(self_hit) >= 0.80 and float(ref_false) <= 0.20 and future is not None and float(future) <= 0.30
    status = "breakthrough" if breakthrough else ("strong_pass" if strong else ("basic_pass" if basic else "fail"))
    return {
        "config_id": config["config_id"],
        "bank_config_id": config["bank_config_id"],
        "route_policy": config["route_policy"],
        "prototype_type": config.get("prototype_type"),
        "threshold": config.get("threshold"),
        "margin": config.get("margin"),
        "lambda_ref": config.get("lambda_ref"),
        "z_threshold": config.get("z_threshold"),
        "final_new": new,
        "final_ref": ref,
        "positive_new": positive,
        "locality_damage": damage,
        "previous_edit_retention": final.get("previous_edit_retention"),
        "mean_previous_edit_forgetting": final.get("mean_previous_edit_forgetting"),
        "retention_ratio": final.get("retention_ratio"),
        "self_hit_rate": self_hit,
        "top1_self_rate": metrics.get("top1_self_rate"),
        "reference_false_activation_rate": ref_false,
        "future_activation_rate": future,
        "temporary_rollback_pass_rate": rollback_rate,
        "record_id_match_rate": match,
        "nan_inf_count": nan,
        "basic_pass": basic,
        "strong_pass": strong,
        "breakthrough": breakthrough,
        "reference_rejection_success": ref_false is not None and float(ref_false) <= 0.20 and self_hit is not None and float(self_hit) >= 0.70,
        "oracle_new_gap": None if new is None else ORACLE_FINAL_NEW - float(new),
        "oracle_ref_gap": None if ref is None else float(ref) - ORACLE_FINAL_REF,
        "oracle_self_hit_gap": None if self_hit is None else ORACLE_SELF_HIT - float(self_hit),
        "rollback_status": rollback.get("status"),
        "status": status,
    }


def _choose_best(scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    learned = [row for row in scores if str(row.get("config_id", "")).startswith("R_") and row.get("config_id") != "R_oracle_self_high"]
    if not learned:
        return None
    return sorted(
        learned,
        key=lambda row: (
            1 if row.get("breakthrough") else 0,
            1 if row.get("strong_pass") else 0,
            1 if row.get("basic_pass") else 0,
            -float(row.get("reference_false_activation_rate") if row.get("reference_false_activation_rate") is not None else 999.0),
            float(row.get("self_hit_rate") or -1.0),
            float(row.get("final_new") or -999.0),
            -float(row.get("final_ref") or 999.0),
        ),
        reverse=True,
    )[0]


def _write_bank_reuse_report(out_dir: Path, records: List[Dict[str, Any]], status: Dict[str, Any]) -> None:
    record_ids = [_record_id(record) for record in records]
    _json_dump(out_dir / "audit" / "selected_record_ids.json", {"record_ids": record_ids, "count": len(record_ids)})
    _json_dump(out_dir / "audit" / "bank_reuse_status.json", status)
    lines = [
        "# Data And Bank Reuse Report",
        "",
        f"- Selected records: `{len(record_ids)}`",
        "- Source: `outputs/medmkeb_engram_projected_lora/modelknown_20/medmkeb_modelknown_20.json`",
        "- Positional matching used: `False`",
        f"- Bank status: `{status.get('status')}`",
        f"- Bank source: `{status.get('source') or status.get('bank_dir')}`",
        f"- Record-id match rate: `{status.get('record_id_match_rate')}`",
        "",
        "## Record IDs",
        "",
    ]
    lines.extend(f"- `{rid}`" for rid in record_ids)
    (out_dir / "audit" / "DATA_AND_BANK_REUSE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_optional(out_dir: Path, scores: List[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]], best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    made: List[str] = []
    try:
        routed = [row for row in scores if row.get("final_new") is not None and row.get("final_ref") is not None]
        if routed:
            for x_key, y_key, filename, xlabel, ylabel in [
                ("final_ref", "final_new", "ref_vs_new_scatter.png", "final_ref", "final_new"),
                ("self_hit_rate", "final_new", "self_hit_vs_new.png", "self_hit_rate", "final_new"),
                ("reference_false_activation_rate", "final_ref", "reference_false_activation_vs_ref.png", "reference_false_activation_rate", "final_ref"),
            ]:
                plt.figure(figsize=(7, 5))
                plt.scatter([float(row.get(x_key) or 0.0) for row in routed], [float(row.get(y_key) or 0.0) for row in routed])
                for row in routed:
                    plt.annotate(str(row["config_id"]).replace("_", "\n"), (float(row.get(x_key) or 0.0), float(row.get(y_key) or 0.0)), fontsize=5)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.tight_layout()
                path = out_dir / "plots" / filename
                plt.savefig(path)
                plt.close()
                made.append(str(path))
            plt.figure(figsize=(8, 5))
            xs = range(len(routed))
            plt.bar(xs, [float(row.get("self_hit_rate") or 0.0) for row in routed], label="self_hit")
            plt.bar(xs, [float(row.get("reference_false_activation_rate") or 0.0) for row in routed], bottom=[float(row.get("self_hit_rate") or 0.0) for row in routed], label="ref_false")
            plt.xticks(list(xs), [str(row["config_id"]).replace("R_", "") for row in routed], rotation=80, fontsize=6)
            plt.ylabel("routing rates")
            plt.legend()
            plt.tight_layout()
            path = out_dir / "plots" / "routing_confusion_by_config.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if best and str(best["config_id"]) in payloads:
            rows = payloads[str(best["config_id"])]["summary_rows"]
            xs = [int(row.get("step") or 0) for row in rows]
            for key, filename, ylabel in [
                ("mean_ref_abs", "sequential_reference_curve_best.png", "reference delta abs"),
                ("mean_new_answer_nll_decrease", "sequential_new_curve_best.png", "new-answer NLL decrease"),
                ("mean_previous_edit_forgetting", "sequential_forgetting_curve_best.png", "previous-edit forgetting"),
            ]:
                plt.figure(figsize=(7, 4))
                plt.plot(xs, [float(row.get(key) or 0.0) for row in rows], marker="o")
                plt.xlabel("step")
                plt.ylabel(ylabel)
                plt.tight_layout()
                path = out_dir / "plots" / filename
                plt.savefig(path)
                plt.close()
                made.append(str(path))
        return {"status": "complete", "files": made}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "files": made}


def _write_best_analysis(out_dir: Path, best: Optional[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]]) -> None:
    if not best:
        (out_dir / "BEST_ROUTING_CONFIG_ANALYSIS.md").write_text("# Best Routing Config Analysis\n\nNo learned config completed.\n", encoding="utf-8")
        _write_csv(out_dir / "best_routing_per_record.csv", [])
        return
    payload = payloads[str(best["config_id"])]
    rows = [row for row in payload["per_record_step_rows"] if int(row.get("step") or -1) == 20]
    failures = [row for row in rows if row.get("is_edited_so_far") and not row.get("self_edit_active")]
    false_refs = [row for row in rows if (row.get("reference_active_edit_count") or 0) > 0]
    _write_csv(out_dir / "best_routing_per_record.csv", rows)
    lines = [
        "# Best Routing Config Analysis",
        "",
        f"- Best learned config: `{best['config_id']}`",
        f"- Status: `{best['status']}`",
        f"- final_new: `{_format(best.get('final_new'))}`",
        f"- final_ref: `{_format(best.get('final_ref'))}`",
        f"- self_hit_rate: `{_format(best.get('self_hit_rate'))}`",
        f"- reference_false_activation_rate: `{_format(best.get('reference_false_activation_rate'))}`",
        f"- future_activation_rate: `{_format(best.get('future_activation_rate'))}`",
        f"- oracle_new_gap: `{_format(best.get('oracle_new_gap'))}`",
        f"- oracle_ref_gap: `{_format(best.get('oracle_ref_gap'))}`",
        "",
        "## Failure Analysis",
        "",
        f"- Self-hit failures: `{len(failures)}`",
        f"- Reference false activations: `{len(false_refs)}`",
        "",
        "## Self-Hit Failures",
        "",
    ]
    lines.extend([f"- `{row['record_id']}` top1={row.get('top1_edit_id')} reason={row.get('reject_reason')}" for row in failures[:20]] or ["- None at final step."])
    lines.extend(["", "## Reference False Activations", ""])
    lines.extend([f"- `{row['record_id']}` active={row.get('reference_active_edit_ids')}" for row in false_refs[:20]] or ["- None at final step."])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Margin and two-stage routing are judged by whether they reduce reference false activation without losing self-hit.",
            "- `R_oracle_reference_reject_top1` separates target routing quality from reference rejection quality.",
            "- If learned configs remain below oracle, the bottleneck is router/prototype quality rather than per-edit delta availability.",
        ]
    )
    (out_dir / "BEST_ROUTING_CONFIG_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    *,
    out_dir: Path,
    scores: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    data_reuse: Dict[str, Any],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
) -> None:
    learned = [row for row in scores if row.get("config_id") != "R_oracle_self_high"]
    strong = [row for row in learned if row.get("strong_pass") or row.get("breakthrough")]
    if strong:
        decision = "A. Learned routed bank reaches strong or breakthrough. Next: validate on 50 MedMKEB model-known edits or external Med-VQA."
    else:
        oracle_ref = next((row for row in scores if row.get("config_id") == "R_oracle_self_high"), None)
        if oracle_ref and oracle_ref.get("strong_pass"):
            decision = "B. Oracle is strong but learned routing remains partial. Next: improve prototype learning or train a small router."
        else:
            decision = "D. Learned routing remains weak and oracle gap is large. Keep routed bank as promising upper bound; implement learned router."
    lines = [
        "# Final MedMKEB Routed Bank Refine 20 Report",
        "",
        "## Starting Point",
        "",
        "- Global merge sequential runs had high locality damage.",
        "- Crisp/CURE did not reach breakthrough.",
        "- Routed-bank oracle reached strong pass.",
        "- Learned routing bottleneck was reference rejection.",
        "",
        "## Data",
        "",
        f"- Exact reused records: `{data_reuse.get('selected_record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- No medical or clinical efficacy claim is made.",
        "",
        "## Method",
        "",
        "- Per-edit ENGRAM-projected LoRA bank; no global merge.",
        "- Target/reference/contrastive prototypes are used for routing and rejection.",
        "- Deltas are applied with temporary forward patches and rolled back per query.",
        "- Thresholds are calibrated on this bounded 20-record development set.",
        "",
        "## Main Results",
        "",
        "| config_id | status | final_new | final_ref | positive_new | damage | self_hit | ref_false | future_active | rollback | match | nan |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores:
        lines.append(
            "| {config_id} | {status} | {new} | {ref} | {positive} | {damage} | {self_hit} | {ref_false} | {future} | {rollback} | {match} | {nan} |".format(
                config_id=row.get("config_id"),
                status=row.get("status"),
                new=_format(row.get("final_new")),
                ref=_format(row.get("final_ref")),
                positive=row.get("positive_new"),
                damage=row.get("locality_damage"),
                self_hit=_format(row.get("self_hit_rate")),
                ref_false=_format(row.get("reference_false_activation_rate")),
                future=_format(row.get("future_activation_rate")),
                rollback=_format(row.get("temporary_rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(["", "## Best Learned Config", ""])
    if best:
        lines.extend(
            [
                f"- Config: `{best['config_id']}`",
                f"- Status: `{best['status']}`",
                f"- final_new: `{_format(best.get('final_new'))}`",
                f"- final_ref: `{_format(best.get('final_ref'))}`",
                f"- self_hit_rate: `{_format(best.get('self_hit_rate'))}`",
                f"- reference_false_activation_rate: `{_format(best.get('reference_false_activation_rate'))}`",
                f"- oracle_new_gap: `{_format(best.get('oracle_new_gap'))}`",
                f"- oracle_ref_gap: `{_format(best.get('oracle_ref_gap'))}`",
            ]
        )
    else:
        lines.append("- No learned config completed.")
    lines.extend(
        [
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
            "- NLL/logprob primary evidence.",
            "- Generation diagnostic only.",
            "- No medical or clinical efficacy claim.",
            "- Routing uses development-set calibration.",
            "- Routed bank adds inference-time routing cost.",
        ]
    )
    (out_dir / "FINAL_MEDMKEB_ROUTED_BANK_REFINE_20_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    records = _load_records(Path(args.selected_records))[: int(args.max_records)]
    _write_bank_reuse_report(out_dir, records, {"status": "prepare_only", "record_id_match_rate": 1.0})
    _json_dump(out_dir / "routed_refine_config_grid.json", {"bank_configs": _bank_configs(), "routing_configs": _refine_configs(compact=not args.full_grid)})
    _write_final_report(out_dir=out_dir, scores=[], best=None, data_reuse={"selected_record_count": len(records), "record_id_match_rate": 1.0}, generation={"status": "skipped"}, plots={"status": "skipped"})
    _write_package_hygiene_report(out_dir)
    return 0


def _run_gpu(args: argparse.Namespace) -> int:
    import torch

    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)
    preflight = _write_preflight(out_dir, hparams_path=Path(args.hparams), input_records=Path(args.selected_records), image_root=Path(args.image_root), test_status=test_status)
    records = _load_records(Path(args.selected_records))[: int(args.max_records)]
    _json_dump(out_dir / "routed_refine_config_grid.json", {"bank_configs": _bank_configs(), "routing_configs": _refine_configs(compact=not args.full_grid)})
    if test_status.get("status") != "pass" or preflight.get("status") != "pass":
        _json_dump(out_dir / "runtime.json", {"status": "blocked_preflight", "test_status": test_status, "preflight": preflight})
        return 2

    heavy = _heavy_imports()
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    EngramBank = heavy["EngramBank"]
    select_linear_layers = heavy["select_linear_layers"]
    _extract_projector_bank = heavy["_extract_projector_bank"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    module_names = _module_scope_names("qk_gate_sampled_depths")
    projector_bank_dir = out_dir / "runtime_projector_banks" / "qk_gate_sampled_depths"
    _configure_hparams_for_scope(hparams=hparams, image_root=Path(args.image_root), bank_dir=projector_bank_dir, device=str(args.device), module_names=module_names, lora_steps=20, lora_ref_loss_weight=0.0)
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    if set(selected) != set(module_names):
        raise RuntimeError({"reason": "selected module mismatch", "selected": selected, "expected": module_names})
    _json_dump(out_dir / "audit" / "selected_modules_qk_gate_sampled_depths.json", {"status": "pass", "selected_modules": selected})
    bank_status = _existing_projector_bank_matches(EngramBank, projector_bank_dir, records)
    if bank_status.get("status") != "complete":
        bank_status = _extract_projector_bank(editor, hparams, Path(args.selected_records), records, projector_bank_dir)
        bank_status["reused"] = False
    _json_dump(out_dir / "audit" / "runtime_projector_bank_status.json", bank_status)

    baselines = _read_json(Path(args.baseline_metrics))
    entries, edit_status = _build_edit_bank(model=editor.model, records=records, image_root=Path(args.image_root), hparams=hparams, projector_bank_dir=projector_bank_dir, out_dir=out_dir, bank_config=_bank_configs()[0])
    _json_dump(out_dir / "bank_metadata" / "bank_high_strength_status.json", edit_status)
    _write_bank_reuse_report(out_dir, records, {"status": edit_status.get("status"), "source": "rebuilt_high_strength_bank" if not edit_status.get("reused") else "cached_runtime_edit_bank", "record_id_match_rate": 1.0, "entry_count": len(entries)})
    query_cache = _build_query_cache(model=editor.model, records=records, image_root=Path(args.image_root), baselines=baselines, module_names=module_names, out_dir=out_dir)
    calibration = _calibration_stats(records=records, entries=entries, query_cache=query_cache)
    _json_dump(out_dir / "routing_calibration_stats.json", calibration)
    _write_csv(out_dir / "routing_calibration_stats.csv", calibration["rows"])

    payloads: Dict[str, Dict[str, Any]] = {}
    scores: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for config in _refine_configs(compact=not args.full_grid):
        payload = _evaluate_config(model=editor.model, records=records, baselines=baselines, entries=entries, config=config, run_dir=out_dir / "runs" / str(config["config_id"]), module_names=module_names, query_cache=query_cache, calibration=calibration, rollback_tolerance=float(args.rollback_tolerance), locality_threshold=float(args.locality_damage_threshold))
        payloads[str(config["config_id"])] = payload
        score = _score(payload)
        scores.append(score)
        metric_rows.append(payload["routing_metrics"])
        _json_dump(out_dir / "runs" / str(config["config_id"]) / "score.json", score)
    best = _choose_best(scores)
    _write_csv(out_dir / "routed_refine_summary.csv", scores)
    _json_dump(out_dir / "routed_refine_summary.json", {"scores": scores})
    _write_csv(out_dir / "routing_metrics.csv", metric_rows)
    _json_dump(out_dir / "routing_metrics.json", {"metrics": metric_rows})
    _write_best_analysis(out_dir, best, payloads)
    plots = _plot_optional(out_dir, scores, payloads, best)
    generation = {"status": "skipped", "reason": "main NLL/logprob gate used --skip-generation"}
    _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", generation)
    _write_final_report(out_dir=out_dir, scores=scores, best=best, data_reuse={"selected_record_count": len(records), "record_id_match_rate": 1.0}, generation=generation, plots=plots)
    _json_dump(out_dir / "runtime.json", {"status": "complete", "best": best})
    _write_package_hygiene_report(out_dir)
    return 0


def _generation_text_metrics(method: str, record: Dict[str, Any], idx: int, generation: Dict[str, Any], route: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    decoded = str(generation.get("decoded_stripped") or generation.get("decoded_skip_special") or "")
    old = str(record.get("old_answer") or record.get("pred") or "")
    new = str(record.get("new_answer") or record.get("target_new") or "")
    route = route or {}
    return {
        "method": method,
        "record_id": _record_id(record),
        "case_index": idx,
        "prompt": record.get("src") or record.get("prompt"),
        "old_answer": old,
        "new_answer": new,
        "generation": decoded,
        "generation_empty": bool(generation.get("generation_empty")),
        "contains_old_answer": old.casefold() in decoded.casefold() if old else False,
        "contains_new_answer": new.casefold() in decoded.casefold() if new else False,
        "exact_new_answer": decoded.strip().casefold() == new.strip().casefold() if new else False,
        "simple_casefold_contains": new.casefold() in decoded.casefold() if new else False,
        "active_edit_ids": route.get("active_edit_ids", []),
        "self_edit_active": route.get("self_edit_active", False),
        "notes": "generation diagnostic only; not primary gate",
    }


def _run_generation(args: argparse.Namespace) -> int:
    import torch

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generation_diagnostics"
    gen_dir.mkdir(parents=True, exist_ok=True)
    records_all = _load_records(Path(args.selected_records))[: int(args.max_records)]
    records = records_all[: int(args.generation_records)]
    summary = _read_json(out_dir / "routed_refine_summary.json")
    scores = summary["scores"]
    best = _choose_best(scores)
    if not best or not best.get("basic_pass"):
        payload = {"status": "skipped", "reason": "no learned routing config reached basic pass"}
        _json_dump(gen_dir / "generation_5records.json", payload)
        return 0
    config_ids = ["R_oracle_self_high", "R_top3_target_adaptive_high", str(best["config_id"])]
    configs = [cfg for cfg in _refine_configs(compact=False) if cfg["config_id"] in set(config_ids)]
    heavy = _heavy_imports()
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    EngramBank = heavy["EngramBank"]
    select_linear_layers = heavy["select_linear_layers"]
    _extract_projector_bank = heavy["_extract_projector_bank"]
    _evaluate_current = heavy["_evaluate_current"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    module_names = _module_scope_names("qk_gate_sampled_depths")
    projector_bank_dir = gen_dir / "runtime_projector_banks" / "qk_gate_sampled_depths_5records"
    selected_path = gen_dir / "generation_selected_records_5.json"
    _json_dump(selected_path, records)
    _configure_hparams_for_scope(hparams=hparams, image_root=Path(args.image_root), bank_dir=projector_bank_dir, device=str(args.device), module_names=module_names, lora_steps=20, lora_ref_loss_weight=0.0)
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    if set(selected) != set(module_names):
        raise RuntimeError({"reason": "selected module mismatch", "selected": selected, "expected": module_names})
    bank_status = _existing_projector_bank_matches(EngramBank, projector_bank_dir, records)
    if bank_status.get("status") != "complete":
        bank_status = _extract_projector_bank(editor, hparams, selected_path, records, projector_bank_dir)
    _json_dump(gen_dir / "generation_projector_bank_status.json", bank_status)
    entries, edit_status = _build_edit_bank(model=editor.model, records=records, image_root=Path(args.image_root), hparams=hparams, projector_bank_dir=projector_bank_dir, out_dir=gen_dir, bank_config=_bank_configs()[0])
    _json_dump(gen_dir / "generation_edit_bank_status.json", edit_status)
    query_cache = _build_query_cache(model=editor.model, records=records, image_root=Path(args.image_root), baselines={_record_id(record): {} for record in records}, module_names=module_names, out_dir=gen_dir)
    calibration = _calibration_stats(records=records, entries=entries, query_cache=query_cache)
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        baseline = _evaluate_current(editor.model, record, Path(args.image_root), max_new_tokens=int(args.generation_max_new_tokens), min_new_tokens=None, skip_generation=False)
        rows.append(_generation_text_metrics("baseline", record, idx, baseline.get("generation") or {}))
        for config in configs:
            threshold = _threshold_value(config.get("threshold"), calibration, _prototype_key(str(config.get("prototype_type") or "target")))
            target_threshold = _threshold_value(config.get("target_threshold"), calibration, "target")
            route = _route_refined(query=query_cache[(_record_id(record), "new")]["query"], entries=entries, query_record_id=_record_id(record), query_kind="new", config=config, threshold=threshold, target_threshold=target_threshold, calibration=calibration)
            active_by_id = {entry["edit_id"]: entry for entry in entries}
            active_entries = [active_by_id[edit_id] for edit_id in route["active_edit_ids"]]
            patch = RoutedLoraPatch(editor.model, active_entries, [1.0 for _ in active_entries])
            patch.install()
            try:
                result = _evaluate_current(editor.model, record, Path(args.image_root), max_new_tokens=int(args.generation_max_new_tokens), min_new_tokens=None, skip_generation=False)
            finally:
                patch.remove()
            rows.append(_generation_text_metrics(str(config["config_id"]), record, idx, result.get("generation") or {}, route))
    payload = {"status": "complete", "primary_gate": False, "best_config_id": best["config_id"], "rows": rows}
    _json_dump(gen_dir / "generation_5records.json", payload)
    _write_csv(gen_dir / "generation_5records.csv", rows)
    for path in [gen_dir / "runtime_projector_banks", gen_dir / "runtime_edit_banks"]:
        if path.exists():
            shutil.rmtree(path)
    plots = {"status": "complete"}
    _write_final_report(out_dir=out_dir, scores=scores, best=best, data_reuse={"selected_record_count": len(records_all), "record_id_match_rate": 1.0}, generation=payload, plots=plots)
    _write_package_hygiene_report(out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded MedMKEB routed-bank routing/ref-rejection refinement.")
    parser.add_argument("--mode", choices=["prepare", "run-gpu", "generation"], default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / REFINE_DIRNAME))
    parser.add_argument("--selected-records", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "medmkeb_modelknown_20.json"))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "baseline_metrics.json"))
    parser.add_argument("--reuse-bank-dir", default=str(PREVIOUS_ROUTED_DIR))
    parser.add_argument("--image-root", default="/Volumes/DataP/knowledge_editing/data/medmkeb/images")
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--generation-records", type=int, default=5)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--full-grid", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "prepare":
        return _prepare(args)
    if args.mode == "generation":
        return _run_generation(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
