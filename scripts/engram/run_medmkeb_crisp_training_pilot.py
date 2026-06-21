#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import subprocess
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
    _finite,
    _format,
    _heavy_imports,
    _json_dump,
    _mean,
    _package_hygiene,
    _plot_optional,
    _run_pytest,
    _safe_div,
    _write_csv,
    _write_env_report,
    _write_git_outputs,
    _write_preflight,
)
from scripts.engram.run_medmkeb_sequential_pareto_refine import (  # noqa: E402
    _aggregate_step,
    _configure_hparams_for_scope,
    _enrich_row,
    _evaluate_step,
    _load_records,
    _module_names_for_scope,
    _read_json,
    _write_data_reuse_report,
)


PILOT_DIRNAME = "crisp_training_pilot_20"
METHOD_C = "C_engram_projected_tiny_lora"
METHOD_T1 = "ENGRAM_projected_LoRA_with_training_time_Crisp_projection"
METHOD_T2 = "ENGRAM_projected_LoRA_with_prev_aware_Crisp_projection"
METHOD_T3 = "ENGRAM_projected_LoRA_with_reference_previous_constraints"

QK_GATE_MODULES = list(EXPECTED_MODULES)
QK_ONLY_MODULES = [name for name in EXPECTED_MODULES if name.endswith(".q_proj") or name.endswith(".k_proj")]
GATE_ONLY_MODULES = [name for name in EXPECTED_MODULES if name.endswith(".gate_proj")]

C_HIGH_NEW = 1.69794
C_HIGH_REF = 0.293396
C_HIGH_DAMAGE = 18
C_LOW_NEW = 0.302485
C_LOW_REF = 0.0387687
C_LOW_DAMAGE = 6
C_BOUNDED_NEW = 0.361779
C_BOUNDED_REF = 0.048950
C_BOUNDED_DAMAGE = 7


def _run_capture(command: List[str], cwd: Path = PROJECT_ROOT) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return proc.stdout
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}\n"


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


def _record_id(record: Dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id"))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_tests_pilot(out_dir: Path, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    if not run_tests:
        payload = {"status": "skipped", "reason": "--skip-tests"}
        _json_dump(test_dir / "test_status.json", payload)
        return payload
    engram_tests = sorted(str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.glob("tests/test_engram_*.py"))
    runs = [
        _run_pytest(test_dir / "test_engram_all.log", [*engram_tests, "-q"]),
        _run_pytest(test_dir / "test_cure_crisp_projection.log", ["tests/test_cure_crisp_projection.py", "-q"]),
        _run_pytest(test_dir / "test_cure_kfac_collector_tiny_mllm.log", ["tests/test_cure_kfac_collector_tiny_mllm.py", "-q"]),
    ]
    payload = {
        "status": "pass" if all(item["returncode"] == 0 for item in runs) else "fail",
        "engram_tests_pass": runs[0]["returncode"] == 0,
        "cure_projection_tests_pass": runs[1]["returncode"] == 0,
        "cure_kfac_tests_pass": runs[2]["returncode"] == 0,
        "cure_tests_pass": runs[1]["returncode"] == 0 and runs[2]["returncode"] == 0,
        "runs": runs,
    }
    _json_dump(test_dir / "test_status.json", payload)
    return payload


def _pilot_config_grid() -> List[Dict[str, Any]]:
    shared = {
        "module_scope": "qk_gate_sampled_depths",
        "token_scope": "all",
        "selected_depths": [0, 8, 16, 24],
        "lora_ref_loss_weight": 0.0,
        "prev_edit_loss_weight": 0.0,
        "skip_generation": True,
        "record_id_metrics": True,
    }
    anchors = [
        {
            **shared,
            "config_id": "Anchor_C_high_strength",
            "method": METHOD_C,
            "beta": 0.5,
            "lora_steps": 20,
            "crisp_projection_scope": "none",
            "projection_mode": "none",
            "cache_source": "none",
            "gate_policy": "engram_projected",
            "anchor": True,
        },
        {
            **shared,
            "config_id": "Anchor_C_low_drift",
            "method": METHOD_C,
            "beta": 0.3,
            "lora_steps": 10,
            "crisp_projection_scope": "none",
            "projection_mode": "none",
            "cache_source": "none",
            "gate_policy": "engram_projected",
            "anchor": True,
        },
        {
            **shared,
            "config_id": "Anchor_C_bounded",
            "method": METHOD_C,
            "beta": 0.35,
            "lora_steps": 10,
            "crisp_projection_scope": "none",
            "projection_mode": "none",
            "cache_source": "none",
            "gate_policy": "engram_projected",
            "anchor": True,
        },
    ]
    t1 = {
        **shared,
        "config_id": "T1_crisp_projected_training_qk_ref",
        "method": METHOD_T1,
        "beta": 0.5,
        "lora_steps": 20,
        "crisp_projection_scope": "qk_only",
        "gate_policy": "engram_fallback",
        "crisp_cache_source": "reference_only",
        "cache_source": "reference_only",
        "projection_mode": "training_time_effective_delta_projection",
        "norm_clamp": True,
        "max_projection_norm_ratio": 1.0,
    }
    t2 = {
        **shared,
        "config_id": "T2_crisp_projected_training_qk_ref_prev_hard",
        "method": METHOD_T2,
        "beta": 0.5,
        "lora_steps": 20,
        "crisp_projection_scope": "qk_only",
        "gate_policy": "engram_fallback",
        "crisp_cache_source": "reference_plus_previous_plus_hard_locality",
        "cache_source": "reference_plus_previous_plus_hard_locality",
        "cache_update_policy": "streaming_average",
        "hard_locality_topk": 5,
        "previous_edit_cache_size": "all_previous_edits",
        "projection_mode": "training_time_effective_delta_projection",
        "norm_clamp": True,
        "max_projection_norm_ratio": 1.0,
    }
    t3 = {
        **shared,
        "config_id": "T3_constraint_only_ref_prev",
        "method": METHOD_T3,
        "beta": 0.5,
        "lora_steps": 20,
        "crisp_projection_scope": "none_or_diagnostic",
        "gate_policy": "engram_projected",
        "use_posthoc_cure_delta": False,
        "cache_source": "constraint_only",
        "projection_mode": "none",
        "reference_loss_weight": 0.05,
        "previous_edit_loss_weight": 0.05,
        "hard_locality_loss_weight": 0.05,
        "norm_clamp": True,
    }

    def low_strength(config: Dict[str, Any], config_id: str) -> Dict[str, Any]:
        copied = copy.deepcopy(config)
        copied["config_id"] = config_id
        copied["beta"] = 0.35
        copied["lora_steps"] = 10
        return copied

    return [
        *anchors,
        t1,
        t2,
        t3,
        low_strength(t1, "T1_crisp_projected_training_qk_ref_beta0.35_steps10"),
        low_strength(t2, "T2_crisp_projected_training_qk_ref_prev_hard_beta0.35_steps10"),
        low_strength(t3, "T3_constraint_only_ref_prev_beta0.35_steps10"),
    ]


def _module_memory_estimates() -> List[Dict[str, Any]]:
    estimates: List[Dict[str, Any]] = []
    for name in QK_GATE_MODULES:
        if name.endswith(".gate_proj"):
            in_dim, out_dim = 4096, 14336
            policy = "engram_fallback_or_diag_only"
        else:
            in_dim, out_dim = 4096, 4096
            policy = "allow_full_kfac_projection_if_cache_available"
        factor_bytes = 4 * (in_dim * in_dim + out_dim * out_dim)
        eig_bytes = 4 * (in_dim * in_dim + out_dim * out_dim)
        mask_bytes = in_dim * out_dim
        total_gib = (factor_bytes + eig_bytes + mask_bytes) / float(1024**3)
        estimates.append(
            {
                "module_name": name,
                "in_dim_estimate": in_dim,
                "out_dim_estimate": out_dim,
                "kfac_factor_gib": factor_bytes / float(1024**3),
                "eigenspace_gib": eig_bytes / float(1024**3),
                "mask_gib_bool_estimate": mask_bytes / float(1024**3),
                "total_cache_gib_estimate": total_gib,
                "policy": policy,
            }
        )
    return estimates


def _write_feasibility_audit(out_dir: Path, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _module_memory_estimates()
    qk_total = sum(float(row["total_cache_gib_estimate"]) for row in rows if row["module_name"] in QK_ONLY_MODULES)
    gate_total = sum(float(row["total_cache_gib_estimate"]) for row in rows if row["module_name"] in GATE_ONLY_MODULES)
    payload = {
        "status": "complete",
        "selected_safe_projection_mode": "training_time_effective_delta_projection",
        "exact_lora_factor_gradient_projection_feasible": False,
        "full_weight_projected_gradient_qk_only_feasible": "possible_but_not_default",
        "qk_projection_cache_total_gib_estimate": qk_total,
        "gate_projection_cache_total_gib_estimate": gate_total,
        "gate_policy": "ENGRAM fallback or diag-only safe mode; full-B K-FAC is not used",
        "configs": configs,
        "module_memory_estimates": rows,
    }
    _json_dump(out_dir / "audit" / "crisp_training_feasibility_audit.json", payload)
    lines = [
        "# Crisp Training Feasibility Audit",
        "",
        "## Answers",
        "",
        "1. Full selected Linear weight gradient projection is safe only for q/k modules under the bounded 4096-dimensional cache setting; it is not the default because the existing LoRA helper optimizes factors, not full weights.",
        "2. LoRA effective updates can be projected safely as dense effective deltas for q/k modules. This runner labels that mode `training_time_effective_delta_projection`.",
        "3. There is no exact general mapping from full-weight projected gradients back to low-rank LoRA factor gradients after AdamW updates. T1/T2 therefore use a bounded shadow effective-delta approximation, not exact LoRA gradient projection.",
        "4. q/k modules can be projected with full K-FAC caches when cache collection succeeds.",
        "5. gate_proj modules need fallback because their output dimension makes full B-factor eigenspaces memory-heavy.",
        "6. gate_proj uses ENGRAM fallback by default; diag/low-rank gate CURE can be added later but full-B K-FAC is intentionally not implemented here.",
        f"7. Estimated q/k projection cache footprint: `{qk_total:.3f}` GiB across sampled q/k modules. Estimated gate footprint: `{gate_total:.3f}` GiB across sampled gate modules.",
        "",
        "## Distinction From Post-Hoc Projection",
        "",
        "- T1/T2 project each optimizer-step effective LoRA delta into a shadow dense update and clamp projected norms.",
        "- This is not exact LoRA gradient projection.",
        "- This is not post-hoc final delta projection, because the trace records one projection decision per optimizer step and module.",
        "",
        "## Module Estimates",
        "",
        "| module | in | out | total_cache_GiB | policy |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {module} | {in_dim} | {out_dim} | {gib:.3f} | {policy} |".format(
                module=row["module_name"],
                in_dim=row["in_dim_estimate"],
                out_dim=row["out_dim_estimate"],
                gib=row["total_cache_gib_estimate"],
                policy=row["policy"],
            )
        )
    (out_dir / "audit" / "CRISP_TRAINING_FEASIBILITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _alias_payload(payload: Dict[str, Any], alias: str, source_config_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    copied = copy.deepcopy(payload)
    copied["config"] = dict(config, source_config_id=source_config_id, anchor_reused=True)
    copied["status"] = copied.get("status", "complete")
    if isinstance(copied.get("final_summary"), dict):
        copied["final_summary"].update(
            {
                "config_id": alias,
                "method": config["method"],
                "beta": config["beta"],
                "lora_steps": config["lora_steps"],
                "module_scope": config["module_scope"],
                "source_config_id": source_config_id,
                "anchor_reused": True,
                "projection_mode": config.get("projection_mode"),
                "cache_source": config.get("cache_source"),
            }
        )
    for key in ("summary_rows", "per_record_step_rows"):
        for row in copied.get(key, []) or []:
            if isinstance(row, dict):
                row.update(
                    {
                        "config_id": alias,
                        "method": config["method"],
                        "beta": config["beta"],
                        "lora_steps": config["lora_steps"],
                        "module_scope": config["module_scope"],
                        "source_config_id": source_config_id,
                        "anchor_reused": True,
                        "projection_mode": config.get("projection_mode"),
                        "cache_source": config.get("cache_source"),
                        "crisp_projection_scope": config.get("crisp_projection_scope"),
                        "gate_policy": config.get("gate_policy"),
                    }
                )
    return copied


def _write_payload_files(run_dir: Path, payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    _json_dump(run_dir / "config.json", config)
    _json_dump(run_dir / "sequential_step_matrix.json", payload.get("per_record_step_rows", []))
    _write_csv(run_dir / "sequential_step_matrix.csv", payload.get("per_record_step_rows", []))
    _json_dump(run_dir / "sequential_summary.json", payload)
    _write_csv(run_dir / "sequential_summary.csv", payload.get("summary_rows", []))
    _json_dump(run_dir / "final_rollback_check.json", payload.get("final_rollback_check", {}))
    if not (run_dir / "projection_trace.json").exists():
        _json_dump(run_dir / "projection_trace.json", {"status": "not_applicable", "rows": []})
    if not (run_dir / "constraint_loss_trace.json").exists():
        _json_dump(run_dir / "constraint_loss_trace.json", {"status": "not_applicable", "rows": []})


def _load_anchor_payloads(out_dir: Path, configs: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    sources = {
        "Anchor_C_high_strength": (
            DEFAULT_OUTPUT_DIR / "sequential_pareto_refine_20" / "runs" / "C_baseline_reproduce",
            "C_baseline_reproduce",
        ),
        "Anchor_C_low_drift": (
            DEFAULT_OUTPUT_DIR / "sequential_rescue_20" / "runs" / "C_beta0.3_steps10_qkgate_ref0",
            "C_beta0.3_steps10_qkgate_ref0",
        ),
        "Anchor_C_bounded": (
            DEFAULT_OUTPUT_DIR / "sequential_pareto_refine_20" / "runs" / "C_beta0.35_steps10_qkgate_ref0",
            "C_beta0.35_steps10_qkgate_ref0",
        ),
    }
    config_by_id = {str(config["config_id"]): config for config in configs}
    payloads: Dict[str, Dict[str, Any]] = {}
    scores: List[Dict[str, Any]] = []
    lines = [
        "# Anchor Reuse",
        "",
        "Anchors are reused from completed MedMKEB sequential rescue/Pareto outputs to avoid rerunning known 7B multimodal baselines.",
        "",
        "| anchor | source | status | final_new | final_ref | damage | positive_new | rollback | match |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias, (src_dir, source_id) in sources.items():
        config = config_by_id[alias]
        run_dir = out_dir / "runs" / alias
        run_dir.mkdir(parents=True, exist_ok=True)
        src_summary = src_dir / "sequential_summary.json"
        if not src_summary.exists():
            payload = {"status": "missing_anchor", "config": config, "source_dir": str(src_dir)}
            _write_payload_files(run_dir, payload, config)
            score = _score_payload(payload, anchor_reused=True)
            payloads[alias] = payload
            scores.append(score)
            lines.append(f"| {alias} | {source_id} | missing |  |  |  |  |  |  |")
            continue
        payload = _alias_payload(_read_json(src_summary), alias, source_id, config)
        if src_dir.exists() and not (run_dir / "anchor_source_copied.flag").exists():
            for child in src_dir.iterdir():
                dst = run_dir / child.name
                if dst.exists():
                    continue
                if child.is_dir():
                    shutil.copytree(
                        child,
                        dst,
                        ignore=shutil.ignore_patterns("*.pt", "*.pth", "*.bin", "__pycache__", "*.pyc", ".DS_Store", "._*"),
                    )
                elif child.name not in {"sequential_summary.json", "sequential_step_matrix.json", "sequential_summary.csv", "sequential_step_matrix.csv"}:
                    shutil.copy2(child, dst)
            (run_dir / "anchor_source_copied.flag").write_text("ok\n", encoding="utf-8")
        _write_payload_files(run_dir, payload, config)
        _json_dump(run_dir / "anchor_reuse_metadata.json", {"source_run": str(src_dir), "source_config_id": source_id, "alias": alias})
        score = _score_payload(payload, anchor_reused=True)
        payloads[alias] = payload
        scores.append(score)
        lines.append(
            "| {alias} | {source} | reused | {new} | {ref} | {damage} | {positive} | {rollback} | {match} |".format(
                alias=alias,
                source=source_id,
                new=_format(score.get("final_new")),
                ref=_format(score.get("final_ref")),
                damage=score.get("locality_damage"),
                positive=score.get("positive_new_answer_edits"),
                rollback=score.get("rollback_pass"),
                match=_format(score.get("record_id_match_rate")),
            )
        )
    (out_dir / "audit" / "ANCHOR_REUSE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payloads, scores


def _score_payload(payload: Dict[str, Any], *, anchor_reused: bool = False) -> Dict[str, Any]:
    config = payload.get("config", {})
    final = payload.get("final_summary", {})
    new = _to_float(final.get("mean_new_answer_nll_decrease") or final.get("mean_new_answer_nll_decrease_edited_records"))
    ref = _to_float(final.get("mean_ref_abs") or final.get("mean_reference_delta_abs_all_records"))
    positive = int(final.get("positive_new_answer_edits") or 0)
    damage = int(final.get("locality_damage_records") or 0)
    rollback = (payload.get("final_rollback_check") or {}).get("status") == "pass" or bool(final.get("rollback_pass"))
    match = float(final.get("record_id_match_rate") or 0.0)
    nan = int(final.get("nan_inf_count") or 0)
    basic = (
        positive == 20
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
        positive >= 18
        and new is not None
        and new >= 0.60
        and ref is not None
        and ref <= 0.08
        and damage <= 8
        and rollback
        and match == 1.0
        and nan == 0
    )
    breakthrough = (
        positive >= 18
        and new is not None
        and new >= 0.60
        and ref is not None
        and ref <= 0.08
        and damage <= 8
        and (_safe_div(new, C_LOW_NEW) or 0.0) >= 1.5
        and (_safe_div(ref, C_HIGH_REF) or 999.0) <= 0.30
        and rollback
        and match == 1.0
        and nan == 0
    )
    if payload.get("status") in {"blocked_preflight", "skipped_feasibility", "missing_anchor"}:
        status = str(payload.get("status"))
    elif anchor_reused:
        status = "anchor_reused"
    elif breakthrough:
        status = "breakthrough"
    elif strong:
        status = "strong_pass"
    elif basic:
        status = "basic_pass"
    else:
        status = "fail"
    return {
        "config_id": config.get("config_id"),
        "method": config.get("method"),
        "beta": config.get("beta"),
        "lora_steps": config.get("lora_steps"),
        "module_scope": config.get("module_scope"),
        "crisp_projection_scope": config.get("crisp_projection_scope"),
        "projection_mode": config.get("projection_mode"),
        "cache_source": config.get("cache_source") or config.get("crisp_cache_source"),
        "reference_loss_weight": config.get("reference_loss_weight"),
        "previous_edit_loss_weight": config.get("previous_edit_loss_weight"),
        "hard_locality_loss_weight": config.get("hard_locality_loss_weight"),
        "anchor_reused": bool(anchor_reused),
        "final_new": new,
        "final_ref": ref,
        "positive_new_answer_edits": positive,
        "locality_damage": damage,
        "previous_edit_retention": final.get("previous_edit_retention"),
        "future_record_drift": final.get("future_record_drift"),
        "rollback_pass": rollback,
        "record_id_match_rate": match,
        "nan_inf_count": nan,
        "new_ratio_vs_high": _safe_div(new, C_HIGH_NEW),
        "ref_ratio_vs_high": _safe_div(ref, C_HIGH_REF),
        "new_ratio_vs_low": _safe_div(new, C_LOW_NEW),
        "ref_ratio_vs_low": _safe_div(ref, C_LOW_REF),
        "new_ratio_vs_bounded": _safe_div(new, C_BOUNDED_NEW),
        "ref_ratio_vs_bounded": _safe_div(ref, C_BOUNDED_REF),
        "basic_pass": basic,
        "strong_pass": strong,
        "breakthrough": breakthrough,
        "status": status,
    }


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class MixedDeltaPatch:
    def __init__(
        self,
        model: Any,
        *,
        dense_deltas: Optional[Dict[str, Any]] = None,
        lora_factors: Optional[Dict[str, Dict[str, Any]]] = None,
        beta: float,
    ) -> None:
        self.model = model
        self.dense_deltas = dense_deltas or {}
        self.lora_factors = lora_factors or {}
        self.beta = float(beta)
        self.original_forwards: Dict[str, Any] = {}

    def install(self) -> None:
        heavy = _heavy_imports()
        torch = heavy["torch"]
        from scripts.engram.run_token_module_ablation_5edit import _module_map

        modules = _module_map(self.model)
        targets = sorted(set(self.dense_deltas) | set(self.lora_factors))
        for name in targets:
            module = modules.get(name)
            if module is None:
                raise RuntimeError(f"Patch target missing: {name}")
            dense = self.dense_deltas.get(name)
            factor = self.lora_factors.get(name)
            dense_tensor = None
            if dense is not None:
                dense_tensor = dense.to(module.weight.device, dtype=torch.float32) * self.beta
            a = b = None
            scale = 1.0
            if factor is not None:
                a = factor["A"].to(module.weight.device, dtype=torch.float32)
                b = factor["B"].to(module.weight.device, dtype=torch.float32)
                scale = float(factor.get("scale", 1.0)) * self.beta
            self.original_forwards[name] = module.forward

            def patched_forward(x, *, _base=module.forward, _dense=dense_tensor, _a=a, _b=b, _scale=scale):
                base = _base(x)
                total = 0.0
                if _dense is not None:
                    total = total + torch.nn.functional.linear(x.to(torch.float32), _dense)
                if _a is not None and _b is not None:
                    low = torch.nn.functional.linear(x.to(torch.float32), _a)
                    total = total + torch.nn.functional.linear(low, _b) * float(_scale)
                return base + total.to(dtype=base.dtype)

            module.forward = patched_forward  # type: ignore[method-assign]

    def remove(self) -> None:
        from scripts.engram.run_token_module_ablation_5edit import _module_map

        modules = _module_map(self.model)
        for name, forward in reversed(list(self.original_forwards.items())):
            modules[name].forward = forward  # type: ignore[method-assign]
        self.original_forwards.clear()


def _sample_reference(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    from scripts.engram.run_localized_replacement_5edit import _reference_sample

    sample = _reference_sample(record, image_root)
    return dict(sample) if sample else None


def _sample_new(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    from scripts.engram.run_localized_replacement_5edit import _new_sample

    return dict(_new_sample(record, image_root))


def _current_effective_deltas(training_patch: Any, module_names: Iterable[str]) -> Dict[str, Any]:
    heavy = _heavy_imports()
    torch = heavy["torch"]
    out: Dict[str, Any] = {}
    for name in module_names:
        factor = training_patch.factors.get(name)
        if not factor:
            continue
        a = factor["A"].detach().float()
        b = factor["B"].detach().float()
        scale = float(training_patch.scale)
        out[name] = (b.matmul(a) * scale).detach().clone()
        if torch.cuda.is_available():
            out[name] = out[name].to(device=a.device)
    return out


def _project_dense_with_engram(delta: Any, projector_edit: Dict[str, Any], module_name: str) -> Tuple[Any, Dict[str, Any]]:
    raw = (projector_edit.get("updates") or {}).get(module_name)
    if raw is None or raw.get("projector") is None:
        return delta, {"module_name": module_name, "engram_projector_available": False, "reason": "missing_projector"}
    projector = raw["projector"].detach().to(device=delta.device, dtype=delta.dtype)
    reason = None
    if projector.shape[0] == delta.shape[1] and projector.shape[1] == delta.shape[1]:
        projected = delta.matmul(projector)
    elif projector.shape[0] == delta.shape[1] + 1 and projector.shape[1] == delta.shape[1] + 1:
        projected = delta.matmul(projector[: delta.shape[1], : delta.shape[1]])
        reason = "used_top_left_projector_block_for_absorbed_bias"
    else:
        projected = delta
        reason = f"projector_shape_mismatch={tuple(projector.shape)} for delta={tuple(delta.shape)}"
    return projected.detach(), {"module_name": module_name, "engram_projector_available": reason is None, "reason": reason}


def _collect_projection_caches(
    *,
    model: Any,
    module_names: List[str],
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    max_dim: int,
) -> Dict[str, Any]:
    heavy = _heavy_imports()
    torch = heavy["torch"]
    from easyeditor.models.engram.crisp_kfac_collector import collect_crisp_kfac_caches

    def loss_fn(local_model: Any, sample: Any) -> Any:
        output = local_model(dict(sample))
        return output.loss

    if not samples:
        return {
            "layer_to_projection_cache": {},
            "diagnostics": [
                {
                    "module_name": name,
                    "skipped": True,
                    "skip_reason": "empty_cache_samples",
                    "energy_threshold": 0.9,
                }
                for name in module_names
            ],
            "sample_count": 0,
            "skipped_modules": {name: "empty_cache_samples" for name in module_names},
        }
    return collect_crisp_kfac_caches(
        model,
        module_names,
        samples,
        loss_fn,
        max_dim=int(max_dim),
        energy_threshold=float(config.get("crisp_energy_threshold", 0.9)),
        build_projection_cache=True,
        projection_device="cpu",
        projection_dtype=torch.float32,
        clear_cuda_cache=True,
    )


def _build_protected_samples(
    *,
    records: List[Dict[str, Any]],
    image_root: Path,
    step_index: int,
    previous_step_rows: List[Dict[str, Any]],
    hard_topk: int,
    max_reference_records: Optional[int],
    max_previous_records: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    refs: List[Tuple[str, Dict[str, Any]]] = []
    for record in records:
        sample = _sample_reference(record, image_root)
        if sample is not None:
            refs.append((_record_id(record), sample))
    if max_reference_records is not None:
        refs = refs[: int(max_reference_records)]
    previous_records = records[:step_index]
    if max_previous_records == 0:
        previous_records = []
    elif max_previous_records is not None:
        previous_records = previous_records[-int(max_previous_records) :]
    previous = [(_record_id(record), _sample_new(record, image_root)) for record in previous_records]
    final_rows = [row for row in previous_step_rows if int(row.get("step") or -1) == step_index]
    final_rows.sort(key=lambda row: float(row.get("locality_reference_delta_abs_vs_step0") or 0.0), reverse=True)
    hard_ids = [str(row.get("record_id")) for row in final_rows[: int(hard_topk)]]
    by_id = {_record_id(record): record for record in records}
    hard = []
    for rid in hard_ids:
        record = by_id.get(rid)
        if record is None:
            continue
        sample = _sample_reference(record, image_root)
        if sample is not None:
            hard.append((rid, sample))
    combined = [sample for _, sample in refs + previous + hard]
    trace = {
        "protected_reference_record_ids": [rid for rid, _ in refs],
        "previous_edit_record_ids": [rid for rid, _ in previous],
        "hard_locality_record_ids": [rid for rid, _ in hard],
    }
    return combined, trace


def _train_lora_with_constraints(
    *,
    model: Any,
    record: Dict[str, Any],
    image_root: Path,
    module_names: List[str],
    rank: int,
    steps: int,
    lr: float,
    scale: float,
    reference_samples: List[Dict[str, Any]],
    previous_samples: List[Dict[str, Any]],
    hard_samples: List[Dict[str, Any]],
    reference_weight: float,
    previous_weight: float,
    hard_weight: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    heavy = _heavy_imports()
    torch = heavy["torch"]
    from scripts.engram.run_localized_replacement_5edit import TinyLoraTrainingPatch

    patch = TinyLoraTrainingPatch(model, module_names, rank=rank, scale=scale)
    target = _sample_new(record, image_root)
    loss_rows: List[Dict[str, Any]] = []
    patch.install()
    try:
        optimizer = torch.optim.AdamW(patch.parameters(), lr=float(lr), weight_decay=0.0)
        for step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            output = model(dict(target))
            loss = output.loss
            target_loss = output.loss

            def avg_loss(samples: List[Dict[str, Any]]) -> Optional[Any]:
                if not samples:
                    return None
                values = []
                for sample in samples:
                    values.append(model(dict(sample)).loss)
                return sum(values) / float(len(values))

            ref_loss = avg_loss(reference_samples)
            prev_loss = avg_loss(previous_samples)
            hard_loss = avg_loss(hard_samples)
            if ref_loss is not None:
                loss = loss + float(reference_weight) * ref_loss
            if prev_loss is not None:
                loss = loss + float(previous_weight) * prev_loss
            if hard_loss is not None:
                loss = loss + float(hard_weight) * hard_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"constraint LoRA loss became non-finite at step {step}: {loss}")
            loss.backward()
            optimizer.step()
            loss_rows.append(
                {
                    "step": step + 1,
                    "new_answer_loss": float(target_loss.detach().cpu()),
                    "reference_loss": float(ref_loss.detach().cpu()) if ref_loss is not None else None,
                    "previous_edit_loss": float(prev_loss.detach().cpu()) if prev_loss is not None else None,
                    "hard_locality_loss": float(hard_loss.detach().cpu()) if hard_loss is not None else None,
                    "protected_sample_count": len(reference_samples) + len(previous_samples) + len(hard_samples),
                }
            )
        state = patch.state()
    finally:
        patch.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state, {"loss_trace": loss_rows, "steps": int(steps), "rank": int(rank), "lr": float(lr), "scale": float(scale)}, loss_rows


def _train_lora_with_step_projection(
    *,
    model: Any,
    record: Dict[str, Any],
    image_root: Path,
    module_names: List[str],
    projected_module_names: List[str],
    projection_caches: Dict[str, Any],
    config: Dict[str, Any],
    rank: int,
    steps: int,
    lr: float,
    scale: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    heavy = _heavy_imports()
    torch = heavy["torch"]
    from easyeditor.models.engram.crisp_projection import apply_crisp_projection_to_delta
    from scripts.engram.run_localized_replacement_5edit import TinyLoraTrainingPatch

    patch = TinyLoraTrainingPatch(model, module_names, rank=rank, scale=scale)
    target = _sample_new(record, image_root)
    losses: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    accumulated: Dict[str, Any] = {}
    patch.install()
    try:
        optimizer = torch.optim.AdamW(patch.parameters(), lr=float(lr), weight_decay=0.0)
        before = _current_effective_deltas(patch, projected_module_names)
        for name, delta in before.items():
            accumulated[name] = torch.zeros_like(delta)
        for step in range(int(steps)):
            before = _current_effective_deltas(patch, projected_module_names)
            optimizer.zero_grad(set_to_none=True)
            output = model(dict(target))
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"projected LoRA loss became non-finite at step {step}: {loss}")
            loss.backward()
            optimizer.step()
            after = _current_effective_deltas(patch, projected_module_names)
            for name in projected_module_names:
                delta_step = after[name] - before[name]
                delta_norm = float(delta_step.detach().float().norm().cpu())
                cache = projection_caches.get(name)
                skipped = cache is None
                skip_reason = "missing_crisp_projection_cache" if skipped else None
                preclamp_norm = None
                postclamp_norm = None
                ratio_pre = None
                ratio_post = None
                clamp_applied = False
                keep_ratio = None
                if cache is None:
                    projected = delta_step
                else:
                    projected = apply_crisp_projection_to_delta(delta_step.detach().float().cpu(), cache).to(delta_step.device)
                    metadata = dict(cache.get("metadata") or {})
                    keep_ratio = metadata.get("keep_ratio")
                    preclamp_norm = float(projected.detach().float().norm().cpu())
                    ratio_pre = _safe_div(preclamp_norm, delta_norm)
                    max_ratio = float(config.get("max_projection_norm_ratio", 1.0))
                    if bool(config.get("norm_clamp", True)) and delta_norm > 0.0 and preclamp_norm is not None and preclamp_norm > delta_norm * max_ratio:
                        projected = projected * ((delta_norm * max_ratio) / max(preclamp_norm, 1.0e-12))
                        clamp_applied = True
                    postclamp_norm = float(projected.detach().float().norm().cpu())
                    ratio_post = _safe_div(postclamp_norm, delta_norm)
                accumulated[name] = accumulated[name] + projected.detach()
                trace_rows.append(
                    {
                        "projection_mode": config.get("projection_mode"),
                        "module_name": name,
                        "step": step + 1,
                        "delta_step_norm": delta_norm,
                        "projected_delta_step_norm_preclamp": preclamp_norm,
                        "projected_delta_step_norm_postclamp": postclamp_norm,
                        "projection_norm_ratio_preclamp": ratio_pre,
                        "projection_norm_ratio_postclamp": ratio_post,
                        "clamp_applied": clamp_applied,
                        "mask_keep_ratio": keep_ratio,
                        "cache_source": config.get("cache_source"),
                        "skipped": skipped,
                        "skip_reason": skip_reason,
                    }
                )
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu()), "target_loss": float(output.loss.detach().cpu())})
        state = patch.state()
    finally:
        patch.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state, {"loss_trace": losses, "steps": int(steps), "rank": int(rank), "lr": float(lr), "scale": float(scale)}, accumulated, trace_rows


def _run_one_training_config(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    config: Dict[str, Any],
    projector_bank_dir: Path,
    hparams: Any,
    run_dir: Path,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    crisp_max_dim: int,
    max_reference_cache_records: Optional[int],
    max_previous_cache_records: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    heavy = _heavy_imports()
    EngramBank = heavy["EngramBank"]
    _project_factors = heavy["_project_factors"]
    _snapshot_modules = heavy["_snapshot_modules"]
    _restore_modules = heavy["_restore_modules"]
    _max_snapshot_diff = heavy["_max_snapshot_diff"]
    torch = heavy["torch"]

    module_names = _module_names_for_scope(str(config["module_scope"]))
    qk_modules = [name for name in module_names if name in QK_ONLY_MODULES]
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    record_id_match_rate = 1.0 if matching.get("mode") == "record_id" else 0.0
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if getattr(hparams, "lora_scale", None) is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    active_patches: List[Any] = []
    patch_specs: List[Dict[str, Any]] = []
    applied_ids: List[str] = []
    train_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    constraint_rows: List[Dict[str, Any]] = []
    cache_trace_rows: List[Dict[str, Any]] = []
    reference_cache: Optional[Dict[str, Any]] = None
    if config.get("cache_source") == "reference_only":
        reference_samples = [sample for record in records if (sample := _sample_reference(record, image_root)) is not None]
        if max_reference_cache_records is not None:
            reference_samples = reference_samples[: int(max_reference_cache_records)]
        reference_cache = _collect_projection_caches(
            model=model,
            module_names=qk_modules,
            samples=reference_samples,
            config=config,
            max_dim=crisp_max_dim,
        )
        cache_trace_rows.append(
            {
                "step": 0,
                "protected_reference_record_ids": [_record_id(record) for record in records[: len(reference_samples)]],
                "previous_edit_record_ids": [],
                "hard_locality_record_ids": [],
                "cache_num_samples": reference_cache.get("sample_count"),
                "modules_with_cache": sorted((reference_cache.get("layer_to_projection_cache") or {}).keys()),
                "skipped_modules": sorted((reference_cache.get("skipped_modules") or {}).keys()),
                "skip_reasons": reference_cache.get("skipped_modules") or {},
            }
        )
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
        for step, (record, edit_id) in enumerate(zip(records, edit_ids), start=1):
            previous_step_rows = rows[:]
            current_cache = reference_cache
            protected_trace = {
                "protected_reference_record_ids": [],
                "previous_edit_record_ids": [],
                "hard_locality_record_ids": [],
            }
            protected_samples: List[Dict[str, Any]] = []
            if config.get("cache_source") == "reference_plus_previous_plus_hard_locality":
                protected_samples, protected_trace = _build_protected_samples(
                    records=records,
                    image_root=image_root,
                    step_index=step - 1,
                    previous_step_rows=previous_step_rows,
                    hard_topk=int(config.get("hard_locality_topk", 5)),
                    max_reference_records=max_reference_cache_records,
                    max_previous_records=max_previous_cache_records,
                )
                current_cache = _collect_projection_caches(
                    model=model,
                    module_names=qk_modules,
                    samples=protected_samples,
                    config=config,
                    max_dim=crisp_max_dim,
                )
                cache_trace_rows.append(
                    {
                        "step": step,
                        **protected_trace,
                        "cache_num_samples": current_cache.get("sample_count"),
                        "modules_with_cache": sorted((current_cache.get("layer_to_projection_cache") or {}).keys()),
                        "skipped_modules": sorted((current_cache.get("skipped_modules") or {}).keys()),
                        "skip_reasons": current_cache.get("skipped_modules") or {},
                    }
                )
            projector_edit = bank.load_edit(str(edit_id))
            dense_deltas: Dict[str, Any] = {}
            lora_factors: Dict[str, Dict[str, Any]] = {}
            if config["method"] in {METHOD_T1, METHOD_T2}:
                factors, train_summary, accumulated, trace = _train_lora_with_step_projection(
                    model=model,
                    record=record,
                    image_root=image_root,
                    module_names=module_names,
                    projected_module_names=qk_modules,
                    projection_caches=(current_cache or {}).get("layer_to_projection_cache") or {},
                    config=config,
                    rank=int(hparams.lora_rank),
                    steps=int(config["lora_steps"]),
                    lr=float(hparams.lora_lr),
                    scale=scale,
                )
                safe_factors, engram_summary = _project_factors(factors, projector_edit)
                for name, delta in accumulated.items():
                    projected_delta, engram_row = _project_dense_with_engram(delta, projector_edit, name)
                    dense_deltas[name] = projected_delta.detach().cpu()
                    projection_rows.append({"record_id": _record_id(record), "edit_step": step, **engram_row})
                for name, factor in safe_factors.items():
                    if name not in dense_deltas:
                        lora_factors[name] = factor
                projection_rows.extend(dict(row, record_id=_record_id(record), edit_step=step) for row in trace)
            elif config["method"] == METHOD_T3:
                reference_samples = [sample for rec in records if (sample := _sample_reference(rec, image_root)) is not None]
                if max_reference_cache_records is not None:
                    reference_samples = reference_samples[: int(max_reference_cache_records)]
                previous_records = records[: step - 1]
                if max_previous_cache_records is not None:
                    previous_records = previous_records[-int(max_previous_cache_records) :]
                previous_samples = [_sample_new(rec, image_root) for rec in previous_records]
                hard_samples, hard_trace = _build_protected_samples(
                    records=records,
                    image_root=image_root,
                    step_index=step - 1,
                    previous_step_rows=previous_step_rows,
                    hard_topk=int(config.get("hard_locality_topk", 5)),
                    max_reference_records=0,
                    max_previous_records=0,
                )
                factors, train_summary, losses = _train_lora_with_constraints(
                    model=model,
                    record=record,
                    image_root=image_root,
                    module_names=module_names,
                    rank=int(hparams.lora_rank),
                    steps=int(config["lora_steps"]),
                    lr=float(hparams.lora_lr),
                    scale=scale,
                    reference_samples=reference_samples,
                    previous_samples=previous_samples,
                    hard_samples=hard_samples,
                    reference_weight=float(config.get("reference_loss_weight", 0.0)),
                    previous_weight=float(config.get("previous_edit_loss_weight", 0.0)),
                    hard_weight=float(config.get("hard_locality_loss_weight", 0.0)),
                )
                safe_factors, engram_summary = _project_factors(factors, projector_edit)
                lora_factors = safe_factors
                constraint_rows.extend(
                    dict(
                        row,
                        config_id=config["config_id"],
                        edit_step=step,
                        record_id=_record_id(record),
                        protected_reference_record_ids=[_record_id(rec) for rec in records[: len(reference_samples)]],
                        previous_edit_record_ids=[_record_id(rec) for rec in previous_records],
                        hard_locality_record_ids=hard_trace["hard_locality_record_ids"],
                    )
                    for row in losses
                )
            else:
                raise RuntimeError(f"Unsupported training config method: {config['method']}")
            patch = MixedDeltaPatch(model, dense_deltas=dense_deltas, lora_factors=lora_factors, beta=float(config["beta"]))
            patch.install()
            active_patches.append(patch)
            rid = _record_id(record)
            patch_specs.append({"record_id": rid, "dense_deltas": dense_deltas, "lora_factors": lora_factors, "beta": float(config["beta"])})
            applied_ids.append(rid)
            train_rows.append(
                {
                    "config_id": config["config_id"],
                    "step": step,
                    "record_id": rid,
                    "method": config["method"],
                    "lora_train": train_summary,
                    "engram_projection": engram_summary,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        "patch_specs": patch_specs,
    }
    _write_payload_files(run_dir, payload, config)
    _json_dump(run_dir / "projection_trace.json", {"status": "complete", "rows": projection_rows})
    _json_dump(run_dir / "constraint_loss_trace.json", {"status": "complete", "rows": constraint_rows})
    _json_dump(run_dir / "cache_trace.json", {"status": "complete", "rows": cache_trace_rows})
    _json_dump(run_dir / "train_trace.json", train_rows)
    return payload, patch_specs


def _choose_best(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in scored if not row.get("anchor_reused") and row.get("final_new") is not None]
    if not candidates:
        return None
    breakthrough = [row for row in candidates if row.get("breakthrough")]
    if breakthrough:
        return sorted(breakthrough, key=lambda row: (-float(row.get("final_new") or 0.0), float(row.get("final_ref") or 999.0)))[0]
    strong = [row for row in candidates if row.get("strong_pass")]
    if strong:
        return sorted(strong, key=lambda row: (-float(row.get("final_new") or 0.0), int(row.get("locality_damage") or 999)))[0]
    basic = [row for row in candidates if row.get("basic_pass")]
    if basic:
        return sorted(basic, key=lambda row: (-float(row.get("final_new") or 0.0), float(row.get("final_ref") or 999.0)))[0]
    return sorted(candidates, key=lambda row: (int(row.get("locality_damage") or 999), -float(row.get("final_new") or 0.0)))[0]


def _write_analysis(out_dir: Path, scored: List[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> None:
    t_rows = [row for row in scored if not row.get("anchor_reused")]
    strong = [row for row in t_rows if row.get("strong_pass")]
    breakthrough = [row for row in t_rows if row.get("breakthrough")]
    lines = [
        "# Crisp Training Analysis",
        "",
        "## Questions",
        "",
        "1. Training-time projection improvement over post-hoc CURE: evaluated from completed T1/T2 rows only.",
        "2. Previous-edit-aware cache drift reduction: evaluated from completed T2 rows only.",
        "3. Constraint-only training: evaluated from completed T3 rows only.",
        "4. Improvement over `C_beta0.35_steps10_qkgate_ref0`: use `new_ratio_vs_bounded` and `ref_ratio_vs_bounded` in the summary.",
        "5. Break target `final_new >= 0.60` with `locality_damage <= 8/20`: see strong/breakthrough flags.",
        "6. Norm shrinkage is tracked through `projection_trace.json` pre/post clamp ratios.",
        "7. Identity-like projection is indicated by ratios near 1 and high mask keep ratios.",
        "8. gate_proj modules use ENGRAM fallback unless a future diag/low-rank implementation is added.",
        "9. `projection_norm_ratio_postclamp` is required to stay <= 1 for projected modules.",
        "10. Hard-locality protection is tracked by repeated hard-locality record IDs in `cache_trace.json`.",
        "",
        "## Outcome",
        "",
    ]
    if not t_rows:
        lines.append("- No new Crisp/CURE variant completed in this environment.")
    elif breakthrough:
        lines.append("- At least one variant met the breakthrough criterion. Validate the best variant on 50 MedMKEB model-known edits or external Med-VQA.")
    elif strong:
        lines.append("- At least one variant met the strong criterion but not breakthrough.")
    else:
        lines.append("- No variant met the strong or breakthrough criterion. If all attempted variants failed, stop Crisp/CURE for now and move to an ENGRAM-routed edit bank.")
    if best:
        lines.append(f"- Best completed variant: `{best.get('config_id')}` with status `{best.get('status')}`.")
    (out_dir / "CRISP_TRAINING_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pilot(out_dir: Path, scored: List[Dict[str, Any]], best: Optional[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    made: List[str] = []
    plot_dir = out_dir / "plots"
    try:
        rows = [row for row in scored if row.get("final_new") is not None and row.get("final_ref") is not None]
        if rows:
            plt.figure(figsize=(7, 5))
            plt.scatter([float(row["final_ref"]) for row in rows], [float(row["final_new"]) for row in rows])
            for row in rows:
                plt.annotate(str(row["config_id"]).replace("_", "\n"), (float(row["final_ref"]), float(row["final_new"])), fontsize=5)
            plt.xlabel("final reference delta abs")
            plt.ylabel("final new-answer NLL decrease")
            plt.tight_layout()
            path = plot_dir / "ref_vs_new_scatter.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))

            plt.figure(figsize=(8, 4))
            plt.bar([str(row["config_id"]) for row in rows], [int(row.get("locality_damage") or 0) for row in rows])
            plt.xticks(rotation=45, ha="right", fontsize=6)
            plt.ylabel("locality damage records")
            plt.tight_layout()
            path = plot_dir / "locality_damage_by_config.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))

            plt.figure(figsize=(8, 4))
            plt.bar([str(row["config_id"]) for row in rows], [float(row.get("ref_ratio_vs_high") or 0.0) for row in rows])
            plt.xticks(rotation=45, ha="right", fontsize=6)
            plt.ylabel("ref ratio vs high-strength C")
            plt.tight_layout()
            path = plot_dir / "projection_norm_by_config.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if best and best.get("config_id") in payloads:
            summary_rows = payloads[str(best["config_id"])].get("summary_rows", [])
            xs = [int(row.get("step") or 0) for row in summary_rows]
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


def _write_final_report(
    *,
    out_dir: Path,
    configs: List[Dict[str, Any]],
    scored: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
    data_reuse: Dict[str, Any],
    feasibility: Dict[str, Any],
    runtime_status: str,
) -> None:
    new_rows = [row for row in scored if not row.get("anchor_reused")]
    breakthrough = [row for row in new_rows if row.get("breakthrough")]
    strong = [row for row in new_rows if row.get("strong_pass")]
    improved = [
        row
        for row in new_rows
        if row.get("final_new") is not None
        and row.get("final_ref") is not None
        and float(row["final_new"]) > C_BOUNDED_NEW
        and float(row["final_ref"]) <= 0.08
    ]
    if runtime_status != "complete":
        decision = "Blocked. The pilot artifacts were prepared, but no new Crisp/CURE metrics should be claimed."
    elif breakthrough:
        decision = "A. Crisp/CURE training achieves a breakthrough. Next: validate the best variant on 50 MedMKEB model-known edits or external Med-VQA."
    elif strong or improved:
        decision = "B. Crisp/CURE training improves but does not break the trade-off. Keep it as partial; consider routed edit bank."
    else:
        decision = "C. Crisp/CURE training does not improve over ENGRAM-projected LoRA rescue. Stop Crisp/CURE for now; move to ENGRAM-routed edit bank."
    lines = [
        "# Final MedMKEB Crisp Training Pilot 20 Report",
        "",
        "## Starting Point",
        "",
        "- MedMKEB nonseq `C_engram_projected_tiny_lora` works.",
        "- MedMKEB sequential remains strength-locality coupled.",
        "- Weak edit settings reduce locality damage but lose edit strength.",
        "- This pilot tests whether training-time Crisp/CURE mechanisms can break that trade-off.",
        "",
        "## Data",
        "",
        f"- Exact reused selected records: `{data_reuse.get('selected_record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- No clinical or medical efficacy claim is made.",
        "",
        "## Feasibility Audit",
        "",
        f"- Selected safe projection mode: `{feasibility.get('selected_safe_projection_mode')}`",
        f"- Exact LoRA factor-gradient projection feasible: `{feasibility.get('exact_lora_factor_gradient_projection_feasible')}`",
        f"- q/k cache estimate: `{_format(feasibility.get('qk_projection_cache_total_gib_estimate'))}` GiB",
        f"- gate cache estimate: `{_format(feasibility.get('gate_projection_cache_total_gib_estimate'))}` GiB",
        f"- gate policy: `{feasibility.get('gate_policy')}`",
        "",
        "## Methods And Configs",
        "",
        f"- Total configs: `{len(configs)}`",
        "- Anchors: high-strength C, low-drift C, bounded C.",
        "- T1: q/k training-time effective-delta Crisp projection with gate ENGRAM fallback.",
        "- T2: T1 plus previous-edit-aware and hard-locality-aware cache.",
        "- T3: constraint-only training with no post-hoc CURE delta projection.",
        "",
        "## Aggregate Table",
        "",
        "| config_id | status | final_new | final_ref | positive_new | damage | retention | rollback | match | nan | projection_mode | cache_source | constraints |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in scored:
        constraints = "ref={ref},prev={prev},hard={hard}".format(
            ref=row.get("reference_loss_weight"),
            prev=row.get("previous_edit_loss_weight"),
            hard=row.get("hard_locality_loss_weight"),
        )
        lines.append(
            "| {config_id} | {status} | {new} | {ref} | {positive} | {damage} | {retention} | {rollback} | {match} | {nan} | {projection_mode} | {cache_source} | {constraints} |".format(
                config_id=row.get("config_id"),
                status=row.get("status"),
                new=_format(row.get("final_new")),
                ref=_format(row.get("final_ref")),
                positive=row.get("positive_new_answer_edits"),
                damage=row.get("locality_damage"),
                retention=_format(row.get("previous_edit_retention")),
                rollback=row.get("rollback_pass"),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
                projection_mode=row.get("projection_mode"),
                cache_source=row.get("cache_source"),
                constraints=constraints,
            )
        )
    lines.extend(
        [
            "",
            "## Projection And Cache Diagnostics",
            "",
            "- Per-step module projection rows are saved in each run directory as `projection_trace.json`.",
            "- Constraint loss rows are saved as `constraint_loss_trace.json`.",
            "- T2 cache membership is saved in `cache_trace.json`.",
            "- gate_proj is not silently skipped: it is handled by ENGRAM fallback unless a future safe diag/low-rank CURE gate path is added.",
            "",
            "## Best Variant",
            "",
        ]
    )
    if best:
        lines.extend(
            [
                f"- Best variant: `{best.get('config_id')}`",
                f"- Status: `{best.get('status')}`",
                f"- final_new: `{_format(best.get('final_new'))}`",
                f"- final_ref: `{_format(best.get('final_ref'))}`",
                f"- locality_damage: `{best.get('locality_damage')}/20`",
                f"- strong pass: `{best.get('strong_pass')}`",
                f"- breakthrough: `{best.get('breakthrough')}`",
            ]
        )
    else:
        lines.append("- No new Crisp/CURE variant completed.")
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
            "- No clinical or medical efficacy claim.",
            "- Projected LoRA optimizer is approximate when exact full-gradient projection is infeasible.",
            "- Do not interpret prepared or preflight-blocked rows as completed metrics.",
        ]
    )
    (out_dir / "FINAL_MEDMKEB_CRISP_TRAINING_PILOT_20_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_package_hygiene_report(out_dir: Path) -> Dict[str, Any]:
    hygiene = _package_hygiene(out_dir, remove_runtime_bank=True)
    forbidden_patterns = [".pt", ".pth", ".bin", "__pycache__", ".pyc", ".DS_Store", "._"]
    found: List[str] = []
    for path in out_dir.rglob("*"):
        name = path.name
        if path.is_dir() and name == "__pycache__":
            found.append(str(path))
        elif name == ".DS_Store" or name.startswith("._") or path.suffix in {".pt", ".pth", ".bin", ".pyc"}:
            found.append(str(path))
    payload = {**hygiene, "forbidden_artifacts_found": found, "forbidden_artifact_count": len(found)}
    lines = [
        "# Package Hygiene Report",
        "",
        f"- Forbidden artifacts found: `{len(found)}`",
        "- Checked exclusions: `.pt`, `.pth`, `.bin`, projector bank tensors, model weights, Hugging Face cache, CUDA cache, `__pycache__`, `.pyc`, `.DS_Store`, and `._*` AppleDouble files.",
        "",
    ]
    if found:
        lines.extend(["## Found", ""])
        lines.extend(f"- `{item}`" for item in found[:100])
    else:
        lines.append("No forbidden artifacts were found under this pilot output directory.")
    (out_dir / "PACKAGE_HYGIENE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "PACKAGE_HYGIENE_REPORT.md.json", payload)
    return payload


def _write_prepare_artifacts(args: argparse.Namespace, *, runtime_status: str, runtime_reason: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    configs = _pilot_config_grid()
    _json_dump(out_dir / "crisp_training_config_grid.json", {"configs": configs})
    records = _load_records(Path(args.selected_records))
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    feasibility = _write_feasibility_audit(out_dir, configs)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    _, anchor_scores = _load_anchor_payloads(out_dir, configs)
    scored = list(anchor_scores)
    for config in configs:
        if config.get("anchor"):
            continue
        run_dir = out_dir / "runs" / str(config["config_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": runtime_status,
            "reason": runtime_reason or "prepare-only; run-gpu was not completed",
            "config": config,
            "summary_rows": [],
            "per_record_step_rows": [],
            "final_summary": {},
            "final_rollback_check": {},
        }
        _write_payload_files(run_dir, payload, config)
        scored.append(_score_payload(payload))
    _write_csv(out_dir / "crisp_training_summary.csv", scored)
    _json_dump(out_dir / "crisp_training_summary.json", {"scores": scored, "runtime_status": runtime_status, "runtime_reason": runtime_reason})
    _json_dump(out_dir / "cache_trace.json", {"status": runtime_status, "rows": []})
    best = _choose_best(scored)
    _write_analysis(out_dir, scored, best)
    plots = _plot_pilot(out_dir, scored, best, {})
    generation = {"status": "skipped", "reason": "no new variant completed basic/strong gate"}
    _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", generation)
    _write_final_report(
        out_dir=out_dir,
        configs=configs,
        scored=scored,
        best=best if best and not best.get("anchor_reused") else None,
        generation=generation,
        plots=plots,
        data_reuse=data_reuse,
        feasibility=feasibility,
        runtime_status=runtime_status,
    )
    _write_package_hygiene_report(out_dir)
    _json_dump(out_dir / "runtime.json", {"status": runtime_status, "reason": runtime_reason})
    return scored, feasibility


def _prepare_only(args: argparse.Namespace) -> int:
    _write_prepare_artifacts(args, runtime_status="prepare_only", runtime_reason="prepare mode writes audits and anchor reuse only")
    return 0


def _preflight_only(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    configs = _pilot_config_grid()
    _json_dump(out_dir / "crisp_training_config_grid.json", {"configs": configs})
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = {"status": "skipped", "reason": "preflight mode; tests are run as a separate stage"}
    _json_dump(out_dir / "test_logs" / "test_status.json", test_status)
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        input_records=Path(args.selected_records),
        image_root=Path(args.image_root),
        test_status=test_status,
    )
    records = _load_records(Path(args.selected_records))
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    feasibility = _write_feasibility_audit(out_dir, configs)
    _load_anchor_payloads(out_dir, configs)
    _write_final_report(
        out_dir=out_dir,
        configs=configs,
        scored=[],
        best=None,
        generation={"status": "skipped", "reason": "preflight only"},
        plots={"status": "skipped", "reason": "preflight only"},
        data_reuse=data_reuse,
        feasibility=feasibility,
        runtime_status="preflight_pass" if preflight.get("status") == "pass" else "blocked_preflight",
    )
    _json_dump(out_dir / "runtime.json", {"status": "preflight", "preflight": preflight})
    return 0 if preflight.get("status") == "pass" else 2


def _run_gpu(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    configs = _pilot_config_grid()
    _json_dump(out_dir / "crisp_training_config_grid.json", {"configs": configs})
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests_pilot(out_dir, run_tests=not args.skip_tests)
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        input_records=Path(args.selected_records),
        image_root=Path(args.image_root),
        test_status=test_status,
    )
    if preflight.get("status") != "pass":
        _write_prepare_artifacts(
            args,
            runtime_status="blocked_preflight",
            runtime_reason=f"Preflight failed: {preflight}",
        )
        return 2
    records = _load_records(Path(args.selected_records))
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    feasibility = _write_feasibility_audit(out_dir, configs)
    anchor_payloads, anchor_scores = _load_anchor_payloads(out_dir, configs)
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
    bootstrap_bank = out_dir / "runtime_projector_banks" / "qk_gate_sampled_depths"
    _configure_hparams_for_scope(
        hparams=hparams,
        image_root=Path(args.image_root),
        bank_dir=bootstrap_bank,
        device=str(args.device),
        module_names=QK_GATE_MODULES,
        lora_steps=20,
        lora_ref_loss_weight=0.0,
    )
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    if set(selected) != set(QK_GATE_MODULES):
        raise RuntimeError({"reason": "selected module mismatch", "selected": selected, "expected": QK_GATE_MODULES})
    _json_dump(out_dir / "audit" / "selected_modules_qk_gate_sampled_depths.json", {"status": "pass", "selected_modules": selected})
    _extract_projector_bank(editor, hparams, Path(args.selected_records), records, bootstrap_bank)

    payloads: Dict[str, Dict[str, Any]] = dict(anchor_payloads)
    scored: List[Dict[str, Any]] = list(anchor_scores)
    all_cache_trace_rows: List[Dict[str, Any]] = []
    for config in configs:
        if config.get("anchor"):
            continue
        run_dir = out_dir / "runs" / str(config["config_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        run_hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
        _configure_hparams_for_scope(
            hparams=run_hparams,
            image_root=Path(args.image_root),
            bank_dir=bootstrap_bank,
            device=str(args.device),
            module_names=_module_names_for_scope(str(config["module_scope"])),
            lora_steps=int(config["lora_steps"]),
            lora_ref_loss_weight=float(config.get("lora_ref_loss_weight", 0.0)),
        )
        payload, _specs = _run_one_training_config(
            model=editor.model,
            records=records,
            image_root=Path(args.image_root),
            baselines=baselines,
            config=config,
            projector_bank_dir=bootstrap_bank,
            hparams=run_hparams,
            run_dir=run_dir,
            rollback_tolerance=float(args.rollback_tolerance),
            locality_threshold=float(args.locality_damage_threshold),
            max_new_tokens=int(args.max_new_tokens),
            crisp_max_dim=int(args.crisp_max_dim),
            max_reference_cache_records=args.max_reference_cache_records,
            max_previous_cache_records=args.max_previous_cache_records,
        )
        payloads[str(config["config_id"])] = payload
        score = _score_payload(payload)
        scored.append(score)
        cache_trace_path = run_dir / "cache_trace.json"
        if cache_trace_path.exists():
            cache_payload = _read_json(cache_trace_path)
            all_cache_trace_rows.extend(cache_payload.get("rows", []))
        _json_dump(run_dir / "score.json", score)
    _write_csv(out_dir / "crisp_training_summary.csv", scored)
    _json_dump(out_dir / "crisp_training_summary.json", {"scores": scored})
    _json_dump(out_dir / "cache_trace.json", {"status": "complete", "rows": all_cache_trace_rows})
    best = _choose_best(scored)
    _write_analysis(out_dir, scored, best)
    plots = _plot_pilot(out_dir, scored, best, payloads)
    generation = {"status": "skipped", "reason": "generation diagnostics require applying mixed dense patches; run NLL/logprob gate first"}
    _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", generation)
    _write_final_report(
        out_dir=out_dir,
        configs=configs,
        scored=scored,
        best=best if best and not best.get("anchor_reused") else None,
        generation=generation,
        plots=plots,
        data_reuse=data_reuse,
        feasibility=feasibility,
        runtime_status="complete",
    )
    _cleanup_runtime_projector_banks(out_dir)
    _write_package_hygiene_report(out_dir)
    _json_dump(out_dir / "runtime.json", {"status": "complete", "best": best})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded MedMKEB 20-edit sequential Crisp/CURE training pilot.")
    parser.add_argument("--mode", choices=["prepare", "preflight", "run-gpu"], default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / PILOT_DIRNAME))
    parser.add_argument("--selected-records", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "medmkeb_modelknown_20.json"))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "baseline_metrics.json"))
    parser.add_argument("--previous-record-preflight", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "record_id_preflight.json"))
    parser.add_argument("--image-root", default="/Volumes/DataP/knowledge_editing/data/medmkeb/images")
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", help="Accepted for compatibility; the main NLL/logprob gate always skips generation.")
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--crisp-max-dim", type=int, default=4097)
    parser.add_argument("--max-reference-cache-records", type=int, default=None)
    parser.add_argument("--max-previous-cache-records", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "prepare":
        return _prepare_only(args)
    if args.mode == "preflight":
        return _preflight_only(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
