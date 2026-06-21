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

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_cure_mededit_5edit import (  # noqa: E402
    _aggregate_sequential_step,
    _extract_direct_failure_metrics,
    _format,
    _json_dump,
    _load_previous_nonseq,
    _run_one_sequential_method,
)
from scripts.engram.run_localized_replacement_5edit import (  # noqa: E402
    EXPECTED_MODULES,
    _configure_hparams,
    _evaluate_current,
    _extract_projector_bank,
    _load_records,
    _prepare_replacement_data,
    _write_csv,
    _write_failure_summary,
    _write_git_outputs,
)


PRIORITY_CURE_CONFIGS = [
    {"config_id": "E_beta0.5_gamma0.5_static", "beta": 0.5, "crisp_energy_threshold": 0.5, "crisp_cache_update_policy": "static"},
    {"config_id": "E_beta0.5_gamma0.7_static", "beta": 0.5, "crisp_energy_threshold": 0.7, "crisp_cache_update_policy": "static"},
    {"config_id": "E_beta0.5_gamma0.9_static", "beta": 0.5, "crisp_energy_threshold": 0.9, "crisp_cache_update_policy": "static"},
    {"config_id": "E_beta0.75_gamma0.7_static", "beta": 0.75, "crisp_energy_threshold": 0.7, "crisp_cache_update_policy": "static"},
    {"config_id": "E_beta0.25_gamma0.7_static", "beta": 0.25, "crisp_energy_threshold": 0.7, "crisp_cache_update_policy": "static"},
    {"config_id": "E_beta0.5_gamma0.5_streaming", "beta": 0.5, "crisp_energy_threshold": 0.5, "crisp_cache_update_policy": "streaming_average"},
    {"config_id": "E_beta0.5_gamma0.7_streaming", "beta": 0.5, "crisp_energy_threshold": 0.7, "crisp_cache_update_policy": "streaming_average"},
    {"config_id": "E_beta0.5_gamma0.9_streaming", "beta": 0.5, "crisp_energy_threshold": 0.9, "crisp_cache_update_policy": "streaming_average"},
]


def _run_capture(command: List[str], cwd: Path) -> str:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _write_env_report(out_dir: Path) -> None:
    lines = [
        f"cwd={Path.cwd()}",
        "python=" + _run_capture([sys.executable, "-c", "import sys; print(sys.executable); print(sys.version)"], PROJECT_ROOT).strip(),
        "nvidia-smi="
        + _run_capture(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader"], PROJECT_ROOT).strip(),
        _run_capture(
            [
                sys.executable,
                "-c",
                (
                    "import torch, transformers, peft, PIL; "
                    "print('torch', torch.__version__); "
                    "print('cuda', torch.cuda.is_available()); "
                    "print('cuda_devices', torch.cuda.device_count()); "
                    "print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
                    "print('transformers', transformers.__version__); "
                    "print('peft', peft.__version__); "
                    "print('PIL', PIL.__version__)"
                ),
            ],
            PROJECT_ROOT,
        ).strip(),
    ]
    (out_dir / "env_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    source_data: Path,
    source_image_root: Path,
    previous_nonseq_dir: Path,
) -> Dict[str, Any]:
    model_path = _yaml_value(hparams_path, "name") or _yaml_value(hparams_path, "model_path") or _yaml_value(hparams_path, "model_name")
    vision_path = (
        _yaml_value(hparams_path, "llava_med_vision_tower")
        or _yaml_value(hparams_path, "clip_vision_path")
        or _yaml_value(hparams_path, "vision_tower")
    )
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "path_exists_hparams": hparams_path.exists(),
        "path_exists_model": bool(model_path and Path(model_path).exists()),
        "path_exists_vision_tower": bool(vision_path and Path(vision_path).exists()),
        "path_exists_replacement_5edit_data": source_data.exists(),
        "path_exists_image_root": source_image_root.exists(),
        "path_exists_previous_sequential_report": Path("outputs/cure_mededit_5edit/sequential_real/FINAL_CURE_SEQUENTIAL_5EDIT_REPORT.md").exists(),
        "path_exists_previous_nonseq_results": (previous_nonseq_dir / "cure_nonseq_results.json").exists(),
        "output_dir_writable": out_dir.exists() and out_dir.is_dir(),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "paths": {
            "hparams": str(hparams_path),
            "model_path": model_path,
            "vision_tower": vision_path,
            "replacement_5edit_data": str(source_data),
            "image_root": str(source_image_root),
            "previous_nonseq_dir": str(previous_nonseq_dir),
            "output_dir": str(out_dir),
        },
    }
    lines = [
        "# CURE Sequential Pareto Preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Python: `{sys.executable}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Paths", ""])
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    if payload["status"] != "pass":
        lines.extend(["", "## Blockers", ""])
        for key, value in checks.items():
            if not value:
                lines.append(f"- {key}")
    (out_dir / "PREFLIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "preflight_status.json", payload)
    return payload


def _annotate_rows(
    rows: List[Dict[str, Any]],
    *,
    config_id: str,
    beta: float,
    threshold: float,
    policy: str,
) -> List[Dict[str, Any]]:
    for row in rows:
        row["config_id"] = config_id
        row["beta"] = float(beta)
        row["crisp_energy_threshold"] = float(threshold)
        row["crisp_cache_update_policy"] = policy
        row["old_answer_nll_increase_vs_step0"] = row.get("old_answer_nll_delta_vs_step0")
    return rows


def _summarize_config(
    rows: List[Dict[str, Any]],
    *,
    config_id: str,
    method: str,
    beta: float,
    threshold: float,
    policy: str,
) -> List[Dict[str, Any]]:
    summary_rows = []
    for step in range(0, 6):
        row = _aggregate_sequential_step(rows, method, step)
        row.update(
            {
                "config_id": config_id,
                "beta": float(beta),
                "crisp_energy_threshold": float(threshold),
                "crisp_cache_update_policy": policy,
            }
        )
        summary_rows.append(row)
    return summary_rows


def _final_row(summary_rows: List[Dict[str, Any]], config_id: str) -> Dict[str, Any]:
    matches = [row for row in summary_rows if row.get("config_id") == config_id and int(row.get("step") or 0) == 5]
    if not matches:
        return {}
    row = dict(matches[0])
    row["mean_new_answer_nll_decrease"] = row.get("mean_new_answer_nll_decrease_edited_records")
    row["mean_reference_delta_abs"] = row.get("mean_reference_delta_abs_all_records")
    row["previous_edit_retention"] = row.get("mean_previous_edit_retention")
    row["rollback_pass"] = float(row.get("rollback_pass_rate") or 0.0) == 1.0
    return row


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _diag_from_trace(config_id: str, trace_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ratios: List[float] = []
    projection_ratios: List[float] = []
    candidate_norms: List[float] = []
    engram_norms: List[float] = []
    cure_norms: List[float] = []
    modules_with_cache = set()
    skipped_modules = set()
    skip_reasons: Dict[str, Any] = {}
    accumulated = []
    for row in trace_rows:
        modules_with_cache.update(row.get("modules_with_cache") or [])
        accumulated.append(int(row.get("accumulated_num_samples") or 0))
        for value in (row.get("mask_keep_ratios") or {}).values():
            if value is not None:
                ratios.append(float(value))
        for module in row.get("skipped_modules") or []:
            if module:
                skipped_modules.add(module)
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
    policy = next((row.get("crisp_cache_update_policy") for row in trace_rows if row.get("crisp_cache_update_policy")), None)
    return {
        "config_id": config_id,
        "average_mask_keep_ratio": _mean(ratios),
        "modules_with_cache": sorted(modules_with_cache),
        "skipped_modules": sorted(skipped_modules),
        "skip_reasons": skip_reasons,
        "cache_update_policy": policy,
        "accumulated_num_samples": max(accumulated) if accumulated else None,
        "projection_norm_ratio": _mean(projection_ratios),
        "delta_candidate_norm": _mean(candidate_norms),
        "delta_engram_norm": _mean(engram_norms),
        "delta_cure_norm": _mean(cure_norms),
    }


def _score_final_rows(final_rows: List[Dict[str, Any]], c_final: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    c_new = float(c_final.get("mean_new_answer_nll_decrease") or 0.0)
    c_ref = float(c_final.get("mean_reference_delta_abs") or 0.0)
    c_ret = float(c_final.get("previous_edit_retention") or 0.0)
    scored: List[Dict[str, Any]] = []
    promising: List[Dict[str, Any]] = []
    for row in final_rows:
        row = dict(row)
        if row.get("method") == "C_engram_projected_tiny_lora":
            row.update({"new_answer_ratio": 1.0, "reference_ratio": 1.0, "retention_ratio": 1.0})
        else:
            new = float(row.get("mean_new_answer_nll_decrease") or 0.0)
            ref = float(row.get("mean_reference_delta_abs") or 0.0)
            ret = float(row.get("previous_edit_retention") or 0.0)
            row["new_answer_ratio"] = new / c_new if c_new else None
            row["reference_ratio"] = ref / c_ref if c_ref else None
            row["retention_ratio"] = ret / c_ret if c_ret else None
        row["locality_score"] = (
            None
            if row.get("mean_new_answer_nll_decrease") is None or row.get("mean_reference_delta_abs") is None
            else float(row["mean_new_answer_nll_decrease"]) - float(row["mean_reference_delta_abs"])
        )
        row["retention_score"] = (
            None
            if row.get("previous_edit_retention") is None or row.get("mean_reference_delta_abs") is None
            else float(row["previous_edit_retention"]) - float(row["mean_reference_delta_abs"])
        )
        is_promising = (
            row.get("method") == "E_cure_dual_projected_tiny_lora"
            and int(row.get("positive_new_answer_edits") or 0) == 5
            and int(row.get("locality_damage_records") or 0) == 0
            and bool(row.get("rollback_pass"))
            and float(row.get("record_id_match_rate") or 0.0) == 1.0
            and int(row.get("nan_inf_count") or 0) == 0
            and row.get("new_answer_ratio") is not None
            and float(row["new_answer_ratio"]) >= 0.97
            and row.get("retention_ratio") is not None
            and float(row["retention_ratio"]) >= 0.97
            and row.get("reference_ratio") is not None
            and float(row["reference_ratio"]) <= 0.75
        )
        row["pareto_promising"] = bool(is_promising)
        if is_promising:
            promising.append(row)
        scored.append(row)
    cure_rows = [row for row in scored if row.get("method") == "E_cure_dual_projected_tiny_lora"]
    best = max(
        cure_rows,
        key=lambda row: (
            bool(row.get("pareto_promising")),
            float(row.get("new_answer_ratio") or -math.inf),
            float(row.get("retention_ratio") or -math.inf),
            -float(row.get("reference_ratio") or math.inf),
            float(row.get("locality_score") or -math.inf),
        ),
    ) if cure_rows else {}
    return scored, best, promising


def _write_plots(out_dir: Path, final_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cure = [row for row in final_rows if row.get("method") == "E_cure_dual_projected_tiny_lora"]
        plots = []

        def scatter(path: Path, x_key: str, y_key: str, xlabel: str, ylabel: str) -> None:
            plt.figure(figsize=(6, 4))
            for row in cure:
                plt.scatter(float(row[x_key]), float(row[y_key]))
                plt.annotate(str(row["config_id"]).replace("E_", ""), (float(row[x_key]), float(row[y_key])), fontsize=6)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            plots.append(str(path))

        scatter(plot_dir / "new_answer_vs_reference_delta.png", "mean_reference_delta_abs", "mean_new_answer_nll_decrease", "mean reference delta abs", "mean new-answer NLL decrease")
        scatter(plot_dir / "retention_vs_reference_delta.png", "mean_reference_delta_abs", "previous_edit_retention", "mean reference delta abs", "previous-edit retention")
        scatter(plot_dir / "pareto_frontier.png", "reference_ratio", "new_answer_ratio", "reference ratio vs C", "new-answer ratio vs C")
        return {"status": "pass", "plots": plots}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _write_final_report(
    out_dir: Path,
    *,
    final_rows: List[Dict[str, Any]],
    scored_rows: List[Dict[str, Any]],
    best_tradeoff: Dict[str, Any],
    promising: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
    plot_status: Dict[str, Any],
) -> None:
    c = next((row for row in scored_rows if row.get("method") == "C_engram_projected_tiny_lora"), {})
    decision = "C. CURE does not improve over ENGRAM-projected LoRA. Do not pursue CURE further until curvature estimation/projection order is revised."
    if promising:
        decision = "A. CURE has a Pareto-promising config. Next gate: 10-edit model-known non-PHI set with C baseline and best CURE config."
    elif best_tradeoff and float(best_tradeoff.get("reference_ratio") or 1.0) < 1.0:
        decision = "B. CURE reduces reference damage but consistently loses too much retention/new-answer strength. Keep ENGRAM-projected LoRA as primary method; use CURE as optional conservative variant."

    lines = [
        "# Final CURE Sequential Pareto Report",
        "",
        "## Starting Point",
        "",
        "- `sequential_real` strict acceptance failed only because `cure_retention_no_worse_than_engram=False`.",
        "- CURE reduced reference delta but slightly reduced new-answer gain and previous-edit retention relative to ENGRAM-projected LoRA.",
        "- This run uses `--skip-generation`; evidence is NLL/logprob-based.",
        "",
        "## Methods And Configs",
        "",
        "- Baseline: `C_engram_projected_tiny_lora`, beta `0.5`, no Crisp projection.",
        "- CURE variants: prioritized compact grid over beta, crisp energy threshold, and cache update policy.",
        "",
        "## Final-Step Aggregates",
        "",
        "| config | method | beta | gamma | policy | new decrease | ref delta | retention | positive | locality damage | rollback | match | nan/inf |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in scored_rows:
        lines.append(
            "| {cfg} | {method} | {beta} | {gamma} | {policy} | {new} | {ref} | {ret} | {pos} | {loc} | {roll} | {match} | {nan} |".format(
                cfg=row.get("config_id"),
                method=row.get("method"),
                beta=_format(row.get("beta")),
                gamma=_format(row.get("crisp_energy_threshold")),
                policy=row.get("crisp_cache_update_policy"),
                new=_format(row.get("mean_new_answer_nll_decrease")),
                ref=_format(row.get("mean_reference_delta_abs")),
                ret=_format(row.get("previous_edit_retention")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_records"),
                roll=row.get("rollback_pass"),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Relative To C Baseline",
            "",
            f"- C baseline new-answer decrease: `{_format(c.get('mean_new_answer_nll_decrease'))}`",
            f"- C baseline reference delta: `{_format(c.get('mean_reference_delta_abs'))}`",
            f"- C baseline retention: `{_format(c.get('previous_edit_retention'))}`",
            "",
            "| config | new_answer_ratio | reference_ratio | retention_ratio | promising |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in scored_rows:
        if row.get("method") != "E_cure_dual_projected_tiny_lora":
            continue
        lines.append(
            "| {cfg} | {new} | {ref} | {ret} | {promising} |".format(
                cfg=row.get("config_id"),
                new=_format(row.get("new_answer_ratio")),
                ref=_format(row.get("reference_ratio")),
                ret=_format(row.get("retention_ratio")),
                promising=row.get("pareto_promising"),
            )
        )
    lines.extend(
        [
            "",
            "## Pareto Analysis",
            "",
            f"- Pareto-promising configs: `{[row.get('config_id') for row in promising]}`",
            f"- Best trade-off config: `{best_tradeoff.get('config_id')}`",
            f"- Best trade-off new_answer_ratio: `{_format(best_tradeoff.get('new_answer_ratio'))}`",
            f"- Best trade-off reference_ratio: `{_format(best_tradeoff.get('reference_ratio'))}`",
            f"- Best trade-off retention_ratio: `{_format(best_tradeoff.get('retention_ratio'))}`",
            "",
            "## Projection Diagnostics",
            "",
            "| config | policy | avg mask keep | projection norm ratio | candidate norm | engram norm | cure norm | skipped modules |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics:
        lines.append(
            "| {cfg} | {policy} | {mask} | {proj} | {cand} | {engram} | {cure} | {skipped} |".format(
                cfg=row.get("config_id"),
                policy=row.get("cache_update_policy"),
                mask=_format(row.get("average_mask_keep_ratio")),
                proj=_format(row.get("projection_norm_ratio")),
                cand=_format(row.get("delta_candidate_norm")),
                engram=_format(row.get("delta_engram_norm")),
                cure=_format(row.get("delta_cure_norm")),
                skipped=len(row.get("skipped_modules") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Compact Plots",
            "",
            f"- Plot status: `{plot_status.get('status')}`",
            f"- Plots: `{plot_status.get('plots')}`",
            "",
            "## Limitations",
            "",
            "- Synthetic non-PHI 5-edit only.",
            "- No 20-edit run.",
            "- No direct ENGRAM erase rerun.",
            "- No clinical or medical efficacy claim.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_CURE_SEQUENTIAL_PARETO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compact sequential CURE Pareto tuning on the 5-edit synthetic gate.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_cure_tiny_lora.yaml")
    parser.add_argument("--source-data", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json")
    parser.add_argument("--source-image-root", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/images")
    parser.add_argument("--output-dir", default="outputs/cure_mededit_5edit/sequential_pareto")
    parser.add_argument("--previous-nonseq-dir", default="outputs/cure_mededit_5edit/nonseq_real")
    parser.add_argument("--sequential-report", default="outputs/engram_sequential_5edit_smoke/FINAL_SEQUENTIAL_5EDIT_SMOKE_REPORT.md")
    parser.add_argument("--best-direct-config", default="outputs/engram_token_module_ablation_5edit/best_overall_config.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
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
    _write_env_report(out_dir)
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        source_data=Path(args.source_data),
        source_image_root=Path(args.source_image_root),
        previous_nonseq_dir=Path(args.previous_nonseq_dir),
    )
    if preflight.get("status") != "pass":
        raise RuntimeError(f"Preflight failed: {preflight}")
    _write_failure_summary(out_dir, Path(args.sequential_report), Path(args.best_direct_config))
    replacement_data, image_root, data_summary = _prepare_replacement_data(Path(args.source_data), Path(args.source_image_root), out_dir)
    shutil.copyfile(args.hparams, out_dir / "base_hparams.used.yaml")

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

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    rollback_checks: List[Dict[str, Any]] = []
    traces_by_config: Dict[str, List[Dict[str, Any]]] = {}

    print("[pareto] running C baseline beta=0.5", flush=True)
    c_rows, c_rollback, _ = _run_one_sequential_method(
        model=editor.model,
        method="C_engram_projected_tiny_lora",
        records=records,
        image_root=image_root,
        baselines=baselines,
        projector_bank_dir=out_dir / "projector_bank",
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        beta=0.5,
        threshold=0.7,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
        record_id_match_rate=1.0,
        crisp_cache_update_policy="static",
    )
    all_rows.extend(_annotate_rows(c_rows, config_id="C_baseline_beta0.5", beta=0.5, threshold=0.7, policy="none"))
    summary_rows.extend(
        _summarize_config(
            c_rows,
            config_id="C_baseline_beta0.5",
            method="C_engram_projected_tiny_lora",
            beta=0.5,
            threshold=0.7,
            policy="none",
        )
    )
    rollback_checks.append(c_rollback)

    for config in PRIORITY_CURE_CONFIGS:
        print(f"[pareto] running {config['config_id']}", flush=True)
        rows, rollback, trace = _run_one_sequential_method(
            model=editor.model,
            method="E_cure_dual_projected_tiny_lora",
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=out_dir / "projector_bank",
            module_names=EXPECTED_MODULES,
            hparams=hparams,
            beta=float(config["beta"]),
            threshold=float(config["crisp_energy_threshold"]),
            rollback_tolerance=args.rollback_tolerance,
            locality_threshold=args.locality_damage_threshold,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
            record_id_match_rate=1.0,
            crisp_cache_update_policy=str(config["crisp_cache_update_policy"]),
        )
        all_rows.extend(
            _annotate_rows(
                rows,
                config_id=str(config["config_id"]),
                beta=float(config["beta"]),
                threshold=float(config["crisp_energy_threshold"]),
                policy=str(config["crisp_cache_update_policy"]),
            )
        )
        summary_rows.extend(
            _summarize_config(
                rows,
                config_id=str(config["config_id"]),
                method="E_cure_dual_projected_tiny_lora",
                beta=float(config["beta"]),
                threshold=float(config["crisp_energy_threshold"]),
                policy=str(config["crisp_cache_update_policy"]),
            )
        )
        rollback_checks.append(rollback)
        for item in trace:
            item["config_id"] = str(config["config_id"])
            item["beta"] = float(config["beta"])
            item["crisp_energy_threshold"] = float(config["crisp_energy_threshold"])
        traces_by_config[str(config["config_id"])] = trace
        _json_dump(out_dir / "partial_pareto_status.json", {"completed_config": config["config_id"], "completed_count": len(traces_by_config), "configs": list(traces_by_config)})

    final_rows = [_final_row(summary_rows, "C_baseline_beta0.5")]
    final_rows.extend(_final_row(summary_rows, str(config["config_id"])) for config in PRIORITY_CURE_CONFIGS)
    c_final = final_rows[0]
    scored_rows, best_tradeoff, promising = _score_final_rows(final_rows, c_final)
    diagnostics = [_diag_from_trace(config_id, trace) for config_id, trace in traces_by_config.items()]
    plot_status = _write_plots(out_dir, scored_rows)

    payload = {
        "status": "complete",
        "data_summary": data_summary,
        "previous_nonseq": _load_previous_nonseq(Path(args.previous_nonseq_dir)).get("acceptance", {}),
        "direct_failure": _extract_direct_failure_metrics(Path(args.sequential_report)),
        "configs": [{"config_id": "C_baseline_beta0.5", "method": "C_engram_projected_tiny_lora", "beta": 0.5, "crisp_energy_threshold": 0.7, "crisp_cache_update_policy": "none"}] + PRIORITY_CURE_CONFIGS,
        "summary_rows": summary_rows,
        "final_rows": scored_rows,
        "best_tradeoff": best_tradeoff,
        "pareto_promising_configs": promising,
        "plot_status": plot_status,
        "rollback_checks": rollback_checks,
    }
    _json_dump(out_dir / "pareto_step_matrix.json", all_rows)
    _write_csv(out_dir / "pareto_step_matrix.csv", all_rows)
    _json_dump(out_dir / "pareto_summary.json", payload)
    _write_csv(out_dir / "pareto_summary.csv", scored_rows)
    _json_dump(out_dir / "projection_diagnostics.json", diagnostics)
    trace_rows = [item for trace in traces_by_config.values() for item in trace]
    _json_dump(out_dir / "crisp_cache_update_trace.json", trace_rows)
    _write_final_report(
        out_dir,
        final_rows=final_rows,
        scored_rows=scored_rows,
        best_tradeoff=best_tradeoff,
        promising=promising,
        diagnostics=diagnostics,
        plot_status=plot_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
