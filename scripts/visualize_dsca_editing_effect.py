#!/usr/bin/env python3
"""Create DSCA editing-effect visualizations from a completed pilot run.

This script is intentionally post-processing only: it reads completed run
artifacts, does not import EasyEdit model code, and does not load any model.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


METRIC_COLUMNS = ["rel", "t_gen", "m_gen", "t_loc", "m_loc", "avg"]
METRIC_LABELS = {
    "rel": "Rel.",
    "t_gen": "T-Gen.",
    "m_gen": "M-Gen.",
    "t_loc": "T-Loc.",
    "m_loc": "M-Loc.",
    "avg": "Avg.",
}
EDIT_SAMPLE_TYPES = {"rel", "t_gen", "m_gen"}
LOCALITY_SAMPLE_TYPES = {"t_loc", "m_loc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-case-success", type=int, default=8)
    parser.add_argument("--num-case-failure", type=int, default=8)
    parser.add_argument("--num-case-locality", type=int, default=8)
    parser.add_argument("--num-case-generalization", type=int, default=8)
    parser.add_argument("--make-html", action="store_true")
    parser.add_argument("--make-markdown", action="store_true")
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def safe_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def safe_bool(value: Any) -> Optional[bool]:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def stringify(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(stringify(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def load_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"Missing required artifact: {path}")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        warnings.append(f"Failed to read JSON artifact {path}: {exc}")
        return {}


def load_jsonl(path: Path, warnings: List[str], required: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            warnings.append(f"Missing required artifact: {path}")
        else:
            warnings.append(f"Optional artifact not found: {path}")
        return []
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            warnings.append(f"Skipping invalid JSONL row {path}:{line_no}: {exc}")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_csv(path: Path, warnings: List[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing required artifact: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Failed to read CSV artifact {path}: {exc}")
        return pd.DataFrame()


def load_config(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"Missing required artifact: {path}")
        return {}
    text = path.read_text(errors="replace")
    config: Dict[str, Any] = {"_raw": text}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            config.update(loaded)
            return config
    except Exception:
        pass
    args: Dict[str, str] = {}
    in_args = False
    for line in text.splitlines():
        if line.startswith("args:"):
            in_args = True
            continue
        if in_args:
            if line and not line.startswith(" "):
                break
            match = re.match(r"\s+([^:]+):\s*(.*)$", line)
            if match:
                args[match.group(1).strip()] = match.group(2).strip().strip("'\"")
    if args:
        config["args"] = args
    return config


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def metric_scale(metrics_df: pd.DataFrame, final_metrics: Dict[str, Any]) -> Tuple[float, str, Tuple[float, float]]:
    values: List[float] = []
    for col in METRIC_COLUMNS:
        if col in metrics_df.columns:
            values.extend([float(v) for v in metrics_df[col].dropna().tolist() if not math.isnan(float(v))])
        if col in final_metrics:
            val = safe_float(final_metrics.get(col))
            if val is not None:
                values.append(val)
    if values and max(values) > 1.5:
        return 1.0, "percent", (0.0, 100.0)
    return 1.0, "fraction", (0.0, 1.0)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    available = [c for c in METRIC_COLUMNS if c in metrics_df.columns]
    if metrics_df.empty or "step" not in metrics_df.columns or not available:
        warnings.append("Skipping fig_01_metric_curves.png: missing step or metric columns.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in available:
        ax.plot(metrics_df["step"], metrics_df[col], marker="o", linewidth=1.8, label=METRIC_LABELS[col])
    _, scale_label, ylim = metric_scale(metrics_df, {})
    ax.set_ylim(*ylim)
    ax.set_xlabel("Edit step")
    ax.set_ylabel(f"Metric ({scale_label})")
    ax.set_title("Metric Curves Over Edit Steps")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    path = output_dir / "fig_01_metric_curves.png"
    save_fig(fig, path)
    return path


def plot_final_metric_bars(final_metrics: Dict[str, Any], metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    labels: List[str] = []
    values: List[float] = []
    for col in METRIC_COLUMNS:
        val = safe_float(final_metrics.get(col))
        if val is None and col in metrics_df.columns:
            series = metrics_df[col].dropna()
            if not series.empty:
                val = safe_float(series.iloc[-1])
        if val is not None:
            labels.append(METRIC_LABELS[col])
            values.append(val)
    if not values:
        warnings.append("Skipping fig_02_final_metric_bars.png: no final metrics available.")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#ECA82C", "#B279A2"][: len(values)]
    ax.bar(labels, values, color=colors)
    _, scale_label, ylim = metric_scale(metrics_df, final_metrics)
    ax.set_ylim(*ylim)
    ax.set_ylabel(f"Metric ({scale_label})")
    ax.set_title("Final Metrics")
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "fig_02_final_metric_bars.png"
    save_fig(fig, path)
    return path


def plot_metric_heatmap(metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    available = [c for c in METRIC_COLUMNS if c in metrics_df.columns]
    if metrics_df.empty or "step" not in metrics_df.columns or not available:
        warnings.append("Skipping fig_03_metric_heatmap_by_step.png: missing step or metric columns.")
        return None
    data = metrics_df[available].T.to_numpy(dtype=float)
    steps = [str(int(s)) if safe_float(s) is not None else str(s) for s in metrics_df["step"].tolist()]
    fig, ax = plt.subplots(figsize=(max(9, len(steps) * 0.45), 4.8))
    image = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([METRIC_LABELS[c] for c in available])
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.set_xlabel("Edit step")
    ax.set_title("Metric Heatmap By Step")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    path = output_dir / "fig_03_metric_heatmap_by_step.png"
    save_fig(fig, path)
    return path


def aliases_for(row: Dict[str, Any]) -> List[str]:
    aliases = row.get("aliases")
    if aliases is None:
        return []
    if isinstance(aliases, str):
        return [aliases]
    if isinstance(aliases, Iterable):
        return [str(x) for x in aliases if not is_missing(x)]
    return []


def prediction_matches(row: Dict[str, Any], prefix: str) -> Optional[bool]:
    pred = row.get(f"{prefix}_prediction")
    target = row.get("target")
    if is_missing(pred) or is_missing(target):
        return None
    pred_norm = normalize_text(pred)
    targets = [normalize_text(target)] + [normalize_text(a) for a in aliases_for(row)]
    targets = [t for t in targets if t]
    if not pred_norm or not targets:
        return None
    return any(pred_norm == t or t in pred_norm for t in targets)


def edited_correct(row: Dict[str, Any]) -> Optional[bool]:
    for key in ["contains_target", "exact_match_normalized", "exact_match_raw"]:
        val = safe_bool(row.get(key))
        if val is not None:
            return val
    if row.get("sample_type") in EDIT_SAMPLE_TYPES:
        val = safe_bool(row.get("correct_or_preserved"))
        if val is not None:
            return val
        score = safe_float(row.get("score"))
        if score is not None:
            return score > 0.5
    return prediction_matches(row, "edited")


def base_correct(row: Dict[str, Any]) -> Optional[bool]:
    return prediction_matches(row, "base")


def locality_preserved(row: Dict[str, Any]) -> Optional[bool]:
    for key in ["preserved_for_locality", "correct_or_preserved"]:
        val = safe_bool(row.get(key))
        if val is not None:
            return val
    score = safe_float(row.get("score"))
    if score is not None:
        return score > 0.5
    return None


def prediction_effect_counts(pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in pred_rows:
        sample_type = str(row.get("sample_type", "unknown"))
        if sample_type in EDIT_SAMPLE_TYPES:
            b = base_correct(row)
            e = edited_correct(row)
            if b is False and e is True:
                key = "base wrong -> edited correct"
            elif b is True and e is True:
                key = "base correct -> edited correct"
            elif b is True and e is False:
                key = "base correct -> edited wrong"
            elif b is False and e is False:
                key = "base wrong -> edited wrong"
            else:
                key = "missing"
            counts[sample_type][key] += 1
        elif sample_type in LOCALITY_SAMPLE_TYPES:
            p = locality_preserved(row)
            if p is True:
                key = "preserved"
            elif p is False:
                key = "changed / locality broken"
            else:
                key = "missing"
            counts[sample_type][key] += 1
        else:
            counts[sample_type]["missing"] += 1
    return counts


def plot_pre_vs_post_counts(counts: Dict[str, Counter], output_dir: Path, warnings: List[str]) -> Optional[Path]:
    if not counts:
        warnings.append("Skipping fig_04_pre_vs_post_effect_counts.png: no predictions available.")
        return None
    categories = [
        "base wrong -> edited correct",
        "base correct -> edited correct",
        "base correct -> edited wrong",
        "base wrong -> edited wrong",
        "preserved",
        "changed / locality broken",
        "missing",
    ]
    sample_types = sorted(counts.keys(), key=lambda x: ["rel", "t_gen", "m_gen", "t_loc", "m_loc", x].index(x) if x in {"rel", "t_gen", "m_gen", "t_loc", "m_loc"} else 99)
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(sample_types))
    palette = {
        "base wrong -> edited correct": "#2E7D32",
        "base correct -> edited correct": "#66A61E",
        "base correct -> edited wrong": "#D62728",
        "base wrong -> edited wrong": "#8C564B",
        "preserved": "#1B9E77",
        "changed / locality broken": "#E41A1C",
        "missing": "#9E9E9E",
    }
    for cat in categories:
        values = np.array([counts[st].get(cat, 0) for st in sample_types])
        if values.sum() == 0:
            continue
        ax.bar(sample_types, values, bottom=bottom, label=cat, color=palette.get(cat))
        bottom += values
    ax.set_ylabel("Prediction rows")
    ax.set_title("Pre-vs-Post Editing Effect Counts")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "fig_04_pre_vs_post_effect_counts.png"
    save_fig(fig, path)
    return path


def plot_contains_vs_exact(pred_rows: Sequence[Dict[str, Any]], output_dir: Path, warnings: List[str]) -> Optional[Path]:
    if not pred_rows:
        warnings.append("Skipping fig_05_contains_vs_exact_match.png: no predictions available.")
        return None
    rows = [r for r in pred_rows if r.get("sample_type") in EDIT_SAMPLE_TYPES]
    has_contains = any("contains_target" in r for r in rows)
    has_exact = any("exact_match_normalized" in r for r in rows)
    if not has_contains and not has_exact:
        warnings.append("Skipping fig_05_contains_vs_exact_match.png: contains/exact fields are absent.")
        return None
    sample_types = ["rel", "t_gen", "m_gen"]
    exact_vals: List[float] = []
    contains_vals: List[float] = []
    for st in sample_types:
        part = [r for r in rows if r.get("sample_type") == st]
        exact = [safe_bool(r.get("exact_match_normalized")) for r in part if safe_bool(r.get("exact_match_normalized")) is not None]
        contains = [safe_bool(r.get("contains_target")) for r in part if safe_bool(r.get("contains_target")) is not None]
        exact_vals.append(float(np.mean(exact)) if exact else np.nan)
        contains_vals.append(float(np.mean(contains)) if contains else np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sample_types))
    width = 0.35
    ax.bar(x - width / 2, exact_vals, width, label="exact_match_normalized", color="#4C78A8")
    ax.bar(x + width / 2, contains_vals, width, label="contains_target", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_types)
    ax.set_ylim(0, 1)
    ax.set_title("Contains Target vs Exact Match")
    ax.set_ylabel("Mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "fig_05_contains_vs_exact_match.png"
    save_fig(fig, path)
    return path


def plot_cluster_growth(metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    cols = [c for c in ["num_clusters", "num_active_dsams"] if c in metrics_df.columns]
    if metrics_df.empty or "step" not in metrics_df.columns or not cols:
        warnings.append("Skipping fig_06_cluster_growth.png: missing cluster columns.")
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in cols:
        ax.plot(metrics_df["step"], metrics_df[col], marker="o", label=col)
    ax.set_xlabel("Edit step")
    ax.set_ylabel("Count")
    ax.set_title("DSCA Cluster And Active DSAM Growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "fig_06_cluster_growth.png"
    save_fig(fig, path)
    return path


def plot_subspace_overlap(metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    if metrics_df.empty or "step" not in metrics_df.columns or "mean_subspace_overlap" not in metrics_df.columns:
        warnings.append("Skipping fig_07_subspace_overlap.png: missing mean_subspace_overlap.")
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    y = metrics_df["mean_subspace_overlap"].astype(float)
    ax.plot(metrics_df["step"], y, marker="o", color="#6F4E7C")
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("Edit step")
    ax.set_ylabel("Mean subspace overlap")
    ax.set_title("Mean Subspace Overlap")
    if y.dropna().abs().max() > 0 and y.dropna().abs().max() < 1.0e-3:
        ax.set_yscale("symlog", linthresh=1.0e-6)
    ax.grid(True, alpha=0.25)
    path = output_dir / "fig_07_subspace_overlap.png"
    save_fig(fig, path)
    return path


def plot_routing_and_residual(metrics_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    cols = [c for c in ["avg_candidates", "residual_norm_mean", "route_weight_l1_replay"] if c in metrics_df.columns]
    if metrics_df.empty or "step" not in metrics_df.columns or not cols:
        warnings.append("Skipping fig_08_routing_and_residual.png: missing routing/residual columns.")
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 3.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(metrics_df["step"], metrics_df[col], marker="o")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Edit step")
    fig.suptitle("Routing And Residual Diagnostics")
    path = output_dir / "fig_08_routing_and_residual.png"
    save_fig(fig, path)
    return path


def diagnostics_df(diag_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not diag_rows:
        return pd.DataFrame()
    flat: List[Dict[str, Any]] = []
    for row in diag_rows:
        item = dict(row)
        summary = item.pop("route_weights_summary", None)
        if isinstance(summary, dict):
            for key, value in summary.items():
                item[f"route_weights_{key}"] = value
        flat.append(item)
    return pd.DataFrame(flat)


def plot_activation_events(diag_df: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    cols = [c for c in ["new_clusters_created", "new_dsams_activated"] if c in diag_df.columns]
    if diag_df.empty or "step" not in diag_df.columns or not cols:
        warnings.append("Skipping fig_09_dsam_activation_events.png: missing activation event columns.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(diag_df))
    width = 0.35
    for i, col in enumerate(cols):
        ax.bar(x + (i - 0.5) * width, diag_df[col], width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in diag_df["step"]], rotation=45, ha="right")
    ax.set_xlabel("Edit step")
    ax.set_ylabel("Event count")
    ax.set_title("Cluster And DSAM Activation Events")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "fig_09_dsam_activation_events.png"
    save_fig(fig, path)
    return path


def safety_flags(final_summary: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    pass_fail = final_summary.get("pass_fail_flags") if isinstance(final_summary.get("pass_fail_flags"), dict) else {}
    return {
        "base_vlm_params_changed": safe_bool(pass_fail.get("base_vlm_params_changed")),
        "R_k_requires_grad_any": safe_bool(pass_fail.get("R_k_requires_grad_any")),
        "duplicate_optimizer_param_groups": safe_bool(pass_fail.get("duplicate_optimizer_param_groups")),
        "repository_save_load": safe_bool(pass_fail.get("repository_save_load")),
        "loss_finite": safe_bool(pass_fail.get("loss_finite")),
    }


def plot_safety_checks(flags: Dict[str, Optional[bool]], output_dir: Path, warnings: List[str]) -> Optional[Path]:
    if not flags:
        warnings.append("Skipping fig_10_safety_checks.png: safety flags unavailable.")
        return None
    labels = list(flags.keys())
    pass_values = []
    colors = []
    for key in labels:
        val = flags[key]
        if key in {"base_vlm_params_changed", "R_k_requires_grad_any", "duplicate_optimizer_param_groups"}:
            passed = val is False
        elif key in {"repository_save_load", "loss_finite"}:
            passed = val is True
        else:
            passed = False
        pass_values.append(1 if passed else 0)
        colors.append("#2E7D32" if passed else "#D62728")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(labels, pass_values, color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Pass")
    ax.set_title("Safety Checks")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for i, (key, val) in enumerate(flags.items()):
        ax.text(i, 1.05, stringify(val), ha="center", va="bottom", fontsize=9)
    path = output_dir / "fig_10_safety_checks.png"
    save_fig(fig, path)
    return path


def aggregate_profile(profile_rows: Sequence[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    done = [r for r in profile_rows if r.get("event") == "done" and safe_float(r.get("elapsed_sec")) is not None]
    if not done:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(done)
    df["elapsed_sec"] = df["elapsed_sec"].astype(float)
    by_step = df.pivot_table(index="step", columns="phase", values="elapsed_sec", aggfunc="sum").reset_index()
    totals = df.groupby("phase", as_index=False)["elapsed_sec"].sum().sort_values("elapsed_sec", ascending=False)
    return by_step, totals


def plot_timing_per_step(metrics_df: pd.DataFrame, profile_by_step: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    if not metrics_df.empty and "step" in metrics_df.columns and "time_per_edit_sec" in metrics_df.columns:
        ax.plot(metrics_df["step"], metrics_df["time_per_edit_sec"], marker="o", label="time_per_edit_sec")
        plotted = True
    for col in ["edit_step_total", "refine_subspaces", "residualized_pca"]:
        if not profile_by_step.empty and col in profile_by_step.columns:
            ax.plot(profile_by_step["step"], profile_by_step[col], marker="o", label=col)
            plotted = True
    if not plotted:
        plt.close(fig)
        warnings.append("Skipping fig_11_timing_per_step.png: timing columns unavailable.")
        return None
    ax.set_xlabel("Edit step")
    ax.set_ylabel("Seconds")
    ax.set_title("Timing Per Step")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = output_dir / "fig_11_timing_per_step.png"
    save_fig(fig, path)
    return path


def plot_top_profile_phases(profile_totals: pd.DataFrame, output_dir: Path, warnings: List[str]) -> Optional[Path]:
    if profile_totals.empty or "phase" not in profile_totals.columns:
        warnings.append("Skipping fig_12_top_slowest_profile_phases.png: profile data unavailable.")
        return None
    top = profile_totals.head(15).sort_values("elapsed_sec", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["phase"], top["elapsed_sec"], color="#4C78A8")
    ax.set_xlabel("Total elapsed seconds")
    ax.set_title("Top Slowest Profile Phases")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / "fig_12_top_slowest_profile_phases.png"
    save_fig(fig, path)
    return path


def resolve_image_path(row: Dict[str, Any], run_dir: Path, image_root: Path) -> Tuple[Optional[Path], List[str]]:
    raw = row.get("image_path")
    tried: List[str] = []
    if is_missing(raw) or not str(raw).strip():
        return None, tried
    raw_path = Path(str(raw))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([image_root / raw_path, run_dir / raw_path])
        dataset_root = image_root.parent if image_root.name == "images" else image_root
        candidates.append(dataset_root / raw_path)
    for cand in candidates:
        tried.append(str(cand))
        if cand.exists():
            return cand, tried
    return None, tried


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:8]


def draw_case_card(
    row: Dict[str, Any],
    category: str,
    status: str,
    run_dir: Path,
    image_root: Path,
    warnings: List[str],
    card_size: Tuple[int, int] = (560, 520),
) -> Image.Image:
    width, height = card_size
    status_color = {
        "success": (224, 245, 226),
        "failure": (253, 232, 232),
        "unknown": (240, 240, 240),
    }.get(status, (240, 240, 240))
    border_color = {
        "success": (46, 125, 50),
        "failure": (198, 40, 40),
        "unknown": (120, 120, 120),
    }.get(status, (120, 120, 120))
    image = Image.new("RGB", (width, height), status_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold = ImageFont.load_default()
    draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=4)
    draw.text((14, 12), f"{category} | step {row.get('step', '')} | {row.get('sample_type', '')}", fill=(0, 0, 0), font=bold)

    thumb_box = (14, 42, 210, 210)
    resolved, tried = resolve_image_path(row, run_dir, image_root)
    if resolved is not None:
        try:
            with Image.open(resolved) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_box[2] - thumb_box[0], thumb_box[3] - thumb_box[1]))
                x = thumb_box[0] + ((thumb_box[2] - thumb_box[0]) - img.width) // 2
                y = thumb_box[1] + ((thumb_box[3] - thumb_box[1]) - img.height) // 2
                draw.rectangle(thumb_box, fill=(255, 255, 255), outline=(180, 180, 180))
                image.paste(img, (x, y))
        except Exception as exc:
            warnings.append(f"Failed to render image for step {row.get('step')} {row.get('sample_type')}: {resolved}: {exc}")
            draw.rectangle(thumb_box, fill=(245, 245, 245), outline=(180, 180, 180))
            draw.text((34, 116), "image render failed", fill=(90, 90, 90), font=font)
    else:
        if tried:
            warnings.append(f"Image not found for step {row.get('step')} {row.get('sample_type')}: tried {tried}")
        draw.rectangle(thumb_box, fill=(245, 245, 245), outline=(180, 180, 180))
        draw.text((50, 116), "image not found", fill=(90, 90, 90), font=font)

    fields = [
        ("Prompt", row.get("prompt")),
        ("Target", row.get("target")),
        ("Aliases", row.get("aliases")),
        ("Base", row.get("base_prediction")),
        ("Edited", row.get("edited_prediction")),
        ("Exact", row.get("exact_match_normalized")),
        ("Contains", row.get("contains_target")),
        ("Preserved", row.get("preserved_for_locality", row.get("correct_or_preserved"))),
        ("Notes", row.get("missing_field_notes", row.get("warning"))),
    ]
    y = 224
    for label, value in fields:
        text = f"{label}: {stringify(value)}"
        for line in wrap_text(draw, text, font, width - 32):
            if y > height - 24:
                break
            draw.text((14, y), line, fill=(0, 0, 0), font=font)
            y += 18
        y += 3
        if y > height - 24:
            draw.text((14, height - 22), "...", fill=(0, 0, 0), font=font)
            break
    return image


def make_case_grid(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    category: str,
    status: str,
    run_dir: Path,
    image_root: Path,
    warnings: List[str],
    max_cases: int,
) -> Optional[Path]:
    selected = list(rows)[:max_cases]
    if not selected:
        warnings.append(f"No cases available for {path.name}.")
        selected = []
    cards = []
    for row in selected:
        row_status = status
        if status == "auto":
            if row.get("sample_type") in LOCALITY_SAMPLE_TYPES:
                preserved = locality_preserved(row)
                row_status = "success" if preserved is True else "failure" if preserved is False else "unknown"
            else:
                correct = edited_correct(row)
                row_status = "success" if correct is True else "failure" if correct is False else "unknown"
        cards.append(draw_case_card(row, category, row_status, run_dir, image_root, warnings))
    if not cards:
        canvas = Image.new("RGB", (900, 180), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 70), f"No cases available for {category}", fill=(80, 80, 80), font=ImageFont.load_default())
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return path
    cols = 2
    rows_count = int(math.ceil(len(cards) / cols))
    card_w, card_h = cards[0].size
    gutter = 18
    canvas = Image.new("RGB", (cols * card_w + (cols + 1) * gutter, rows_count * card_h + (rows_count + 1) * gutter), (255, 255, 255))
    for idx, card in enumerate(cards):
        row_idx = idx // cols
        col_idx = idx % cols
        x = gutter + col_idx * (card_w + gutter)
        y = gutter + row_idx * (card_h + gutter)
        canvas.paste(card, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def select_case_rows(pred_rows: Sequence[Dict[str, Any]], args: argparse.Namespace, warnings: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    rel = [r for r in pred_rows if r.get("sample_type") == "rel"]
    t_gen = [r for r in pred_rows if r.get("sample_type") == "t_gen"]
    m_gen = [r for r in pred_rows if r.get("sample_type") == "m_gen"]
    loc = [r for r in pred_rows if r.get("sample_type") in LOCALITY_SAMPLE_TYPES]

    reliability_success = [r for r in rel if edited_correct(r) is True and base_correct(r) is not True]
    reliability_failure = [r for r in rel if edited_correct(r) is not True]
    text_generalization = sorted(t_gen, key=lambda r: 0 if edited_correct(r) is True else 1)
    modal_generalization = sorted(m_gen, key=lambda r: 0 if edited_correct(r) is True else 1)
    locality_preserved_rows = [r for r in loc if locality_preserved(r) is True]
    locality_broken_rows = [r for r in loc if locality_preserved(r) is False]
    return {
        "reliability_success": reliability_success[: args.num_case_success],
        "reliability_failure": reliability_failure[: args.num_case_failure],
        "text_generalization": text_generalization[: args.num_case_generalization],
        "modal_generalization": modal_generalization[: args.num_case_generalization],
        "locality_preserved": locality_preserved_rows[: args.num_case_locality],
        "locality_broken": locality_broken_rows[: args.num_case_locality],
    }


def make_case_galleries(
    pred_rows: Sequence[Dict[str, Any]],
    run_dir: Path,
    image_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    warnings: List[str],
) -> List[Path]:
    gallery_dir = output_dir / "case_gallery"
    selected = select_case_rows(pred_rows, args, warnings)
    specs = [
        ("reliability_success_grid.png", "Reliability Success", "success", "reliability_success"),
        ("reliability_failure_grid.png", "Reliability Failure", "failure", "reliability_failure"),
        ("text_generalization_grid.png", "Text Generalization", "auto", "text_generalization"),
        ("modal_generalization_grid.png", "Modal Generalization", "auto", "modal_generalization"),
        ("locality_preserved_grid.png", "Locality Preserved", "success", "locality_preserved"),
        ("locality_broken_grid.png", "Locality Broken", "failure", "locality_broken"),
    ]
    paths: List[Path] = []
    for filename, title, status, key in specs:
        path = make_case_grid(selected[key], gallery_dir / filename, title, status, run_dir, image_root, warnings, max_cases=999)
        if path is not None:
            paths.append(path)
    return paths


def dataframe_tail_html(df: pd.DataFrame, rows: int = 5) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    return df.tail(rows).to_html(index=False, escape=True, border=0, classes="data-table")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    work = df.copy()
    work = work.where(pd.notnull(work), "")
    headers = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in work.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in work.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def relative_link(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def format_metric(value: Any) -> str:
    val = safe_float(value)
    if val is None:
        return "N/A"
    return f"{val:.4g}"


def interpretation_notes(
    metrics_df: pd.DataFrame,
    pred_rows: Sequence[Dict[str, Any]],
    counts: Dict[str, Counter],
) -> List[str]:
    notes: List[str] = []
    rel_success = counts.get("rel", Counter()).get("base wrong -> edited correct", 0)
    rel_total = sum(counts.get("rel", Counter()).values())
    notes.append(f"Reliability improved on {rel_success}/{rel_total} reliability prediction rows.")
    for st, label in [("t_gen", "T-Gen"), ("m_gen", "M-Gen")]:
        total = sum(counts.get(st, Counter()).values())
        success = counts.get(st, Counter()).get("base wrong -> edited correct", 0) + counts.get(st, Counter()).get("base correct -> edited correct", 0)
        if total:
            notes.append(f"{label} had {success}/{total} rows counted as edited-correct.")
    locality_total = sum(sum(counts.get(st, Counter()).values()) for st in LOCALITY_SAMPLE_TYPES)
    locality_broken = sum(counts.get(st, Counter()).get("changed / locality broken", 0) for st in LOCALITY_SAMPLE_TYPES)
    locality_missing = sum(counts.get(st, Counter()).get("missing", 0) for st in LOCALITY_SAMPLE_TYPES)
    if locality_total:
        if locality_broken == 0:
            notes.append(f"No explicit locality breaks were observed among {locality_total} locality rows; {locality_missing} rows were missing/unknown.")
        else:
            notes.append(f"Locality was broken in {locality_broken}/{locality_total} rows; {locality_missing} rows were missing/unknown.")
    if "mean_subspace_overlap" in metrics_df.columns and not metrics_df["mean_subspace_overlap"].dropna().empty:
        max_overlap = float(metrics_df["mean_subspace_overlap"].dropna().max())
        notes.append(f"Mean subspace overlap remained at or below {max_overlap:.4g}.")
    if "num_active_dsams" in metrics_df.columns and not metrics_df["num_active_dsams"].dropna().empty:
        max_active = int(metrics_df["num_active_dsams"].dropna().max())
        first_step = metrics_df.loc[metrics_df["num_active_dsams"] == max_active, "step"].iloc[0]
        notes.append(f"Active DSAMs reached {max_active} by step {int(first_step)}.")
    return notes


def build_visual_summary(
    final_summary: Dict[str, Any],
    metrics_df: pd.DataFrame,
    pred_rows: Sequence[Dict[str, Any]],
    counts: Dict[str, Counter],
    figure_paths: Sequence[Path],
    case_paths: Sequence[Path],
    output_dir: Path,
    warnings: List[str],
) -> Dict[str, Any]:
    final_metrics = final_summary.get("final_metrics") if isinstance(final_summary.get("final_metrics"), dict) else {}
    mean_metrics = final_summary.get("mean_metrics") if isinstance(final_summary.get("mean_metrics"), dict) else {}
    if not mean_metrics and not metrics_df.empty:
        mean_metrics = {c: safe_float(metrics_df[c].mean()) for c in METRIC_COLUMNS if c in metrics_df.columns}
    sample_counts = Counter(str(r.get("sample_type", "unknown")) for r in pred_rows)
    edit_success = {
        st: {
            "success": counts.get(st, Counter()).get("base wrong -> edited correct", 0)
            + counts.get(st, Counter()).get("base correct -> edited correct", 0),
            "failure": counts.get(st, Counter()).get("base correct -> edited wrong", 0)
            + counts.get(st, Counter()).get("base wrong -> edited wrong", 0),
            "missing": counts.get(st, Counter()).get("missing", 0),
        }
        for st in ["rel", "t_gen", "m_gen"]
    }
    locality = {
        "preserved": sum(counts.get(st, Counter()).get("preserved", 0) for st in LOCALITY_SAMPLE_TYPES),
        "broken": sum(counts.get(st, Counter()).get("changed / locality broken", 0) for st in LOCALITY_SAMPLE_TYPES),
        "missing": sum(counts.get(st, Counter()).get("missing", 0) for st in LOCALITY_SAMPLE_TYPES),
    }
    last = metrics_df.tail(1).iloc[0].to_dict() if not metrics_df.empty else {}
    return {
        "final_metrics": final_metrics,
        "mean_metrics": mean_metrics,
        "examples_per_sample_type": dict(sample_counts),
        "editing_effect_counts": {k: dict(v) for k, v in counts.items()},
        "edit_success_failure": edit_success,
        "locality": locality,
        "final_num_clusters": safe_float(last.get("num_clusters")),
        "final_num_active_dsams": safe_float(last.get("num_active_dsams")),
        "final_mean_subspace_overlap": safe_float(last.get("mean_subspace_overlap")),
        "max_residual_norm_mean": safe_float(metrics_df["residual_norm_mean"].max()) if "residual_norm_mean" in metrics_df.columns else None,
        "avg_candidates_mean": safe_float(metrics_df["avg_candidates"].mean()) if "avg_candidates" in metrics_df.columns else None,
        "avg_candidates_max": safe_float(metrics_df["avg_candidates"].max()) if "avg_candidates" in metrics_df.columns else None,
        "safety_flags": safety_flags(final_summary),
        "artifact_paths": {
            "figures": [relative_link(p, output_dir) for p in figure_paths],
            "case_galleries": [relative_link(p, output_dir) for p in case_paths],
        },
        "warnings": warnings,
    }


def metadata_from(final_summary: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg_args = config.get("args") if isinstance(config.get("args"), dict) else {}
    return {
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": cfg_args.get("model") or final_summary.get("model") or "blip2",
        "dataset": cfg_args.get("dataset") or final_summary.get("dataset"),
        "num_edits": cfg_args.get("num_edits") or final_summary.get("num_edits"),
        "rank": cfg_args.get("rank"),
        "min_samples": cfg_args.get("min_samples"),
        "refine_interval": cfg_args.get("refine_interval"),
        "device": cfg_args.get("device"),
        "total_runtime_sec": final_summary.get("total_runtime_sec"),
        "peak_gpu_memory_mb": final_summary.get("peak_gpu_memory_mb"),
    }


def make_html_report(
    output_dir: Path,
    metadata: Dict[str, Any],
    final_summary: Dict[str, Any],
    metrics_df: pd.DataFrame,
    figure_paths: Sequence[Path],
    case_paths: Sequence[Path],
    warnings: List[str],
    notes: Sequence[str],
) -> Path:
    final_metrics = final_summary.get("final_metrics") if isinstance(final_summary.get("final_metrics"), dict) else {}
    flags = safety_flags(final_summary)
    metric_cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(METRIC_LABELS[col])}</div><div class='value'>{html.escape(format_metric(final_metrics.get(col)))}</div></div>"
        for col in METRIC_COLUMNS
    )
    safety_cards = "\n".join(
        f"<div class='card safety'><div class='label'>{html.escape(k)}</div><div class='value'>{html.escape(stringify(v))}</div></div>"
        for k, v in flags.items()
    )
    figure_html = "\n".join(
        f"<figure><img src='{html.escape(relative_link(p, output_dir))}' alt='{html.escape(p.name)}'><figcaption>{html.escape(p.name)}</figcaption></figure>"
        for p in figure_paths
    )
    case_html = "\n".join(
        f"<figure><img src='{html.escape(relative_link(p, output_dir))}' alt='{html.escape(p.name)}'><figcaption>{html.escape(p.name)}</figcaption></figure>"
        for p in case_paths
    )
    warnings_html = "<ul>" + "\n".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul>" if warnings else "<p>No warnings.</p>"
    notes_html = "<ul>" + "\n".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>"
    metadata_rows = "\n".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(stringify(v))}</td></tr>" for k, v in metadata.items())
    pretty_json = html.escape(json.dumps(json_ready(final_summary), indent=2, ensure_ascii=False))
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DSCA MedMKEB 20-edit BLIP2-OPT Visualization Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ color: #1f2933; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 24px; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .card .label {{ font-size: 12px; color: #57606a; }}
    .card .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .safety .value {{ font-size: 18px; }}
    figure {{ margin: 20px 0; }}
    figure img {{ max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }}
    figcaption {{ color: #57606a; font-size: 13px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    pre {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>DSCA MedMKEB 20-edit BLIP2-OPT Visualization Report</h1>
  <h2>Run Metadata</h2>
  <table>{metadata_rows}</table>
  <h2>Final Metrics</h2>
  <div class="cards">{metric_cards}</div>
  <h2>Safety Checks</h2>
  <div class="cards">{safety_cards}</div>
  <h2>Interpretation Notes</h2>
  {notes_html}
  <h2>Figures</h2>
  {figure_html}
  <h2>Case Galleries</h2>
  {case_html}
  <h2>Last 5 Metric Rows</h2>
  {dataframe_tail_html(metrics_df, 5)}
  <h2>Warnings And Missing Artifacts</h2>
  {warnings_html}
  <h2>Final Summary JSON</h2>
  <pre>{pretty_json}</pre>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(doc)
    return path


def make_markdown_report(
    output_dir: Path,
    metadata: Dict[str, Any],
    final_summary: Dict[str, Any],
    metrics_df: pd.DataFrame,
    figure_paths: Sequence[Path],
    case_paths: Sequence[Path],
    warnings: List[str],
    notes: Sequence[str],
    pred_rows: Sequence[Dict[str, Any]],
) -> Path:
    final_metrics = final_summary.get("final_metrics") if isinstance(final_summary.get("final_metrics"), dict) else {}
    lines: List[str] = ["# DSCA MedMKEB 20-edit BLIP2-OPT Visualization Report", ""]
    lines.append("## Run Metadata")
    for key, value in metadata.items():
        lines.append(f"- `{key}`: {stringify(value)}")
    lines.extend(["", "## Final Metrics", "", "| Metric | Value |", "|---|---|"])
    for col in METRIC_COLUMNS:
        lines.append(f"| {METRIC_LABELS[col]} | {format_metric(final_metrics.get(col))} |")
    lines.extend(["", "## Safety Flags"])
    for key, value in safety_flags(final_summary).items():
        lines.append(f"- `{key}`: {stringify(value)}")
    lines.extend(["", "## Interpretation Notes"])
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Figures"])
    for path in figure_paths:
        rel = relative_link(path, output_dir)
        lines.append(f"- [{path.name}]({rel})")
        lines.append(f"  ![{path.name}]({rel})")
    lines.extend(["", "## Case Galleries"])
    for path in case_paths:
        rel = relative_link(path, output_dir)
        lines.append(f"- [{path.name}]({rel})")
        lines.append(f"  ![{path.name}]({rel})")
    lines.extend(["", "## Last 5 Metric Rows", ""])
    if metrics_df.empty:
        lines.append("No metric rows available.")
    else:
        lines.append(dataframe_to_markdown(metrics_df.tail(5)))
    lines.extend(["", "## Selected Case Examples"])
    for row in list(pred_rows)[:10]:
        lines.append(
            f"- step {row.get('step')} `{row.get('sample_type')}`: prompt={stringify(row.get('prompt'))!r}; "
            f"target={stringify(row.get('target'))!r}; base={stringify(row.get('base_prediction'))!r}; "
            f"edited={stringify(row.get('edited_prediction'))!r}; score={stringify(row.get('score'))!r}"
        )
    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Next Recommended Experiment",
            "",
            "Run a no-profile 20-edit baseline for clean timing, then move to a larger MedMKEB pilot only if the timing and metric behavior are acceptable.",
            "",
        ]
    )
    path = output_dir / "visual_report.md"
    path.write_text("\n".join(lines))
    return path


def create_archive(output_dir: Path) -> Path:
    archive_path = output_dir.with_suffix(".tar.gz")
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path.is_dir():
                continue
            tar.add(path, arcname=path.relative_to(output_dir))
    return archive_path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    artifacts = {
        "final_summary": run_dir / "final_summary.json",
        "metrics": run_dir / "metrics_per_step.csv",
        "predictions": run_dir / "predictions.jsonl",
        "diagnostics": run_dir / "dsca_diagnostics.jsonl",
        "config": run_dir / "config_resolved.yaml",
        "profile": run_dir / "dsca_edit_step_profile.jsonl",
        "profile_summary": run_dir / "profile_summary.txt",
    }

    final_summary = load_json(artifacts["final_summary"], warnings)
    metrics_df = load_csv(artifacts["metrics"], warnings)
    pred_rows = load_jsonl(artifacts["predictions"], warnings)
    diag_rows = load_jsonl(artifacts["diagnostics"], warnings)
    config = load_config(artifacts["config"], warnings)
    profile_rows = load_jsonl(artifacts["profile"], warnings, required=False)
    if not artifacts["profile_summary"].exists():
        warnings.append(f"Optional artifact not found: {artifacts['profile_summary']}")

    final_metrics = final_summary.get("final_metrics") if isinstance(final_summary.get("final_metrics"), dict) else {}
    diag_df = diagnostics_df(diag_rows)
    profile_by_step, profile_totals = aggregate_profile(profile_rows)
    counts = prediction_effect_counts(pred_rows)

    figure_paths: List[Path] = []
    for maybe_path in [
        plot_metric_curves(metrics_df, output_dir, warnings),
        plot_final_metric_bars(final_metrics, metrics_df, output_dir, warnings),
        plot_metric_heatmap(metrics_df, output_dir, warnings),
        plot_pre_vs_post_counts(counts, output_dir, warnings),
        plot_contains_vs_exact(pred_rows, output_dir, warnings),
        plot_cluster_growth(metrics_df, output_dir, warnings),
        plot_subspace_overlap(metrics_df, output_dir, warnings),
        plot_routing_and_residual(metrics_df, output_dir, warnings),
        plot_activation_events(diag_df, output_dir, warnings),
        plot_safety_checks(safety_flags(final_summary), output_dir, warnings),
        plot_timing_per_step(metrics_df, profile_by_step, output_dir, warnings),
        plot_top_profile_phases(profile_totals, output_dir, warnings),
    ]:
        if maybe_path is not None:
            figure_paths.append(maybe_path)

    case_paths = make_case_galleries(pred_rows, run_dir, image_root, output_dir, args, warnings)
    notes = interpretation_notes(metrics_df, pred_rows, counts)
    metadata = metadata_from(final_summary, config, args)

    summary = build_visual_summary(final_summary, metrics_df, pred_rows, counts, figure_paths, case_paths, output_dir, warnings)
    summary_path = output_dir / "visual_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))

    warnings_path = output_dir / "visualization_warnings.json"
    warnings_path.write_text(json.dumps(json_ready({"warnings": warnings}), indent=2, ensure_ascii=False))

    html_path: Optional[Path] = None
    md_path: Optional[Path] = None
    if args.make_html:
        html_path = make_html_report(output_dir, metadata, final_summary, metrics_df, figure_paths, case_paths, warnings, notes)
    if args.make_markdown:
        md_path = make_markdown_report(output_dir, metadata, final_summary, metrics_df, figure_paths, case_paths, warnings, notes, pred_rows)
    archive_path = create_archive(output_dir)

    print(json.dumps(json_ready({
        "success": True,
        "output_dir": output_dir,
        "archive_path": archive_path,
        "html": html_path,
        "markdown": md_path,
        "summary": summary_path,
        "figures": [p.name for p in figure_paths],
        "case_galleries": [relative_link(p, output_dir) for p in case_paths],
        "warnings_count": len(warnings),
    }), indent=2))


if __name__ == "__main__":
    main()
