#!/usr/bin/env python3
"""Analyze saved LLaVA-Med DSCA generation-path diagnostic outputs.

This script intentionally does not run model inference. It only reads files
already produced by the generation-path diagnostic.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


UNAVAILABLE = "not_captured_in_existing_diagnostic"
GENERIC_PREFIXES = (
    "this",
    "the most likely",
    "the marked area",
    "the image",
    "the picture",
    "it shows",
    "there is",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze existing LLaVA-Med generation diagnostic outputs without running inference."
    )
    parser.add_argument("--diag-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_literal(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def text_len(text: str) -> int:
    return len([part for part in str(text or "").strip().split() if part])


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def is_generic_prefix(text: str) -> bool:
    norm = normalize_text(text)
    return any(norm == item or norm.startswith(item + " ") for item in GENERIC_PREFIXES)


def load_dataset_by_id(summary: Dict[str, Any], diag_dir: Path) -> Dict[str, Dict[str, Any]]:
    dataset_path_text = summary.get("dataset_path")
    if not dataset_path_text:
        return {}
    dataset_path = Path(dataset_path_text)
    if not dataset_path.is_absolute():
        dataset_path = (diag_dir / dataset_path).resolve()
    if not dataset_path.exists():
        return {}
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in data:
        if isinstance(row, dict) and "id" in row:
            by_id[str(row["id"])] = row
    return by_id


def first_target_token_text(target: str) -> str:
    parts = str(target or "").strip().split()
    return parts[0] if parts else ""


def target_word_count(target: str) -> int:
    return text_len(target)


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "values": []}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "values": list(lengths),
    }


def collect_events_by_sample(events: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_sample[as_int(event.get("sample_id"), -1)].append(event)
    return dict(by_sample)


def has_stop_like_early_generation(text: str, cached_decode_count: int) -> bool:
    # Existing diagnostics do not store EOS/stopping reasons. Treat only very
    # short decoded text with <=2 cached decode calls as stop-like evidence.
    return text_len(text) <= 2 and cached_decode_count <= 2


def make_sample_rows(
    per_sample: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    diag_dir: Path,
    events_by_sample: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    dataset_by_id = load_dataset_by_id(summary, diag_dir)
    rows: List[Dict[str, Any]] = []
    for raw in per_sample:
        sample_id = as_int(raw.get("sample_id"))
        record_id = str(raw.get("record_id", ""))
        dataset_row = dataset_by_id.get(record_id, {})
        prompt = dataset_row.get("src") or dataset_row.get("prompt") or dataset_row.get("question") or ""
        target = str(raw.get("target") or dataset_row.get("alt") or "")
        base_text = str(raw.get("base_free_text") or "")
        edited_text = str(raw.get("edited_free_text") or "")
        base_rank = as_int(raw.get("first_target_base_rank"))
        edited_rank = as_int(raw.get("first_target_edited_rank"))
        cached_decode_count = as_int(raw.get("cached_decode_hook_event_count"))
        rank_delta = edited_rank - base_rank if base_rank and edited_rank else 0
        rank_improved = bool(base_rank and edited_rank and edited_rank < base_rank)
        base_nll = as_float(raw.get("base_target_nll"))
        edited_nll = as_float(raw.get("edited_target_nll"))
        delta_nll = edited_nll - base_nll if math.isfinite(base_nll) and math.isfinite(edited_nll) else math.nan
        target_len = target_word_count(target)
        generated_len = text_len(edited_text)
        event_rows = events_by_sample.get(sample_id, [])
        hook_errors = sum(1 for event in event_rows if event.get("error"))
        max_event_residual = max(finite_values(as_float(event.get("residual_norm")) for event in event_rows), default=0.0)
        rows.append(
            {
                "sample_id": sample_id,
                "record_id": record_id,
                "prompt": prompt,
                "target": target,
                "target_word_count": target_len,
                "base_decoded_text": base_text,
                "edited_decoded_text": edited_text,
                "edited_equals_base": as_bool(raw.get("edited_equals_base")),
                "edited_contains_target": as_bool(raw.get("edited_contains_target")),
                "base_target_nll": base_nll,
                "edited_target_nll": edited_nll,
                "delta_nll": delta_nll,
                "nll_improved": math.isfinite(delta_nll) and delta_nll < 0,
                "first_target_token": first_target_token_text(target),
                "base_first_token_rank": base_rank,
                "edited_first_token_rank": edited_rank,
                "first_token_rank_delta": rank_delta,
                "first_token_rank_improved": rank_improved,
                "base_first_token_logprob": UNAVAILABLE,
                "edited_first_token_logprob": UNAVAILABLE,
                "base_top10_next_tokens": UNAVAILABLE,
                "edited_top10_next_tokens": UNAVAILABLE,
                "target_token_in_base_top10": base_rank > 0 and base_rank <= 10,
                "target_token_in_edited_top10": edited_rank > 0 and edited_rank <= 10,
                "target_token_in_base_top50": base_rank > 0 and base_rank <= 50,
                "target_token_in_edited_top50": edited_rank > 0 and edited_rank <= 50,
                "target_token_in_base_top100": base_rank > 0 and base_rank <= 100,
                "target_token_in_edited_top100": edited_rank > 0 and edited_rank <= 100,
                "mean_target_rank": edited_rank if edited_rank else "",
                "target_positions_improved": 1 if rank_improved else 0,
                "target_positions_top1": 1 if edited_rank == 1 else 0,
                "target_positions_observed": 1,
                "generated_length_words": generated_len,
                "cached_decode_hook_event_count": cached_decode_count,
                "early_stop_like": has_stop_like_early_generation(edited_text, cached_decode_count),
                "generic_prefix_like": is_generic_prefix(edited_text),
                "generation_residual_norm_mean": as_float(raw.get("generation_residual_norm_mean")),
                "active_logits_delta_norm": as_float(raw.get("active_logits_delta_norm")),
                "hook_entered": as_bool(raw.get("hook_entered")),
                "hook_error_count": as_int(raw.get("hook_error_count")) + hook_errors,
                "cached_decode_route_reused": as_bool(raw.get("cached_decode_route_reused")),
                "current_token_apply_mask_sum": raw.get("current_token_apply_mask_sum", ""),
                "residual_nonzero_by_step": raw.get("residual_nonzero_by_step", ""),
                "top_token_data_available": False,
            }
        )
    return rows


def classify_root_cause(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    if not rows:
        return "unknown", ["No diagnostic samples were available."], ["Rerun the diagnostic with per-sample outputs enabled."]

    n = len(rows)
    edited_equals_base_rate = sum(1 for row in rows if row["edited_equals_base"]) / n
    generic_prefix_rate = sum(1 for row in rows if row["generic_prefix_like"]) / n
    early_stop_count = sum(1 for row in rows if row["early_stop_like"])
    nll_improved_count = sum(1 for row in rows if row["nll_improved"])
    edited_top10_count = sum(1 for row in rows if row["target_token_in_edited_top10"])
    edited_top50_count = sum(1 for row in rows if row["target_token_in_edited_top50"])
    edited_top100_count = sum(1 for row in rows if row["target_token_in_edited_top100"])
    improved_ranks = sum(1 for row in rows if row["first_token_rank_improved"])
    rank_deltas = [abs(int(row["first_token_rank_delta"])) for row in rows if row["base_first_token_rank"]]
    base_ranks = [int(row["base_first_token_rank"]) for row in rows if row["base_first_token_rank"]]
    edited_ranks = [int(row["edited_first_token_rank"]) for row in rows if row["edited_first_token_rank"]]
    mean_abs_rank_delta = mean(rank_deltas) if rank_deltas else 0.0
    mean_edited_rank = mean(edited_ranks) if edited_ranks else math.inf
    mean_base_rank = mean(base_ranks) if base_ranks else math.inf

    reasons: List[str] = []
    recommendations: List[str] = []

    if early_stop_count >= math.ceil(0.6 * n):
        reasons.append("Most samples stop after 1-2 decoded words/events.")
        recommendations.append("Inspect generation config, eos token, and stopping criteria before any editing run.")
        return "possible_generation_stop_token_issue", reasons, recommendations

    if generic_prefix_rate >= 0.8 and edited_equals_base_rate >= 0.8:
        # Keep template as secondary unless ranks are near a useful range. In
        # this diagnostic, very poor target ranks are stronger evidence that
        # the target is not competitive under the next-token distribution.
        reasons.append("Most decoded answers are generic prompt-like prefixes and edited text equals base.")
        recommendations.append("Test the official LLaVA-Med conversation template and an explicit short-answer prompt.")

    if nll_improved_count >= math.ceil(0.6 * n) and edited_top50_count == 0 and mean_edited_rank > 100:
        reasons.append(
            "Teacher-forced NLL improves on most samples, but the first target token never enters top-50/top-100."
        )
        recommendations.append("Run a one-edit decoded overfit diagnostic before any 20-edit pilot.")
        recommendations.append("If the one-edit overfit still leaves target rank far from top-k, do not run 20-edit.")
        if generic_prefix_rate >= 0.8:
            recommendations.append("Also test a short-answer/template diagnostic, but treat weak target rank as the primary blocker.")
        return "weak_logit_shift", reasons, recommendations

    if mean_abs_rank_delta < max(100.0, 0.05 * mean_base_rank) and edited_top10_count == 0:
        reasons.append("Logits/residual effects are present, but target rank movement is small and never reaches top-10.")
        recommendations.append("Do not run 20-edit; inspect why residual updates are not moving task-relevant top tokens.")
        return "residual_not_affecting_top_tokens", reasons, recommendations

    if edited_top100_count > 0 and not any(row["edited_contains_target"] for row in rows):
        reasons.append("Some target tokens approach the candidate set, but exact decoded target matching remains absent.")
        recommendations.append("Try constrained short-answer decoding and alias/normalization-aware scoring.")
        return "target_too_long_or_exact_match_too_strict", reasons, recommendations

    if generic_prefix_rate >= 0.8:
        return "decoding_template_problem", reasons, recommendations

    reasons.append("No single failure mode is decisive from the saved diagnostic fields.")
    recommendations.append("Rerun a richer diagnostic that stores top-k next-token tables for base and edited logits.")
    return "unknown", reasons, recommendations


def build_summary(rows: Sequence[Dict[str, Any]], diag_summary: Dict[str, Any]) -> Dict[str, Any]:
    n = len(rows)
    base_nlls = finite_values(float(row["base_target_nll"]) for row in rows)
    edited_nlls = finite_values(float(row["edited_target_nll"]) for row in rows)
    delta_nlls = finite_values(float(row["delta_nll"]) for row in rows)
    generated_lengths = [int(row["generated_length_words"]) for row in rows]
    target_lengths = [int(row["target_word_count"]) for row in rows]
    first_top10_before = sum(1 for row in rows if row["target_token_in_base_top10"])
    first_top10_after = sum(1 for row in rows if row["target_token_in_edited_top10"])
    first_top50_before = sum(1 for row in rows if row["target_token_in_base_top50"])
    first_top50_after = sum(1 for row in rows if row["target_token_in_edited_top50"])
    root_cause, reasons, recommendations = classify_root_cause(rows, diag_summary)
    approve_20_edit = False
    return {
        "diag_dir": str(diag_summary.get("_diag_dir", "")),
        "num_samples": n,
        "edited_equals_base_rate": (sum(1 for row in rows if row["edited_equals_base"]) / n) if n else None,
        "target_contains_count": sum(1 for row in rows if row["edited_contains_target"]),
        "mean_base_nll": mean(base_nlls) if base_nlls else None,
        "mean_edited_nll": mean(edited_nlls) if edited_nlls else None,
        "mean_delta_nll": mean(delta_nlls) if delta_nlls else None,
        "teacher_forced_nll_improved_count": sum(1 for row in rows if row["nll_improved"]),
        "first_token_rank_improved_count": sum(1 for row in rows if row["first_token_rank_improved"]),
        "first_token_top10_count_before": first_top10_before,
        "first_token_top10_count_after": first_top10_after,
        "first_token_top50_count_before": first_top50_before,
        "first_token_top50_count_after": first_top50_after,
        "first_token_top100_count_before": sum(1 for row in rows if row["target_token_in_base_top100"]),
        "first_token_top100_count_after": sum(1 for row in rows if row["target_token_in_edited_top100"]),
        "mean_generated_length_words": mean(generated_lengths) if generated_lengths else None,
        "generated_length_summary": summarize_lengths(generated_lengths),
        "early_stop_count": sum(1 for row in rows if row["early_stop_like"]),
        "generic_prefix_count": sum(1 for row in rows if row["generic_prefix_like"]),
        "target_length_distribution_words": dict(Counter(target_lengths)),
        "target_length_summary_words": summarize_lengths(target_lengths),
        "top_token_data_available": False,
        "top_token_data_note": "Saved diagnostic does not contain top-10/top-k token identities or per-token logprobs.",
        "root_cause": root_cause,
        "root_cause_reasons": reasons,
        "recommendations": recommendations,
        "approved_for_20_edit": approve_20_edit,
        "next_command": next_command(root_cause),
    }


def next_command(root_cause: str) -> str:
    if root_cause == "weak_logit_shift":
        return (
            "CUDA_VISIBLE_DEVICES=1 /root/anaconda3/bin/python scripts/overfit_dsca_one_medmkeb_edit.py "
            "--model llava-med --hparams hparams/DSCA/llava_med.yaml "
            "--training-hparams hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml "
            "--dataset-path datasets/MedMKEB/eval.json --image-root datasets/MedMKEB/images "
            "--device cuda --output-dir outputs/dsca_medmkeb_llava_med_one_edit_overfit/$(date +%Y%m%d_%H%M%S)"
        )
    if root_cause == "decoding_template_problem":
        return (
            "CUDA_VISIBLE_DEVICES=1 /root/anaconda3/bin/python scripts/smoke_llava_med_official_loader.py "
            "--model-path /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b "
            "--vision-tower-path /remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336 "
            "--model-name llava-med-v1.5-mistral-7b --source-root third_party/LLaVA-Med "
            "--image-root datasets/MedMKEB/images --device cuda --dtype float16 "
            "--output outputs/medical_vlm_backbone_feasibility/llava_med_template_smoke.json"
        )
    if root_cause == "possible_generation_stop_token_issue":
        return "Inspect hparams/DSCA/llava_med.yaml generation config, eos token, and stopping criteria before rerunning diagnostics."
    if root_cause == "target_too_long_or_exact_match_too_strict":
        return "Run a constrained short-answer decoding diagnostic with alias/normalization-aware scoring before 20-edit."
    return "Rerun generation-path diagnostic with saved top-k token tables before 20-edit."


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "sample",
        "target",
        "base->edited NLL",
        "delta",
        "first-rank base->edited",
        "top50 after",
        "decoded",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        target = str(row["target"])
        if len(target) > 36:
            target = target[:33] + "..."
        decoded = str(row["edited_decoded_text"]).replace("|", "/")
        if len(decoded) > 42:
            decoded = decoded[:39] + "..."
        lines.append(
            "| {sample_id} | {target} | {base:.4f}->{edited:.4f} | {delta:.4f} | {br}->{er} | {top50} | {decoded} |".format(
                sample_id=row["sample_id"],
                target=target.replace("|", "/"),
                base=float(row["base_target_nll"]),
                edited=float(row["edited_target_nll"]),
                delta=float(row["delta_nll"]),
                br=row["base_first_token_rank"],
                er=row["edited_first_token_rank"],
                top50="yes" if row["target_token_in_edited_top50"] else "no",
                decoded=decoded,
            )
        )
    return "\n".join(lines)


def write_top_token_debug(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    parts = [
        "# Top Token Debug",
        "",
        "The saved generation-path diagnostic does not contain top-10 next-token identities, per-token logprobs, or full target-token rank tables.",
        "This analysis therefore reports top-k membership from the saved first-target-token rank only.",
        "",
    ]
    for row in rows:
        parts.extend(
            [
                f"## Sample {row['sample_id']} / record {row['record_id']}",
                "",
                f"- prompt: {row['prompt']}",
                f"- target: {row['target']}",
                f"- base decoded: {row['base_decoded_text']}",
                f"- edited decoded: {row['edited_decoded_text']}",
                f"- first target token: `{row['first_target_token']}`",
                f"- first target rank: {row['base_first_token_rank']} -> {row['edited_first_token_rank']}",
                f"- base top-10 next tokens: {UNAVAILABLE}",
                f"- edited top-10 next tokens: {UNAVAILABLE}",
                f"- target in edited top-10/top-50/top-100: {row['target_token_in_edited_top10']}/{row['target_token_in_edited_top50']}/{row['target_token_in_edited_top100']}",
                "",
            ]
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_failure_report(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    parts = [
        "# LLaVA-Med Generation Failure Analysis",
        "",
        f"- diagnostic dir: `{summary['diag_dir']}`",
        f"- samples: {summary['num_samples']}",
        f"- edited equals base rate: {summary['edited_equals_base_rate']}",
        f"- target contains count: {summary['target_contains_count']}",
        f"- mean target NLL: {summary['mean_base_nll']} -> {summary['mean_edited_nll']}",
        f"- mean delta NLL: {summary['mean_delta_nll']}",
        f"- first-token rank improved count: {summary['first_token_rank_improved_count']}",
        f"- first-token top10 before/after: {summary['first_token_top10_count_before']}/{summary['first_token_top10_count_after']}",
        f"- first-token top50 before/after: {summary['first_token_top50_count_before']}/{summary['first_token_top50_count_after']}",
        f"- generated length summary: {summary['generated_length_summary']}",
        f"- early stop count: {summary['early_stop_count']}",
        f"- generic prefix count: {summary['generic_prefix_count']}",
        f"- root cause: `{summary['root_cause']}`",
        f"- approved for 20-edit: {summary['approved_for_20_edit']}",
        "",
        "## Key Table",
        "",
        markdown_table(rows),
        "",
        "## Root Cause Evidence",
        "",
    ]
    for reason in summary["root_cause_reasons"]:
        parts.append(f"- {reason}")
    parts.extend(["", "## Recommendations", ""])
    for recommendation in summary["recommendations"]:
        parts.append(f"- {recommendation}")
    parts.extend(
        [
            "",
            "## Next Command",
            "",
            "```bash",
            summary["next_command"],
            "```",
            "",
            "## Limitations",
            "",
            f"- {summary['top_token_data_note']}",
            "- No model inference was run by this analysis script.",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    diag_dir = args.diag_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = [
        "generation_path_summary.json",
        "generation_path_per_sample.csv",
        "generation_path_debug.jsonl",
        "generation_path_report.md",
    ]
    missing = [name for name in required if not (diag_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing diagnostic files in {diag_dir}: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    diag_summary = read_json(diag_dir / "generation_path_summary.json")
    diag_summary["_diag_dir"] = str(diag_dir)
    per_sample = read_csv(diag_dir / "generation_path_per_sample.csv")
    debug_rows = read_jsonl(diag_dir / "generation_path_debug.jsonl")
    hook_events = read_jsonl(diag_dir / "generation_hook_events.jsonl")
    # Debug rows duplicate per-sample fields in current diagnostics. Keep them
    # loaded to validate presence and future-proof richer diagnostics.
    _ = debug_rows
    rows = make_sample_rows(per_sample, diag_summary, diag_dir, collect_events_by_sample(hook_events))
    summary = build_summary(rows, diag_summary)

    sample_fields = [
        "sample_id",
        "record_id",
        "prompt",
        "target",
        "base_decoded_text",
        "edited_decoded_text",
        "edited_equals_base",
        "base_target_nll",
        "edited_target_nll",
        "delta_nll",
        "first_target_token",
        "base_first_token_rank",
        "edited_first_token_rank",
        "first_token_rank_delta",
        "first_token_rank_improved",
        "target_token_in_base_top10",
        "target_token_in_edited_top10",
        "target_token_in_base_top50",
        "target_token_in_edited_top50",
        "target_token_in_base_top100",
        "target_token_in_edited_top100",
        "generation_residual_norm_mean",
        "active_logits_delta_norm",
    ]
    decoded_fields = [
        "sample_id",
        "record_id",
        "prompt",
        "target",
        "base_decoded_text",
        "edited_decoded_text",
        "edited_equals_base",
        "edited_contains_target",
        "generated_length_words",
        "early_stop_like",
        "generic_prefix_like",
        "cached_decode_hook_event_count",
    ]
    rank_fields = [
        "sample_id",
        "record_id",
        "target",
        "target_word_count",
        "first_target_token",
        "base_first_token_rank",
        "edited_first_token_rank",
        "first_token_rank_improved",
        "base_first_token_logprob",
        "edited_first_token_logprob",
        "mean_target_rank",
        "target_positions_observed",
        "target_positions_improved",
        "target_positions_top1",
        "base_top10_next_tokens",
        "edited_top10_next_tokens",
        "target_token_in_edited_top10",
        "target_token_in_edited_top50",
        "target_token_in_edited_top100",
        "top_token_data_available",
    ]
    write_csv(output_dir / "sample_rank_movement.csv", rows, sample_fields)
    write_csv(output_dir / "decoded_text_comparison.csv", rows, decoded_fields)
    write_csv(output_dir / "target_token_rank_table.csv", rows, rank_fields)
    write_top_token_debug(output_dir / "top_token_debug.md", rows)
    write_failure_report(output_dir / "generation_failure_report.md", rows, summary)
    with (output_dir / "generation_failure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
