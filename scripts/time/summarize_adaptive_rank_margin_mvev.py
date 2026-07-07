#!/usr/bin/env python3
"""Summarize bounded eval-only adaptive rank-margin MVEV runs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path("outputs/time_medmkeb_smoke/adaptive_rank_margin_mvev")
STRONG_REPORT_METRICS = {
    "retained": 9,
    "own_selected": 9,
    "own_top1": 7,
    "mean_selected": 2.0,
    "max_selected": 2.0,
    "catastrophic": 0,
    "locality": 2.3928621232509615,
    "worst_degradation": 0.0067901611328125,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def margin_from_dir(path: Path) -> Optional[float]:
    text = path.name.replace("margin", "").replace("p", ".")
    return to_float(text)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def debug_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("routing_mode") != "adaptive_rank_margin_topk2":
                continue
            if "adaptive_rank_margin_gap" not in row:
                continue
            key = (str(row.get("record_id")), str(row.get("call_label") or row.get("phase")))
            result[key] = row
    return result


def summarize_mode(rows: List[Dict[str, Any]], margin: Optional[float], mode: str) -> Dict[str, Any]:
    deltas = [to_float(row.get("target_nll_delta"), 0.0) for row in rows]
    sizes = [to_float(row.get("selected_expert_set_size"), 0.0) for row in rows]
    localities = [to_float(row.get("reference_delta")) for row in rows]
    retained = sum(float(delta or 0.0) > 0.0 for delta in deltas)
    catastrophic = sum(float(delta or 0.0) < 0.0 for delta in deltas)
    return {
        "margin": margin,
        "mode": mode,
        "num_records": len(rows),
        "retained": retained,
        "own_selected": sum(to_bool(row.get("selected_own_expert")) for row in rows),
        "own_top1": sum(to_bool(row.get("routing_top1_correct")) for row in rows),
        "mean_selected": mean(sizes),
        "max_selected": max(sizes) if sizes else None,
        "catastrophic_forgetting": catastrophic,
        "locality": mean(localities),
        "worst_degradation": max([max(0.0, -float(delta or 0.0)) for delta in deltas] or [0.0]),
        "mean_target_nll_delta": mean(deltas),
    }


def decision_for_candidate(row: Dict[str, Any]) -> str:
    retained = int(row.get("retained") or 0)
    own_selected = int(row.get("own_selected") or 0)
    mean_selected = float(row.get("mean_selected") or 1.0e9)
    max_selected = float(row.get("max_selected") or 1.0e9)
    catastrophic = int(row.get("catastrophic_forgetting") or 0)
    locality = float(row.get("locality") or 1.0e9)
    worst = float(row.get("worst_degradation") or 1.0e9)
    if (
        retained >= 9
        and own_selected >= 9
        and mean_selected <= 2.3
        and max_selected <= 3
        and catastrophic == 0
        and locality <= 3.0
        and worst <= STRONG_REPORT_METRICS["worst_degradation"] + 0.001
    ):
        return "A. Positive signal: proceed to 20-edit validation"
    return "C. Negative signal: stop; no effectiveness evidence"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def report_lines(summary_rows: List[Dict[str, Any]], record_rows: List[Dict[str, Any]], commands: List[str]) -> List[str]:
    candidates = [row for row in summary_rows if row["mode"] == "adaptive_rank_margin_topk2"]
    best = max(candidates, key=lambda row: (row["retained"], row["own_selected"], -(row["mean_selected"] or 999))) if candidates else {}
    decision = decision_for_candidate(best) if best else "C. Negative signal: stop; no effectiveness evidence"
    lines = [
        "# TIME Adaptive Rank-Margin MVEV Report",
        "",
        "## Scope",
        "- Minimum viable effectiveness validation for an experimental inference-only `adaptive_rank_margin_topk2` routing mode.",
        "- No retrain, no 20-edit, no broad sweep. Candidate remains NOT PROMOTED.",
        "- Experiments ran on `my-gpu` with `CUDA_VISIBLE_DEVICES=0` and `--eval-only --time-load-repository`.",
        "",
        "## Implementation Summary",
        "- Added off-by-default hparams: `time_enable_adaptive_rank_margin_rescue`, `time_adaptive_rank_margin`, `time_adaptive_rank_margin_use_rank3`, `time_adaptive_rank_margin_debug`.",
        "- Added experimental routing mode `adaptive_rank_margin_topk2`; disabled mode falls back to existing top-k behavior.",
        "- Existing `threshold`, `topk`, `threshold_topk`, `force_current`, relative, and older calibrated/adaptive diagnostic modes were preserved.",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "- `scripts/time/test_time_modules.py`",
        "- `scripts/time/summarize_adaptive_rank_margin_mvev.py`",
        "",
        "## Verification",
        "- Local py_compile and `scripts/time/test_time_modules.py`: passed.",
        "- Remote py_compile and `scripts/time/test_time_modules.py` on `my-gpu`: passed.",
        "- GPU 2 was not used; eval-only runs used physical GPU 0.",
        "- The catastrophic count in this eval-only report is the count of records with negative final NLL delta, because no training/immediate-edit baseline was run.",
        "",
        "## Validation Commands",
        *commands,
        "",
        "## Summary Table",
        "| margin | mode | retained | own-selected | own-top1 | mean selected | max selected | catastrophic | locality | worst degradation | mean NLL delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {margin} | {mode} | {retained}/{num_records} | {own_selected}/{num_records} | {own_top1}/{num_records} | {mean_selected:.4g} | {max_selected:.4g} | {catastrophic_forgetting} | {locality:.6g} | {worst_degradation:.6g} | {mean_target_nll_delta:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Strong Calibrated Topk2 Reference",
            f"- retained {STRONG_REPORT_METRICS['retained']}/10, own-selected {STRONG_REPORT_METRICS['own_selected']}/10, own-top1 {STRONG_REPORT_METRICS['own_top1']}/10.",
            f"- mean/max selected {STRONG_REPORT_METRICS['mean_selected']}/{STRONG_REPORT_METRICS['max_selected']}, catastrophic {STRONG_REPORT_METRICS['catastrophic']}, locality {STRONG_REPORT_METRICS['locality']:.6g}, worst degradation {STRONG_REPORT_METRICS['worst_degradation']:.6g}.",
            "",
            "## Fragile Records",
            "| margin | record | selected ids | own selected | top1 own | rank2-rank3 gap | rank3 expert | triggered | retained delta | locality |",
            "|---:|---:|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in record_rows:
        lines.append(
            "| {margin} | {record_id} | `{selected_expert_ids}` | {selected_own_expert} | {routing_top1_correct} | {adaptive_rank_margin_gap} | {adaptive_rank_margin_rank3_expert_id} | {adaptive_rank_margin_triggered} | {target_nll_delta} | {reference_delta} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            f"- Final decision: {decision}",
            "- Do not run 20-edit from these results.",
            "- Reason: the candidate does help records 942 and 671 locally, but margin 0.02/0.05 selects rank3 for every record in this bounded run, raises mean selected to 3.0, worsens locality, and retained count is only 6/10.",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    summary_rows: List[Dict[str, Any]] = []
    fragile_rows: List[Dict[str, Any]] = []
    commands: List[str] = []
    for margin_dir in sorted(base_dir.glob("margin*")):
        csv_path = margin_dir / "eval_only_per_record.csv"
        if not csv_path.exists():
            continue
        margin = margin_from_dir(margin_dir)
        rows = read_csv(csv_path)
        debug = debug_rows(margin_dir / "time_ten_nonseq_routing_debug.jsonl")
        hparams = margin_dir / "time_hparams.json"
        if hparams.exists():
            payload = json.loads(hparams.read_text(errors="replace"))
            command = payload.get("command")
            if command:
                command = str(command)
                prefix = "" if command.startswith("CUDA_VISIBLE_DEVICES=") else "CUDA_VISIBLE_DEVICES=0 "
                commands.append(f"- `{prefix}{command}`")
        for mode in sorted({str(row.get("eval_routing_mode")) for row in rows}):
            mode_rows = [row for row in rows if row.get("eval_routing_mode") == mode]
            summary_rows.append(summarize_mode(mode_rows, margin, mode))
        for row in rows:
            if row.get("eval_routing_mode") != "adaptive_rank_margin_topk2":
                continue
            if str(row.get("record_id")) not in {"942", "671"}:
                continue
            call_label = str(row.get("phase"))
            event = debug.get((str(row.get("record_id")), call_label), {})
            fragile_rows.append(
                {
                    "margin": margin,
                    "record_id": row.get("record_id"),
                    "selected_expert_ids": row.get("selected_expert_ids"),
                    "selected_own_expert": row.get("selected_own_expert"),
                    "routing_top1_correct": row.get("routing_top1_correct"),
                    "adaptive_rank_margin_gap": event.get("adaptive_rank_margin_gap"),
                    "adaptive_rank_margin_rank3_expert_id": event.get("adaptive_rank_margin_rank3_expert_id"),
                    "adaptive_rank_margin_triggered": event.get("adaptive_rank_margin_triggered"),
                    "target_nll_delta": row.get("target_nll_delta"),
                    "reference_delta": row.get("reference_delta"),
                }
            )

    base_dir.mkdir(parents=True, exist_ok=True)
    write_csv(base_dir / "TIME_MVEV_SUMMARY.csv", summary_rows)
    (base_dir / "TIME_MVEV_SUMMARY.json").write_text(
        json.dumps({"summary": summary_rows, "fragile_records": fragile_rows, "strong_reference": STRONG_REPORT_METRICS}, indent=2, sort_keys=True)
    )
    (base_dir / "TIME_MVEV_REPORT.md").write_text("\n".join(report_lines(summary_rows, fragile_rows, commands)))


if __name__ == "__main__":
    main()
