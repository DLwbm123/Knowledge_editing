#!/usr/bin/env python3
"""Build a DSCA MedMKEB failure triage report from diagnostic artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(errors="replace"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def classify(
    failure: Dict[str, Any],
    label: Dict[str, Any],
    nll: Dict[str, Any],
    routing: Dict[str, Any],
    overfit: Dict[str, Any],
) -> List[str]:
    classes: List[str] = []
    rows = int(failure.get("prediction_rows") or 0)
    edited_contains = int(failure.get("number_edited_contains_target") or 0)
    base_equals = int(failure.get("number_base_equals_edited") or 0)
    if rows and edited_contains == 0:
        classes.append("real modeling failure, not an evaluation normalization artifact")
    elif rows and edited_contains > 0:
        classes.append("possible evaluation normalization problem")
    if label and not bool(label.get("pass")):
        classes.append("target label/mask problem")
    if rows and base_equals / max(rows, 1) > 0.8:
        classes.append("generation path may be unchanged or residual/routing ineffective")
    route_rate = routing.get("assigned_cluster_in_candidates_rate")
    active_rate = routing.get("active_dsam_available_rate")
    if route_rate is not None and float(route_rate) < 0.5:
        classes.append("routing failure")
    if active_rate is not None and float(active_rate) < 0.8:
        classes.append("insufficient active DSAMs due to min_samples/clustering")
    if nll.get("num_improved_target_nll") is not None and int(nll.get("num_improved_target_nll") or 0) == 0:
        classes.append("target task optimization did not improve target NLL")
    if int(nll.get("num_residual_has_no_logits_effect") or 0) > 0:
        classes.append("DSAM residual layer/position ineffective for some samples")
    if overfit:
        final_nll = overfit.get("final_target_nll")
        teacher_forced_success = final_nll is not None and float(final_nll) < 1.0e-2
        if bool(overfit.get("success")):
            classes.append("insufficient optimization steps/routing in multi-edit run")
        elif teacher_forced_success:
            classes.append("teacher-forced single-edit overfit succeeds but decoded prediction remains wrong")
            classes.append("generation/evaluation decode path not reflecting edited target logits")
        else:
            classes.append("single-edit task-only force-route overfit failure")
            classes.append("possible BLIP2 medical-domain limitation or implementation path issue")
    return list(dict.fromkeys(classes))


def recommend(classes: List[str]) -> str:
    if "target label/mask problem" in classes:
        return "Fix MedMKEB target label/answer_mask construction, then rerun the 20-edit pilot unchanged."
    if "generation/evaluation decode path not reflecting edited target logits" in classes:
        return "Run a generation-path diagnostic that compares edited free generation, full-sequence argmax decoding, and label-window argmax on the existing 20-edit repository before changing DSCA hyperparameters."
    if "routing failure" in classes or "insufficient active DSAMs due to min_samples/clustering" in classes:
        return "Run a 20-edit diagnostic with min_samples=1 and route-threshold logging enabled, keeping DSCA formulas unchanged."
    if "target task optimization did not improve target NLL" in classes or "single-edit task-only force-route overfit failure" in classes:
        return "Run a one-edit force-route task-only sweep over learning rate {1e-4, 1e-3, 3e-3} before any 20-edit retry."
    if "possible evaluation normalization problem" in classes:
        return "Patch metric normalization to use exact normalized and alias-aware contains fields, then rescore the existing run only."
    return "Run one 20-edit retry with the same data and added NLL/routing logging to confirm the diagnosis."


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    failure = read_json(out / "failure_summary.json")
    label = read_json(out / "label_masks" / "label_mask_summary.json")
    nll = read_json(out / "target_nll" / "target_nll_summary.json")
    routing = read_json(out / "routing_trace" / "routing_trace_summary.json")
    overfit_paths = sorted(out.glob("one_edit_overfit_sample*/overfit_summary.json"))
    overfit_summaries = [read_json(path) for path in overfit_paths]
    overfit = overfit_summaries[0] if overfit_summaries else {}
    rows = read_csv(out / "failure_case_table.csv")
    classifications = classify(failure, label, nll, routing, overfit)
    recommendation = recommend(classifications)

    prediction_rows = int(failure.get("prediction_rows") or 0)
    edited_equals_rate = (
        (int(failure.get("number_base_equals_edited") or 0) / prediction_rows)
        if prediction_rows
        else None
    )

    report_lines = [
        "# DSCA MedMKEB Failure Triage Report",
        "",
        f"- output directory: `{out}`",
        f"- analyzed prediction rows: {prediction_rows}",
        f"- available sample types: `{failure.get('sample_types_available', {})}`",
        "",
        "## 1. Metric Integrity",
        "",
        f"- exact/contains fields missing counts: `{failure.get('exact_fields_missing_from_predictions_jsonl', {})}`",
        f"- edited==base rate: {pct(edited_equals_rate)} ({failure.get('number_base_equals_edited', 0)}/{prediction_rows})",
        f"- edited predictions containing target: {failure.get('number_edited_contains_target', 0)}",
        f"- base predictions containing target: {failure.get('number_base_contains_target', 0)}",
        f"- missing or empty edited predictions: {failure.get('number_empty_edited_predictions', 0)}",
        "",
        "## 2. Label/Mask Integrity",
        "",
        f"- pass: {label.get('pass') if label else 'n/a'}",
        f"- checked samples: {label.get('num_checked') if label else 'n/a'}",
        f"- min target token count: {label.get('min_target_token_count') if label else 'n/a'}",
        f"- min labels != -100 count: {label.get('min_labels_not_ignore_count') if label else 'n/a'}",
        f"- min answer_mask sum: {label.get('min_answer_mask_sum') if label else 'n/a'}",
        f"- labels align with answer_mask tail: {label.get('all_labels_align_answer_mask') if label else 'n/a'}",
        f"- failures: `{label.get('failures', []) if label else []}`",
        "",
        "## 3. NLL Diagnosis",
        "",
        f"- mean base target NLL: {num(nll.get('mean_base_target_nll'))}",
        f"- mean edited target NLL: {num(nll.get('mean_edited_target_nll'))}",
        f"- mean delta NLL: {num(nll.get('mean_delta_nll'))}",
        f"- samples with improved target NLL: {nll.get('num_improved_target_nll')}",
        f"- route-missing samples: {nll.get('num_route_missing')}",
        f"- residual has no logits effect samples: {nll.get('num_residual_has_no_logits_effect')}",
        "",
        "## 4. Routing Diagnosis",
        "",
        f"- assigned cluster in candidates rate: {pct(routing.get('assigned_cluster_in_candidates_rate'))}",
        f"- active DSAM availability rate: {pct(routing.get('active_dsam_available_rate'))}",
        f"- mean route weight to assigned cluster: {num(routing.get('mean_route_weight_assigned_cluster'))}",
        f"- candidate count distribution: `{routing.get('candidate_count_distribution', {})}`",
        f"- final active DSAM ids: `{routing.get('active_dsam_ids_final', [])}`",
        "",
        "## 5. One-Edit Overfit",
        "",
        f"- decoded success: {overfit.get('success') if overfit else 'n/a'}",
        f"- first decoded success step: {overfit.get('first_success_step') if overfit else 'n/a'}",
        f"- sample0 final target NLL: {num(overfit.get('final_target_nll')) if overfit else 'n/a'}",
        f"- sample0 final prediction: `{overfit.get('final_prediction') if overfit else 'n/a'}`",
        f"- all one-edit final target NLLs: `{[item.get('final_target_nll') for item in overfit_summaries]}`",
        f"- teacher-forced overfit successes at NLL < 0.01: {sum(1 for item in overfit_summaries if item.get('final_target_nll') is not None and float(item['final_target_nll']) < 1.0e-2)}/{len(overfit_summaries)}",
        "",
        "## 6. Root-Cause Classification",
        "",
    ]
    report_lines.extend([f"- {item}" for item in classifications] or ["- inconclusive"])
    report_lines.extend(
        [
            "",
            "## 7. Recommended Next Experiment",
            "",
            recommendation,
            "",
            "## 8. Representative Failure Rows",
            "",
        ]
    )
    for row in rows[:5]:
        report_lines.extend(
            [
                f"- step {row.get('step')} `{row.get('sample_type')}` target `{row.get('target')}`",
                f"  base: `{row.get('base_prediction')}`",
                f"  edited: `{row.get('edited_prediction')}`",
            ]
        )

    (out / "failure_triage_report.md").write_text("\n".join(report_lines) + "\n")
    summary = {
        "output_dir": str(out),
        "metric_artifact_or_real": "metric_artifact_possible"
        if "possible evaluation normalization problem" in classifications
        else "real_failure",
        "edited_equals_base_rate": edited_equals_rate,
        "edited_contains_target_count": failure.get("number_edited_contains_target"),
        "label_mask_pass": label.get("pass") if label else None,
        "mean_base_target_nll": nll.get("mean_base_target_nll"),
        "mean_edited_target_nll": nll.get("mean_edited_target_nll"),
        "num_improved_target_nll": nll.get("num_improved_target_nll"),
        "assigned_cluster_routed_rate": routing.get("assigned_cluster_in_candidates_rate"),
        "active_dsam_available_rate": routing.get("active_dsam_available_rate"),
        "one_edit_overfit_success": overfit.get("success") if overfit else None,
        "one_edit_teacher_forced_overfit_success_count": sum(
            1
            for item in overfit_summaries
            if item.get("final_target_nll") is not None and float(item["final_target_nll"]) < 1.0e-2
        ),
        "one_edit_overfit_runs": len(overfit_summaries),
        "root_cause_classification": classifications,
        "recommended_next_run": recommendation,
    }
    (out / "triage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
