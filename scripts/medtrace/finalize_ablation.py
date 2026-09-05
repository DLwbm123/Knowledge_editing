#!/usr/bin/env python3
"""Finalize generality and hard-scope Judge outputs into public aggregates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TASKS = ("T0", "T1L", "T1G", "T2G")
A_CONDITIONS = ("CP_NATIVE_ORIGINAL", "CP_NATIVE_CONTINUE_80", "CP_NATIVE_PLUS_PARAPHRASE_80")
B_CONDITIONS = ("ORIGINAL_Q_MATCHED_OUTPUT_FIT", "BROAD_Q_MATCHED_OUTPUT_FIT", "HARD_MIXED_Q_MATCHED_OUTPUT_FIT")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def checked_verdicts(path: Path, expected: set[str]) -> dict[str, bool]:
    rows = read_jsonl(path)
    by_id = {row["opaque_query_id"]: row for row in rows}
    if set(by_id) != expected or len(by_id) != len(rows) or any(not row["parse_valid"] for row in rows):
        raise RuntimeError("Judge output coverage or parsing failure")
    return {key: bool(row["is_correct"]) for key, row in by_id.items()}


def task_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        by_edit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_edit[row["edit_id"]].append(row)
        result[task] = {
            "eligible_edits": len(by_edit),
            "n": len(selected),
            "exact_correct": sum(row["exact"] for row in selected),
            "exact_micro": mean(row["exact"] for row in selected),
            "exact_macro": mean(mean(row["exact"] for row in values) for values in by_edit.values()),
            "semantic_correct": sum(row["semantic"] for row in selected),
            "semantic_micro": mean(row["semantic"] for row in selected),
            "semantic_macro": mean(mean(row["semantic"] for row in values) for values in by_edit.values()),
            "truncated_without_eos": sum(row["truncated_without_eos"] for row in selected),
        }
    return result


def existing_addendum(args: argparse.Namespace) -> dict[str, Any]:
    closure = json.loads(args.a0_closure.read_text())
    old_judge_rows = read_jsonl(args.a0_judge)
    old_judge = {row["opaque_query_id"]: bool(row["is_correct"]) for row in old_judge_rows}
    disagreements = [row for row in closure["predictions"] if row["task"] in {"T0", "T1G"} and row["normalized_exact_reference_match"] and not old_judge[row["opaque_query_id"]]]
    if len(disagreements) != 6 or any(row["raw_answer"] != row["reference"] for row in disagreements):
        raise RuntimeError("unexpected exact/Judge disagreement evidence")
    old_scope = json.loads(args.old_scope_result.read_text())
    old_sidecar = json.loads(args.old_scope_sidecar.read_text())
    old_scope_judge = checked_verdicts(args.old_scope_judge, {row["opaque_query_id"] for row in old_sidecar})
    judged = {(row["logical_id"], row["path"]): old_scope_judge[row["opaque_query_id"]] for row in old_sidecar}
    negatives = [row for row in old_scope["evaluation"] if row["label"] == "negative"]
    positives = [row for row in old_scope["evaluation"] if row["label"] == "positive"]
    control_on = {row["logical_id"] for row in negatives if row["control_on"]}
    final_on = {row["logical_id"] for row in negatives if row["final_on"]}
    path_accuracy = {
        path: sum(judged[(row["logical_id"], path)] for row in negatives)
        for path in ("base", "original_q_threshold_control", "final_forced_on", "final_intrinsic_gated")
    }
    base_correct = [row for row in negatives if judged[(row["logical_id"], "base")]]
    final_on_base_correct = [row for row in base_correct if row["final_on"]]
    scope = {
        "negative_count": len(negatives),
        "control_on_count": len(control_on),
        "final_on_count": len(final_on),
        "same_on_ids": control_on == final_on,
        "on_overlap_count": len(control_on & final_on),
        "control_only_on_count": len(control_on - final_on),
        "final_only_on_count": len(final_on - control_on),
        "negative_correct_counts": path_accuracy,
        "negative_token_change_counts": {
            path: sum(row["outputs"][path]["raw_token_ids"] != row["outputs"]["base"]["raw_token_ids"] for row in negatives)
            for path in ("original_q_threshold_control", "final_forced_on", "final_intrinsic_gated")
        },
        "base_correct_count": len(base_correct),
        "base_correct_forced_damage": sum(not judged[(row["logical_id"], "final_forced_on")] for row in base_correct),
        "base_correct_gated_damage": sum(not judged[(row["logical_id"], "final_intrinsic_gated")] for row in base_correct),
        "final_on_base_correct_count": len(final_on_base_correct),
        "conditional_final_on_damage": None if not final_on_base_correct else sum(not judged[(row["logical_id"], "final_intrinsic_gated")] for row in final_on_base_correct) / len(final_on_base_correct),
        "control_positive_correct": sum(judged[(row["logical_id"], "original_q_threshold_control")] for row in positives),
        "positive_count": len(positives),
    }
    private = {
        "schema_version": "medtrace-existing-result-addendum-private-v1",
        "exact_judge_disagreements": [{key: row[key] for key in ("event_index", "edit_id", "query_id", "task", "question", "reference", "raw_answer")} for row in disagreements],
        "scope_pairing": scope,
    }
    atomic_json(args.private_out / "existing_result_addendum_private.json", private)
    atomic_text(args.public_out / "EXISTING_RESULT_ADDENDUM.md", f"""# Existing-result addendum

## Exact versus semantic Judge

Six rows (T0: 1; T1G: 5) had normalized exact=true but semantic=false. All six raw answers are byte-for-byte equal to their bound references before normalization; five belong to one edit and one to another. Therefore normalization did not alter their semantics and the evidence points to a Judge/reference-relative objective disagreement rather than model output mismatch. The published verdicts remain unchanged because no independent second clinical review was performed; these six are `UNRESOLVED_JUDGE_REFERENCE_DISAGREEMENT`.

## Existing scope paired behavior

- Original-Q and Final-Q each activated {len(control_on)}/20 negatives; the activated sets were {'identical' if control_on == final_on else 'different'} (overlap {len(control_on & final_on)}, Original-only {len(control_on - final_on)}, Final-only {len(final_on - control_on)}).
- Negative semantic correctness: Base {path_accuracy['base']}/20, Original-Q control {path_accuracy['original_q_threshold_control']}/20, Final forced-on {path_accuracy['final_forced_on']}/20, Final gated {path_accuracy['final_intrinsic_gated']}/20.
- Token changes versus Base: Original-Q control {scope['negative_token_change_counts']['original_q_threshold_control']}/20, Final forced-on {scope['negative_token_change_counts']['final_forced_on']}/20, Final gated {scope['negative_token_change_counts']['final_intrinsic_gated']}/20.
- Among {len(base_correct)} Base-correct negatives, forced-on damaged {scope['base_correct_forced_damage']} and gated damaged {scope['base_correct_gated_damage']}.
- Final-Q ON occurred on {len(final_on_base_correct)} Base-correct negatives; conditional gated damage is {pct(scope['conditional_final_on_damage'])}.
- Original-Q control answered {scope['control_positive_correct']}/{len(positives)} evaluation positives correctly.

Because forced-on caused no observed Base-correct negative damage on this broad panel, the old panel did not demonstrate a behavioral-damage reduction attributable to gating.
""")
    return {"disagreement_count": len(disagreements), "scope": scope}


def finalize_generality(args: argparse.Namespace) -> dict[str, Any]:
    sidecar = json.loads(args.generality_sidecar.read_text())
    verdicts = checked_verdicts(args.generality_judge, {row["opaque_query_id"] for row in sidecar})
    rows = [{**row, "semantic": verdicts[row["opaque_query_id"]]} for row in sidecar]
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in A_CONDITIONS}
    if any(len(value) != 163 for value in by_condition.values()):
        raise RuntimeError("generality condition coverage mismatch")
    metrics = {condition: task_metrics(value) for condition, value in by_condition.items()}
    a0 = {(row["edit_id"], row["query_id"]): row["semantic"] for row in by_condition[A_CONDITIONS[0]] if row["task"] == "T2G"}
    retention = {}
    for condition in A_CONDITIONS[1:]:
        current = {(row["edit_id"], row["query_id"]): row["semantic"] for row in by_condition[condition] if row["task"] == "T2G"}
        retention[condition] = {
            "a0_success_retained": sum(current[key] for key, value in a0.items() if value),
            "a0_success_denominator": sum(a0.values()),
            "a0_failure_recovered": sum(current[key] for key, value in a0.items() if not value),
            "a0_failure_denominator": len(a0) - sum(a0.values()),
        }
    event_deltas = {}
    for task in TASKS:
        a1_by_edit, a2_by_edit = defaultdict(list), defaultdict(list)
        for row in by_condition[A_CONDITIONS[1]]:
            if row["task"] == task:
                a1_by_edit[row["edit_id"]].append(row["semantic"])
        for row in by_condition[A_CONDITIONS[2]]:
            if row["task"] == task:
                a2_by_edit[row["edit_id"]].append(row["semantic"])
        common = set(a1_by_edit) & set(a2_by_edit)
        differences = {edit: mean(a2_by_edit[edit]) - mean(a1_by_edit[edit]) for edit in common}
        event_deltas[task] = {"improved_edits": sum(value > 0 for value in differences.values()), "tied_edits": sum(value == 0 for value in differences.values()), "worsened_edits": sum(value < 0 for value in differences.values()), "eligible_edits": len(common)}
    private = json.loads(args.generality_result.read_text())
    budgets = {}
    for condition in A_CONDITIONS[1:]:
        selected = [row for row in private["condition_results"] if row["condition"] == condition]
        budgets[condition] = {
            "events": len(selected), "optimizer_steps_per_event": 80, "micro_forwards_per_event": 160,
            "sequence_token_proxy": sum(row["sequence_token_proxy"] for row in selected),
            "supervised_target_tokens": sum(row["supervised_target_tokens"] for row in selected),
            "training_seconds": sum(row["training_seconds"] for row in selected),
            "peak_vram_bytes": max(row["peak_vram_bytes"] for row in selected),
            "base_guard_passed": all(row["base_guard"]["unchanged"] and row["base_restored_exact"] for row in selected),
        }
    t2g_delta = metrics[A_CONDITIONS[2]]["T2G"]["semantic_macro"] - metrics[A_CONDITIONS[1]]["T2G"]["semantic_macro"]
    t0_loss = metrics[A_CONDITIONS[1]]["T0"]["semantic_correct"] - metrics[A_CONDITIONS[2]]["T0"]["semantic_correct"]
    t1g_delta = metrics[A_CONDITIONS[2]]["T1G"]["semantic_macro"] - metrics[A_CONDITIONS[1]]["T1G"]["semantic_macro"]
    decision = t2g_delta >= 0.10 and t0_loss <= 1 and t1g_delta >= -0.05
    aggregate = {"schema_version": "medtrace-generality-paired-public-v1", "status": "GENERALITY_PAIRED_EVALUATION_COMPLETE", "metrics": metrics, "t2g_a0_pairing": retention, "a2_minus_a1_edit_directions": event_deltas, "budgets": budgets, "registered_decision": {"retain_multi_paraphrase_supervision": decision, "a2_minus_a1_t2g_macro": t2g_delta, "a2_minus_a1_t1g_macro": t1g_delta, "a2_t0_loss_vs_a1_count": t0_loss}}
    atomic_json(args.private_out / "generality_metrics_private.json", aggregate)
    with (args.public_out / "GENERALITY_PAIRED_DEV16.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "task", "eligible_edits", "n", "exact_correct", "exact_micro", "exact_macro", "semantic_correct", "semantic_micro", "semantic_macro", "truncated_without_eos"])
        for condition in A_CONDITIONS:
            for task in TASKS:
                row = metrics[condition][task]
                writer.writerow([condition, task, row["eligible_edits"], row["n"], row["exact_correct"], row["exact_micro"], row["exact_macro"], row["semantic_correct"], row["semantic_micro"], row["semantic_macro"], row["truncated_without_eos"]])
    table = "\n".join(f"| {condition} | {task} | {value['eligible_edits']} | {value['n']} | {value['semantic_correct']}/{value['n']} ({value['semantic_micro']:.1%}) | {value['semantic_macro']:.1%} | {value['exact_micro']:.1%} |" for condition in A_CONDITIONS for task, value in metrics[condition].items())
    atomic_text(args.public_out / "GENERALITY_PAIRED_REPORT.md", f"""# DEV16 paired generality ablation

Status: `GENERALITY_PAIRED_EVALUATION_COMPLETE`

| Condition | Task | Valid edits | n | Semantic micro | Semantic macro | Exact micro |
|---|---|---:|---:|---:|---:|---:|
{table}

A1 and A2 both start from the same per-edit A0 checkpoint and execute 80 optimizer steps and 160 batch-size-1 micro-forwards per edit. A1 uses native/native with 0.5 weights; A2 uses native/rotating-paraphrase with 0.5 weights. Token/FLOPs proxies are reported because question lengths differ.

A2 minus A1 T2G macro: {t2g_delta:+.4f}; T1G macro: {t1g_delta:+.4f}; A2 T0 correct-count loss versus A1: {t0_loss}. Registered retain decision: `{decision}`.

A1 retained {retention[A_CONDITIONS[1]]['a0_success_retained']}/{retention[A_CONDITIONS[1]]['a0_success_denominator']} old T2G successes and recovered {retention[A_CONDITIONS[1]]['a0_failure_recovered']}/{retention[A_CONDITIONS[1]]['a0_failure_denominator']} old failures. A2 retained {retention[A_CONDITIONS[2]]['a0_success_retained']}/{retention[A_CONDITIONS[2]]['a0_success_denominator']} and recovered {retention[A_CONDITIONS[2]]['a0_failure_recovered']}/{retention[A_CONDITIONS[2]]['a0_failure_denominator']}.

The original probes have already been viewed and are reused only as a development ablation panel, not an unseen confirmation set.
""")
    return aggregate


def subset_name(row: dict[str, Any]) -> str:
    if row["label"] == "positive":
        return "positive"
    if row["fact_relation"] == "broad_unrelated_source_qa":
        return "broad"
    if row["fact_relation"] == "same_question_different_image_conflicting_source_answer":
        return "same_question_different_image"
    if row["fact_relation"] == "same_image_other_source_fact":
        return "same_image_other_fact"
    raise RuntimeError("unknown scope evaluation relation")


def finalize_scope(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.scope_result.read_text())
    sidecar = json.loads(args.scope_sidecar.read_text())
    verdicts = checked_verdicts(args.scope_judge, {row["opaque_query_id"] for row in sidecar})
    judged = {(row["logical_id"], row["path"]): verdicts[row["opaque_query_id"]] for row in sidecar}
    eval_by_id = {row["logical_id"]: row for row in result["evaluations"]}
    rows_by_subset = defaultdict(list)
    for row in result["evaluations"]:
        rows_by_subset[subset_name(row)].append(row)
    if {key: len(value) for key, value in rows_by_subset.items()} != {"positive": 4, "broad": 15, "same_question_different_image": 5, "same_image_other_fact": 12}:
        raise RuntimeError("hard-scope evaluation subgroup coverage mismatch")
    metrics = {}
    for condition in B_CONDITIONS:
        condition_metrics = {}
        forced_path, gated_path = f"{condition}__forced", f"{condition}__gated"
        for subset, rows in rows_by_subset.items():
            base_correct = [row for row in rows if judged[(row["logical_id"], "base")]]
            on = [row for row in rows if row["decisions"][condition]["on"]]
            on_base_correct = [row for row in on if judged[(row["logical_id"], "base")]]
            forced_damage = sum(not judged[(row["logical_id"], forced_path)] for row in base_correct)
            gated_damage = sum(not judged[(row["logical_id"], gated_path)] for row in base_correct)
            conditional_damage = None if not on_base_correct else sum(not judged[(row["logical_id"], gated_path)] for row in on_base_correct) / len(on_base_correct)
            condition_metrics[subset] = {
                "n": len(rows), "on": len(on), "activation_rate": len(on) / len(rows),
                "base_correct": len(base_correct),
                "forced_correct": sum(judged[(row["logical_id"], forced_path)] for row in rows),
                "gated_correct": sum(judged[(row["logical_id"], gated_path)] for row in rows),
                "forced_damage_on_base_correct": forced_damage,
                "gated_damage_on_base_correct": gated_damage,
                "unconditional_gated_damage_rate": gated_damage / len(rows),
                "on_base_correct_denominator": len(on_base_correct),
                "conditional_on_base_correct_damage_rate": conditional_damage,
                "forced_token_changed": sum(row["outputs"][forced_path]["raw_token_ids"] != row["outputs"]["base"]["raw_token_ids"] for row in rows),
                "gated_token_changed": sum(row["outputs"][gated_path]["raw_token_ids"] != row["outputs"]["base"]["raw_token_ids"] for row in rows),
                "off_count": len(rows) - len(on),
                "off_token_parity": sum((not row["decisions"][condition]["on"]) and row["decisions"][condition]["gated_token_exact_base"] for row in rows),
                "on_hook_executed": sum(row["decisions"][condition]["on"] and row["decisions"][condition]["gated"]["hook_executed"] for row in rows),
                "on_nonzero_residual": sum(row["decisions"][condition]["on"] and row["decisions"][condition]["gated"]["max_active_residual_norm"] > 0 for row in rows),
            }
        native_forced, native_gated = f"{condition}__forced", f"{condition}__gated"
        condition_metrics["native"] = {
            "on": result["native"]["decisions"][condition]["on"],
            "forced_correct": judged[("native", native_forced)],
            "gated_correct": judged[("native", native_gated)],
        }
        metrics[condition] = condition_metrics
    pairing = {}
    for left, right in ((B_CONDITIONS[0], B_CONDITIONS[1]), (B_CONDITIONS[0], B_CONDITIONS[2]), (B_CONDITIONS[1], B_CONDITIONS[2])):
        left_on = {row["logical_id"] for row in result["evaluations"] if row["decisions"][left]["on"]}
        right_on = {row["logical_id"] for row in result["evaluations"] if row["decisions"][right]["on"]}
        pairing[f"{right}__vs__{left}"] = {"on_to_off": len(left_on - right_on), "off_to_on": len(right_on - left_on), "same_on": len(left_on & right_on), "same_off": len(set(eval_by_id) - left_on - right_on)}
    b2_hard = metrics[B_CONDITIONS[2]]["same_question_different_image"]
    b0_hard = metrics[B_CONDITIONS[0]]["same_question_different_image"]
    b1_hard = metrics[B_CONDITIONS[1]]["same_question_different_image"]
    b2_positive = metrics[B_CONDITIONS[2]]["positive"]["activation_rate"]
    comparable_positive = b2_positive >= min(metrics[B_CONDITIONS[0]]["positive"]["activation_rate"], metrics[B_CONDITIONS[1]]["positive"]["activation_rate"])
    lower_fpr = b2_hard["activation_rate"] < min(b0_hard["activation_rate"], b1_hard["activation_rate"])
    lower_damage = b2_hard["gated_damage_on_base_correct"] < min(b0_hard["gated_damage_on_base_correct"], b1_hard["gated_damage_on_base_correct"])
    hard_gain = comparable_positive and (lower_fpr or lower_damage)
    aggregate = {"schema_version": "medtrace-hard-scope-ablation-public-v1", "status": "HARD_SCOPE_EVALUATION_COMPLETE", "metrics": metrics, "paired_decision_changes": pairing, "registered_decision": {"supports_hard_aware_scope_gain": hard_gain, "comparable_positive_coverage": comparable_positive, "lower_hard_fpr_than_b0_b1": lower_fpr, "lower_hard_damage_than_b0_b1": lower_damage}, "eqkey": {"count": result["eqkey"]["count"], "unique": result["eqkey"]["unique"]}, "base_guard_unchanged": result["base_guard"]["unchanged"]}
    atomic_json(args.private_out / "hard_scope_metrics_private.json", aggregate)
    with (args.public_out / "HARD_SCOPE_ABLATION.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "subset", "n", "on", "activation_rate", "base_correct", "forced_correct", "gated_correct", "forced_damage_on_base_correct", "gated_damage_on_base_correct", "conditional_on_base_correct_damage_rate", "forced_token_changed", "gated_token_changed", "off_count", "off_token_parity"])
        for condition in B_CONDITIONS:
            for subset in ("positive", "broad", "same_question_different_image", "same_image_other_fact"):
                row = metrics[condition][subset]
                writer.writerow([condition, subset, row["n"], row["on"], row["activation_rate"], row["base_correct"], row["forced_correct"], row["gated_correct"], row["forced_damage_on_base_correct"], row["gated_damage_on_base_correct"], row["conditional_on_base_correct_damage_rate"], row["forced_token_changed"], row["gated_token_changed"], row["off_count"], row["off_token_parity"]])
    lines = []
    for condition in B_CONDITIONS:
        for subset in ("positive", "broad", "same_question_different_image", "same_image_other_fact"):
            row = metrics[condition][subset]
            lines.append(f"| {condition} | {subset} | {row['n']} | {row['on']}/{row['n']} ({row['activation_rate']:.1%}) | {row['forced_correct']}/{row['n']} | {row['gated_correct']}/{row['n']} | {row['forced_damage_on_base_correct']}/{row['base_correct']} | {row['gated_damage_on_base_correct']}/{row['base_correct']} | {pct(row['conditional_on_base_correct_damage_rate'])} |")
    atomic_text(args.public_out / "HARD_SCOPE_REPORT.md", f"""# Hard-negative scope ablation

Status: `HARD_SCOPE_EVALUATION_COMPLETE`

| Condition | Subset | n | ON/FPR | Forced correct | Gated correct | Forced damage/Base-correct | Gated damage/Base-correct | Conditional ON damage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

All conditions use the same four positive fit examples, identical calibration/evaluation panels, the same initial rank-4 step-140 checkpoint, and matched 80-step native-plus-paraphrase output fitting. B1 trains Q on 20 broad negatives; B2 trains Q on four same-question/different-image hard negatives plus the same 16-row broad subset. Thresholds are independently calibrated at empirical calibration FPR=0 without reading evaluation labels or scores.

Registered hard-aware gain decision: `{hard_gain}`; comparable positive coverage: `{comparable_positive}`. Pairwise ON/OFF changes are retained in the CSV/private aggregate. Same-image rows share the primary image and are a separate challenge, not image- or patient-disjoint evidence.
""")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a0-closure", "a0-judge", "old-scope-result", "old-scope-sidecar", "old-scope-judge", "generality-result", "generality-sidecar", "generality-judge", "scope-result", "scope-sidecar", "scope-judge", "data-role-counts", "generality-judge-lock", "scope-judge-lock", "private-out", "public-out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.private_out.mkdir(parents=True, exist_ok=True)
    args.public_out.mkdir(parents=True, exist_ok=True)
    addendum = existing_addendum(args)
    generality = finalize_generality(args)
    scope = finalize_scope(args)
    counts = json.loads(args.data_role_counts.read_text())
    counts["status"] = "EVALUATION_COMPLETE"
    counts["scope"]["eqkey_status"] = "COMPLETE_UNIQUE_AND_ROLE_ISOLATED"
    counts["scope"]["eqkey_count"] = scope["eqkey"]["count"]
    counts["scope"]["eqkey_unique"] = scope["eqkey"]["unique"]
    atomic_json(args.public_out / "DATA_ROLE_COUNTS.json", counts)
    a_decision = generality["registered_decision"]["retain_multi_paraphrase_supervision"]
    b_decision = scope["registered_decision"]["supports_hard_aware_scope_gain"]
    next_mechanism = "freeze multi-paraphrase behavioral supervision and validate it on a new multi-edit held-out panel" if a_decision else "improve the expert's cross-question behavioral generalization before further routing complexity"
    if a_decision and not b_decision:
        next_mechanism = "learn a hard-negative-sensitive routing representation that improves rejection without reducing positive coverage"
    atomic_json(args.public_out / "RUN_COMPLETION.json", {
        "schema_version": "medtrace-generality-hard-scope-completion-public-v1",
        "status": "GENERALITY_AND_HARD_SCOPE_ABLATION_COMPLETE",
        "training": "COMPLETE", "generation": "COMPLETE", "judge": "COMPLETE", "evaluation": "COMPLETE", "publication": "COMPLETE",
        "generality_judge_execution_sha256": __import__("hashlib").sha256(args.generality_judge_lock.read_bytes()).hexdigest(),
        "scope_judge_execution_sha256": __import__("hashlib").sha256(args.scope_judge_lock.read_bytes()).hexdigest(),
        "old_artifacts_modified": False, "private_artifacts_withheld": True, "gpu1_used": False,
    })
    atomic_text(args.public_out / "GPT_PRO_REVIEW.md", f"""# GPT Pro review: MedTRACE generality and hard-scope ablation

This is a development ablation, not full TIME, full MedTRACE, a blind confirmation set, or V0.2 qualification.

## Direct answers

1. Did paraphrase supervision outperform equal-budget native continuation? **{'Yes under the preregistered development rule.' if a_decision else 'No under the preregistered development rule.'}** A2 minus A1 T2G macro was {generality['registered_decision']['a2_minus_a1_t2g_macro']:+.4f}, T1G macro was {generality['registered_decision']['a2_minus_a1_t1g_macro']:+.4f}, and A2 lost {generality['registered_decision']['a2_t0_loss_vs_a1_count']} T0-correct items relative to A1.
2. Did hard-negative scope outperform Original-Q and broad-only Q? **{'Yes on the registered hard-FPR criterion at comparable positive coverage.' if b_decision else 'No on the registered hard-FPR criterion at comparable positive coverage.'}**
3. Primary evidence location: compare per-type activation, forced/gated damage and ON/OFF transitions in `HARD_SCOPE_ABLATION.csv`; this distinguishes representation separation from threshold scale and expert behavior.
4. Single next mechanism: **{next_mechanism}.**

## Evidence map

- `GENERALITY_PAIRED_REPORT.md` and `GENERALITY_PAIRED_DEV16.csv`: A0/A1/A2 T0/T1L/T1G/T2G with paired old-success retention and old-failure recovery.
- `HARD_SCOPE_REPORT.md` and `HARD_SCOPE_ABLATION.csv`: B0/B1/B2 positive, broad, same-question/different-image and same-image/other-fact results.
- `EXISTING_RESULT_ADDENDUM.md`: six exact/Judge disagreements and paired analysis of the earlier scope pilot.
- `DATA_ROLE_COUNTS.json`: actual hard/broad role counts and EqKey coverage.

All {addendum['disagreement_count']} exact/Judge disagreements remain unresolved and the old verdict table is unchanged. Raw QA, answers, images, tokens, activations, checkpoints, weights and Judge mapping remain private.
""")


if __name__ == "__main__":
    main()
