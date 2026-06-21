#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _read_json(root: Path, rel_path: str) -> Dict[str, Any]:
    return json.loads((root / rel_path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _generation_summary(root: Path, tag: str) -> Dict[str, Any]:
    data = _read_json(root, f"{tag}/generation_diagnostics.json")
    total = 0
    empty = 0
    stop_reasons: Dict[str, int] = {}
    rollbacks: List[float] = []
    for result in data["results"]:
        for case in result["case_results"]:
            rollbacks.append(float(case["rollback_max_abs_diff"]))
            for versions in case["generations"].values():
                for generated in versions.values():
                    total += 1
                    empty += int(bool(generated.get("generation_empty")))
                    reason = str(generated.get("stop_reason"))
                    stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "status": data["status"],
        "total": total,
        "empty": empty,
        "stop_reasons": stop_reasons,
        "max_rollback_diff": max(rollbacks) if rollbacks else None,
    }


def _alpha_report(root: Path, tag: str, alpha_label: str) -> str:
    metrics = _read_json(root, f"{tag}/behavioral_metrics.json")
    generation = _generation_summary(root, tag)
    aggregate = metrics["aggregate"]
    lines = [
        f"# ENGRAM 5-Edit Behavioral Smoke: alpha={alpha_label}",
        "",
        "## Verdict",
        "",
        "Runtime and rollback checks passed, but this alpha does not demonstrate successful behavioral erasure on the 5 synthetic LLaVA-Med edits.",
        "",
        "## Aggregate Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "num_edits",
        "num_metric_available",
        "num_metric_unavailable",
        "mean_target_nll_increase",
        "mean_target_logprob_drop",
        "mean_reference_nll_delta_abs",
        "target_to_reference_delta_ratio",
        "num_edits_with_positive_erasure_signal",
        "num_edits_with_locality_damage",
        "rollback_all_within_tolerance",
    ]:
        lines.append(f"| {key} | {_fmt(aggregate.get(key))} |")
    lines += [
        "",
        "## Per-Edit Target/Reference Deltas",
        "",
        "| record | target NLL increase | target logprob drop | reference NLL abs delta | rollback max diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics["rows"]:
        lines.append(
            "| {record_id} | {nll} | {logprob} | {reference} | {rollback} |".format(
                record_id=row.get("record_id"),
                nll=_fmt(row.get("erase_success_nll_increase")),
                logprob=_fmt(row.get("erase_success_logprob_drop")),
                reference=_fmt(row.get("reference_delta_abs")),
                rollback=_fmt(row.get("rollback_max_abs_diff")),
            )
        )
    lines += [
        "",
        "## Generation Diagnostics",
        "",
        (
            f"Status: `{generation['status']}`. Total deterministic generations: {generation['total']}. "
            f"Empty generations: {generation['empty']}. Stop reasons: "
            f"`{json.dumps(generation['stop_reasons'], sort_keys=True)}`. "
            f"Max rollback diff: `{_fmt(generation['max_rollback_diff'])}`."
        ),
        "",
        "Generation outputs were empty despite non-empty prompts and `min_new_tokens=1`; token ids, prompt text, raw decode, stripped decode, EOS/special-token flags, and reason guesses are saved in the JSON diagnostics.",
        "",
        "## Files",
        "",
        f"- Behavioral metrics: `{root}/{tag}/behavioral_metrics.json`",
        f"- Generation diagnostics: `{root}/{tag}/generation_diagnostics.json`",
        f"- Run log: `{root}/{tag}/run.log`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write ENGRAM 5-edit behavioral smoke reports.")
    parser.add_argument("--root", default="outputs/engram_5edit_behavioral_smoke")
    args = parser.parse_args()

    root = Path(args.root)
    (root / "alpha0/REPORT_5EDIT_ALPHA0.md").write_text(_alpha_report(root, "alpha0", "0.0"), encoding="utf-8")
    (root / "alpha005/REPORT_5EDIT_ALPHA005.md").write_text(
        _alpha_report(root, "alpha005", "0.05"),
        encoding="utf-8",
    )

    index = json.loads((root / "bank_alpha005/index.json").read_text(encoding="utf-8"))
    (root / "bank_alpha005/bank_list_clean.json").write_text(
        json.dumps(index.get("edits", []), indent=2),
        encoding="utf-8",
    )

    sweep = _read_json(root, "alpha_sweep/alpha_sweep_5edit.json")
    overlap = _read_json(root, "overlap_alpha005/engram_overlap.json")
    pairs = overlap.get("pairs", [])
    overlap_values = [
        float(pair["aggregate_overlap"])
        for pair in pairs
        if pair.get("aggregate_overlap") is not None
    ]
    overlap_summary = {
        "num_pairs": len(pairs),
        "min_overlap": min(overlap_values) if overlap_values else None,
        "max_overlap": max(overlap_values) if overlap_values else None,
        "mean_overlap": statistics.mean(overlap_values) if overlap_values else None,
        "num_warnings": sum(1 for pair in pairs if pair.get("warning")),
    }
    gen_alpha0 = _generation_summary(root, "alpha0")
    gen_alpha005 = _generation_summary(root, "alpha005")
    rows = sweep["rows"]

    decision = [
        "# Sequential Decision",
        "",
        "## Recommendation",
        "",
        "C: Do not proceed to sequential or 20-edit validation yet.",
        "",
        "## Basis",
        "",
        "- Runtime, bank save/load, rollback, and finite old-answer NLL/logprob evaluation all worked on the real LLaVA-Med stack.",
        "- Behavioral erasure did not pass: alpha=0.05 had mean target NLL increase `-0.003715`, only `1/5` positive target NLL increases, and mean target logprob drop `-0.021202`.",
        "- The sweep did not identify a better alpha: 0.01, 0.05, and 0.1 all had negative mean target NLL increase.",
        "- Generation diagnostics were not behaviorally informative: alpha=0 and alpha=0.05 both produced empty text for all 30 deterministic generations, although token-level diagnostics were captured.",
        "- alpha=0.1 triggered the safety skip threshold for `llava_model.model.mm_projector.0`, so larger alpha is not a clean scale-up path under current hparams.",
        "",
        "## Required Before Scale-Up",
        "",
        "- Re-check erase-mode update sign and target/reference orientation with a tiny deterministic model where the old-answer likelihood should decrease analytically.",
        "- Replace or augment the synthetic color-block dataset with prompts/images that produce non-empty LLaVA-Med generations, or treat generation accuracy as unavailable and rely only on likelihood metrics.",
        "- Add a pass criterion before sequential runs, for example mean target NLL increase > 0 with at least 4/5 positive edits and no locality damage above threshold.",
    ]
    (root / "SEQUENTIAL_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")

    final = [
        "# Final 5-Edit Behavioral Smoke Report",
        "",
        "## Scope",
        "",
        "This run validates the ENGRAM v0.3 engineering path on five synthetic, non-PHI MedMKEB-compatible LLaVA-Med edits under `outputs/engram_5edit_behavioral_smoke/`. Previous output roots were not overwritten.",
        "",
        "## Code Changes Validated",
        "",
        "- Added LLaVA-Med old-answer causal NLL/logprob evaluation with `logits[:, :-1]` scored against `labels[:, 1:]` and `-100` masks ignored.",
        "- Added mock LLM tests for target and multimodal locality likelihood metrics, including unavailable-logits fallback.",
        "- Expanded generation diagnostics to record prompt text, input/output/generated token ids, raw and stripped decodes, EOS/special-token state, stop reason, and empty-generation reason guesses.",
        "- Added the 5-edit synthetic data generator and resolved LLaVA-Med hparams for alpha=0 and alpha=0.05.",
        "",
        "## Validation Commands",
        "",
        "- Local: `pytest tests/test_engram_erasure_metrics.py tests/test_engram_erasure_metrics_llm_mock.py -q` -> 5 passed.",
        "- Remote: `pytest tests/test_engram_erasure_metrics.py tests/test_engram_erasure_metrics_llm_mock.py tests/test_engram_integration_tiny_mllm.py tests/test_engram_editor_linear.py tests/test_engram_solver.py tests/test_engram_bank.py tests/test_engram_overlap.py tests/test_engram_covariance.py -q` -> 22 passed.",
        "",
        "## Dataset",
        "",
        "- Data file: `outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json`.",
        "- Number of edits: 5.",
        "- Data summary: `outputs/engram_5edit_behavioral_smoke/data_summary.json`.",
        "- PHI status: synthetic color-block images only; no patient/private data.",
        "",
        "## Alpha Results",
        "",
        "| alpha | mean target NLL increase | mean target logprob drop | mean reference NLL abs delta | positive edits | locality damage | rollback ok |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        final.append(
            (
                f"| {row['alpha']} | {_fmt(row['mean_target_nll_increase'])} | "
                f"{_fmt(row['mean_target_logprob_drop'])} | "
                f"{_fmt(row['mean_reference_nll_delta_abs'])} | "
                f"{row['num_edits_with_positive_erasure_signal']}/5 | "
                f"{row['num_edits_with_locality_damage']} | "
                f"{row['rollback_all_within_tolerance']} |"
            )
        )
    final += [
        "",
        "## Generation Diagnostics",
        "",
        (
            f"- alpha=0: {gen_alpha0['empty']}/{gen_alpha0['total']} deterministic generations were empty; "
            f"stop reasons `{json.dumps(gen_alpha0['stop_reasons'], sort_keys=True)}`; "
            f"max rollback diff `{_fmt(gen_alpha0['max_rollback_diff'])}`."
        ),
        (
            f"- alpha=0.05: {gen_alpha005['empty']}/{gen_alpha005['total']} deterministic generations were empty; "
            f"stop reasons `{json.dumps(gen_alpha005['stop_reasons'], sort_keys=True)}`; "
            f"max rollback diff `{_fmt(gen_alpha005['max_rollback_diff'])}`."
        ),
        "- Because generation was empty for both diagnostic alphas, the decision uses old-answer NLL/logprob as the primary behavioral metric.",
        "",
        "## Bank And Overlap",
        "",
        "- alpha=0.05 bank contains 5 edits with 4 layers each.",
        (
            f"- Pairwise aggregate overlap: min `{_fmt(overlap_summary['min_overlap'])}`, "
            f"mean `{_fmt(overlap_summary['mean_overlap'])}`, "
            f"max `{_fmt(overlap_summary['max_overlap'])}`, "
            f"warnings `{overlap_summary['num_warnings']}/{overlap_summary['num_pairs']}`."
        ),
        "- Overlap artifacts: `outputs/engram_5edit_behavioral_smoke/overlap_alpha005/engram_overlap.json`, `.csv`, and `.png`.",
        "",
        "## Decision",
        "",
        "Recommendation C: do not proceed to sequential or 20-edit validation yet.",
        "",
        "The implementation path is runnable and produces finite metrics, but the behavioral signal is in the wrong direction on average: increasing alpha makes the old answer more likely rather than less likely in this synthetic smoke. The next fix should target erase-mode update direction, target/reference construction, or a more informative validation dataset before scaling.",
        "",
        "## Key Artifacts",
        "",
        "- `outputs/engram_5edit_behavioral_smoke/alpha0/behavioral_metrics.json`",
        "- `outputs/engram_5edit_behavioral_smoke/alpha005/behavioral_metrics.json`",
        "- `outputs/engram_5edit_behavioral_smoke/alpha_sweep/alpha_sweep_5edit.json`",
        "- `outputs/engram_5edit_behavioral_smoke/alpha_sweep/alpha_sweep_5edit.csv`",
        "- `outputs/engram_5edit_behavioral_smoke/SEQUENTIAL_DECISION.md`",
    ]
    (root / "FINAL_5EDIT_BEHAVIORAL_SMOKE_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "alpha0_report": str(root / "alpha0/REPORT_5EDIT_ALPHA0.md"),
                "alpha005_report": str(root / "alpha005/REPORT_5EDIT_ALPHA005.md"),
                "decision": str(root / "SEQUENTIAL_DECISION.md"),
                "final": str(root / "FINAL_5EDIT_BEHAVIORAL_SMOKE_REPORT.md"),
                "bank_list_clean": str(root / "bank_alpha005/bank_list_clean.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
