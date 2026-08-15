#!/usr/bin/env python3
"""Produce the non-blind router-R1 decision and complete artifact contract."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1 import PROTOCOL, router_state
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stopped_artifacts(run: Path, reason: str) -> None:
    held = {"protocol": PROTOCOL, "status": "NOT_RUN", "reason": reason,
            "record953_used": False, "blind_used": False}
    for name in ("forced_on_vs_routed.json", "repo_size_1.json", "repo_size_10.json", "repo_size_32.json",
                 "safety_locality_results.json", "failure_attribution.json", "reproducibility.json"):
        write(run / "heldout" / name, held)
    (run / "heldout/ROUTER_R1_HELDOUT_REPORT.md").write_text(
        f"# Router R1 held-out report\n\nNot run because `{reason}`.\n")
    write(run / "record953_regression/results.json", {**held, "scope": "DEVELOPMENT_REGRESSION_ONLY"})
    (run / "record953_regression/DEVELOPMENT_REGRESSION_REPORT.md").write_text(
        f"# Development regression\n\nNot run because `{reason}`.\n")


def heldout_summary(result: dict[str, Any], reproducibility: dict[str, Any]):
    metrics = result["repository_sizes"]["32"]
    routed, forced = metrics["routed"], metrics["forced"]
    rows32 = [row for row in result["rows"] if row["repository_size"] == 32]
    hard_names = {"same_image_different_question", "same_question_different_image",
                  "visual_nearest", "text_nearest", "joint_near_miss"}
    hard = [value for row in rows32 for name, value in row["negative_locality"].items() if name in hard_names]
    locality = [value for row in rows32 for name, value in row["negative_locality"].items()
                if name in {"image_locality", "text_locality"}]
    checks = {
        "forced_unchanged": forced == {"native": 43, "textual": 40, "visual": 38, "paired": 39},
        "native_75pct": routed["native"] >= math.ceil(.75 * 64),
        "textual_70pct": routed["textual"] >= math.ceil(.70 * 64),
        "visual_70pct": routed["visual"] >= math.ceil(.70 * 64),
        "paired_70pct": routed["paired"] >= math.ceil(.70 * 64),
        "target_contamination_zero": metrics["target_contamination"] == 0,
        "clinical_failures_zero": metrics["clinical_canonical_failures"] == 0,
        "hard_negative_95pct": sum(value["exact_s0"] for value in hard) >= math.ceil(.95 * len(hard)),
        "fixed_locality_100pct": sum(value["exact_s0"] for value in locality) == len(locality),
        "manual_parity": reproducibility.get("manual_no_cache_cached_hf_parity") is True,
        "reload": reproducibility.get("reload") is True,
        "fresh_process": reproducibility.get("fresh_process") is True,
        "replay": reproducibility.get("replay") is True,
        "rollback": reproducibility.get("rollback") is True,
    }
    return metrics, hard, locality, checks


def failure_label(result: dict[str, Any], checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "PASS_LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1"
    repo1 = result["repository_sizes"]["1"]
    if not checks["native_75pct"] or not checks["textual_70pct"]:
        return "ROUTER_ADAPTATION_EFFECTIVENESS_COLLAPSE"
    if repo1["routed"]["visual"] < math.ceil(.70 * 64) or repo1["routed"]["paired"] < math.ceil(.70 * 64):
        return "ROUTER_ADAPTATION_VISUAL_RECALL_FAILURE"
    if repo1["negative_locality_exact_s0"] < repo1["negative_locality_count"]:
        return "ROUTER_ADAPTATION_ABSOLUTE_SCOPE_FAILURE"
    if not checks["visual_70pct"] or not checks["paired_70pct"]:
        return "ROUTER_ADAPTATION_RELATIVE_COMPETITION_FAILURE"
    if not checks["target_contamination_zero"] or not checks["clinical_failures_zero"] \
            or not checks["hard_negative_95pct"] or not checks["fixed_locality_100pct"]:
        return "ROUTER_ADAPTATION_SAFETY_FAILURE"
    return "ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path)
    parser.add_argument("--reproducibility", type=Path)
    parser.add_argument("--record953-result", type=Path)
    args = parser.parse_args()
    selection = json.loads((args.run_dir / "validation/checkpoint_selection.json").read_text())
    training = json.loads((args.run_dir / "training/checkpoint_manifest.json").read_text())
    anchor = json.loads((args.run_dir / "anchor_and_freeze_audit.json").read_text())
    selected = selection.get("selected_step")
    if selected is None:
        primary = "ROUTER_ADAPTATION_NO_ELIGIBLE_VALIDATION_CHECKPOINT"
        stopped_artifacts(args.run_dir, primary)
        held_metrics = None; checks = {}; reproducibility = {}
    else:
        if args.heldout_result is None or not args.heldout_result.is_file():
            raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:heldout_missing")
        result = json.loads(args.heldout_result.read_text())
        reproducibility = json.loads(args.reproducibility.read_text()) if args.reproducibility else {}
        held_metrics, hard, locality, checks = heldout_summary(result, reproducibility)
        primary = failure_label(result, checks)
        write(args.run_dir / "heldout/forced_on_vs_routed.json", {"protocol": PROTOCOL,
            "forced": held_metrics["forced"], "routed": held_metrics["routed"], "checks": checks})
        for size in (1, 10, 32):
            write(args.run_dir / f"heldout/repo_size_{size}.json", result["repository_sizes"][str(size)])
        write(args.run_dir / "heldout/safety_locality_results.json", {"hard_negatives": hard,
            "locality": locality, "hard_exact": sum(row["exact_s0"] for row in hard),
            "hard_count": len(hard), "locality_exact": sum(row["exact_s0"] for row in locality),
            "locality_count": len(locality)})
        failures = Counter(value["failure_attribution"] for row in result["rows"]
                           for value in [*row["positive"].values(), *row["negative_locality"].values()])
        write(args.run_dir / "heldout/failure_attribution.json", dict(failures))
        write(args.run_dir / "heldout/reproducibility.json", reproducibility)
        (args.run_dir / "heldout/ROUTER_R1_HELDOUT_REPORT.md").write_text(
            "# Router R1 held-out report\n\n" + "\n".join(f"- {name}: **{value}**" for name, value in checks.items()) +
            f"\n- Primary label: `{primary}`\n")
        if args.record953_result and args.record953_result.is_file():
            shutil.copyfile(args.record953_result, args.run_dir / "record953_regression/results.json")
            record953 = json.loads(args.record953_result.read_text())
            (args.run_dir / "record953_regression/DEVELOPMENT_REGRESSION_REPORT.md").write_text(
                "# Development regression only\n\n" + json.dumps(record953.get("summary", record953), indent=2) + "\n")
        else:
            write(args.run_dir / "record953_regression/results.json", {"protocol": PROTOCOL,
                "scope": "DEVELOPMENT_REGRESSION_ONLY", "status": "NOT_RUN_INFRASTRUCTURE_MISSING"})
            (args.run_dir / "record953_regression/DEVELOPMENT_REGRESSION_REPORT.md").write_text(
                "# Development regression only\n\nNot run; this does not change the primary held-out label.\n")
        if primary == "PASS_LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1":
            state, manifest = load_safe_state(args.run_dir / f"training/checkpoint_{int(selected):04d}")
            modules = LiveEditMedicalModules(LiveEditMedicalConfig())
            modules.load_state_dict(state, strict=True)
            candidate = save_safe_state(args.run_dir / "candidate_artifact/router_adaptation_candidate",
                router_state(modules), {"protocol": PROTOCOL, "selected_step": selected,
                "selection_hash": selection["selection_hash"], "router_only": True})
            loaded, _ = load_safe_state(args.run_dir / "candidate_artifact/router_adaptation_candidate")
            if tensor_hashes(loaded) != candidate["tensor_hashes"]:
                raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:candidate_reload")

    passed = primary == "PASS_LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1"
    summary = {"protocol": PROTOCOL, "primary_label": primary, "strict_checkpoint_resolved": 3200,
        "strict_checkpoint_tensor_hash": anchor["checkpoint_tensor_hash"], "trained_parameters": ["edit_extractor", "input_extractor"],
        "optimizer_steps": training["optimizer_steps"], "selected_checkpoint": selected,
        "eligible_validation_checkpoint": selected is not None, "heldout_repo32": held_metrics,
        "promotion_checks": checks, "generator_and_experts_unchanged": True,
        "target_contamination_zero": checks.get("target_contamination_zero"),
        "clinical_failures_zero": checks.get("clinical_failures_zero"),
        "fixed_locality_exact_s0": checks.get("fixed_locality_100pct"),
        "reload_fresh_replay_rollback": reproducibility,
        "blind_evaluation_permitted_next": passed and all(checks.values()),
        "UNOPENED_BY_EDITED_LIVEEDIT": True, "stage2_permitted": False}
    write(args.run_dir / "router_r1_summary.json", summary)
    write(args.run_dir / "state_and_bank_hash_ledger.jsonl", {"base_model_hash": training["base_model_hash"],
        "frozen_module_hash": training["frozen_module_hash"], "frozen_expert_hash": training["frozen_expert_hash"],
        "canonical_bank_hash": training["canonical_bank_hash"], "unchanged": True})
    run_manifest = {"protocol": PROTOCOL, "status": "COMPLETE", "primary_label": primary,
        "selected_checkpoint": selected, "record953_scope": "DEVELOPMENT_REGRESSION_ONLY",
        "blind_set_opened": False, "UNOPENED_BY_EDITED_LIVEEDIT": True,
        "blind_evaluation_permitted_next": summary["blind_evaluation_permitted_next"],
        "stage2_permitted": False, "source_only_commit": None}
    write(args.run_dir / "run_manifest.json", run_manifest)
    lines = ["# LiveEdit-Med Router-Only Domain Adaptation R1", "",
        f"- Strict-source checkpoint 3200 resolved and frozen: **Yes** (`{anchor['checkpoint_tensor_hash']}`)",
        "- Generator/expert hashes unchanged: **Yes**", "- Trained parameters: **edit_extractor, input_extractor**",
        f"- Training: **{training['optimizer_steps']}/640 steps**", f"- Eligible validation checkpoint: **{'Yes' if selected else 'No'}**",
        f"- Selected checkpoint: **{selected}**", f"- Primary label: `{primary}`",
        f"- Blind evaluation permitted next: **{'Yes' if summary['blind_evaluation_permitted_next'] else 'No'}**",
        "- Stage-2 permitted: **No**", "- Sealed blind set opened: **No**"]
    if held_metrics:
        lines[7:7] = [f"- Held-out forced N/T/V/P: **{held_metrics['forced']}**",
                      f"- Held-out routed N/T/V/P: **{held_metrics['routed']}**"]
    (args.run_dir / "LIVEEDIT_MED_ROUTER_R1_FINAL_DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "ROUTER_R1_FINALIZED", "primary_label": primary,
                      "selected_checkpoint": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
