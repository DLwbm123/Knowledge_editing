#!/usr/bin/env python3
"""Freeze router-R1 validation selection without record-953 or blind data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1 import PROTOCOL, canonical_hash, select_checkpoint, validation_eligibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.out_dir / "checkpoint_selection.json"
    if selection_path.exists():
        raise FileExistsError(selection_path)
    results = [json.loads(path.read_text()) for path in args.result]
    if len(results) != 8 or sorted(int(row["checkpoint_step"]) for row in results) != [80,160,240,320,400,480,560,640]:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:validation_incomplete")
    forced_rows = [row["repository_sizes"]["32"]["forced"] for row in results]
    if any(row != forced_rows[0] for row in forced_rows[1:]):
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:forced_upper_bound_drift")
    forced = forced_rows[0]
    # Frozen strict-source validation Stage-C upper bound at repository size 1.
    strict_expected = {"native": 44, "textual": 40, "visual": 43, "paired": 37}
    if forced != strict_expected:
        raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:forced_strict_mismatch:{forced}")
    rows = []
    with (args.out_dir / "checkpoint_results.jsonl").open("x") as handle:
        for result, path in sorted(zip(results, args.result), key=lambda item: int(item[0]["checkpoint_step"])):
            size32 = result["repository_sizes"]["32"]
            routed = size32["routed"]
            row = {"step": int(result["checkpoint_step"]),
                "routed_native": routed["native"], "routed_textual": routed["textual"],
                "routed_visual": routed["visual"], "routed_paired": routed["paired"],
                "target_contamination": size32["target_contamination"],
                "clinical_canonical_failures": size32["clinical_canonical_failures"],
                "negative_locality_exact_s0": size32["negative_locality_exact_s0"],
                "negative_locality_count": size32["negative_locality_count"],
                "mean_candidate_count": size32["mean_candidate_count"],
                "text_relative_competition_failures": size32["text_relative_competition_failures"],
                "negative_locality_kl": size32["negative_locality_kl"],
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            row["eligible"] = validation_eligibility(row, forced)
            rows.append(row); handle.write(json.dumps(row, sort_keys=True) + "\n")
    selected = select_checkpoint(rows, forced)
    label = "ROUTER_ADAPTATION_NO_ELIGIBLE_VALIDATION_CHECKPOINT" if selected is None else "ROUTER_R1_VALIDATION_CHECKPOINT_SELECTED"
    selection = {"protocol": PROTOCOL, "label": label, "status": "STOP" if selected is None else "SELECTED",
        "record953_used": False, "sealed_blind_used": False, "complete_validation_edit_count": 64,
        "all_eight_checkpoints_complete": True, "forced_on_upper_bound": forced,
        "eligibility_floor": {"native": "ceil(0.90*forced)", "textual": "ceil(0.75*forced)",
                              "visual": "ceil(0.75*forced)", "paired": "ceil(0.75*forced)",
                              "target_contamination": 0, "clinical_canonical_failures": 0},
        "rows": rows, "selected_step": None if selected is None else selected["step"],
        "heldout_permitted": selected is not None}
    selection["selection_hash"] = canonical_hash(selection)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "forced_on_upper_bound.json").write_text(json.dumps({"protocol": PROTOCOL,
        "checkpoint_step": 3200, "frozen_experts": True, "strict_source_match": True,
        "counts": forced}, indent=2, sort_keys=True) + "\n")
    if selected is not None:
        selected_result = next(row for row in results if int(row["checkpoint_step"]) == int(selected["step"]))
        for size in (1, 10, 32):
            (args.out_dir / f"repo_size_{size}.json").write_text(json.dumps(
                selected_result["repository_sizes"][str(size)], indent=2, sort_keys=True) + "\n")
    report = ["# Router R1 frozen validation", "", f"- Complete checkpoints: **{len(rows)}/8**",
        "- Validation edits per checkpoint: **64/64**", f"- Forced-on upper bound: `{forced}`",
        f"- Eligible checkpoints: **{sum(row['eligible'] for row in rows)}**",
        f"- Selected checkpoint: **{selection['selected_step']}**", f"- Label: `{label}`",
        "- Record 953 used: **No**", "- Sealed blind set used: **No**"]
    (args.out_dir / "ROUTER_R1_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": label, "selected_step": selection["selected_step"],
                      "selection_hash": selection["selection_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
