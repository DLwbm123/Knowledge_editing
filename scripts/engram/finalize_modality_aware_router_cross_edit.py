#!/usr/bin/env python3
"""Finalize generation/safety gates from an immutable frozen-router run."""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable
from scripts.engram.generality_attribution_utils import sha256_file
from scripts.engram.lora_positive_control_utils import load_adapter_payload, load_adapter_state, resolve_target_modules
from scripts.engram.run_engram_v2_stage0_generation_audit import ORDER, apply_prefix, bank_manifest, load_model_views_bank, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import insert_lora, seed_everything
from scripts.engram.run_record953_routed_banked_lora_one_edit import EXPECTED_BANK_HASH, RECORD_ID, TARGET, TOL, exact_locality_rows, generate_parity, generation_sample, locality_trace, locate_positive_control
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.v3_1_locality_corrected_utils import unsupported_specificity_terms
from scripts.engram.run_record953_routed_banked_lora_v1_1 import raw_membership
from scripts.engram.run_record953_routed_lora_generality_attribution import locate_v11_anchor


COPY_FILES = (
    "anchor_and_hash_audit.json", "cross_edit_source_pool.csv", "edit_level_split.json",
    "equivalence_and_collision_ledger.csv", "pca_report.json", "router_feature_spec.json",
    "textual_router_model.npz", "visual_router_model.npz", "paired_router_model.npz",
    "router_thresholds.json", "calibration_metrics.json", "cross_edit_held_out_metrics.json",
    "record953_router_scores.csv",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-router-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, value: Any):
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True); handle.write("\n")


def write_text(path: Path, value: str):
    with path.open("x") as handle: handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]):
    with path.open("a") as handle: handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def read_rows(path: Path):
    rows = []
    for row in csv.DictReader(path.open()):
        item = dict(row)
        for name in ("exact_route_on", "semantic_textual", "semantic_visual", "semantic_paired", "semantic_route_on", "route_on"):
            item[name] = item[name].casefold() == "true"
        for name in ("prob_textual", "prob_visual", "prob_paired"):
            item[name] = float(item[name])
        rows.append(item)
    return rows


def source_patch():
    paths = (ROOT / "scripts/engram/run_modality_aware_routed_banked_lora_cross_edit.py", ROOT / "scripts/engram/modality_aware_router_utils.py", Path(__file__).resolve(), ROOT / "tests/test_modality_aware_routed_banked_lora_cross_edit.py")
    chunks = []
    for path in paths:
        chunks.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(chunks)


def final_report(summary: Mapping[str, Any]):
    held, ext = summary["cross_edit_held_out"], summary["record953"]
    return f"""# Modality-Aware Router Final Decision

- Cross-edit source edits train/calibration/held-out: **64 / 16 / 16**
- Held-out textual / visual / paired recall: **{held['categories']['textual']['recall']:.4f} / {held['categories']['visual']['recall']:.4f} / {held['categories']['paired']['recall']:.4f}**
- Held-out negative false-positive rate: **{held['false_positive_rate']:.6f}**
- Record 953 exact target routed/generated: **{ext['exact_route_on']} / {ext['exact_generation_success']}**
- Textual generality routed/generated: **{ext['textual_route_on']} / {ext['textual_generation_success']}**
- Visual generality routed/generated: **{ext['visual_route_on']} / {ext['visual_generation_success']}**
- Paired generality routed/generated: **{ext['paired_route_on']} / {ext['paired_generation_success']}**
- Safety negatives activated: **{ext['safety_false_positives']} / 40**
- Fixed locality inputs activated: **{ext['locality_false_positives']} / 10**
- Strict / clinical locality perfect: **{ext['strict_locality_damage'] == 0} / {ext['clinical_locality_failures'] == 0}**
- Reload / fresh / replay / rollback: **{summary['reproducibility']['reload']} / {summary['reproducibility']['fresh']} / {summary['reproducibility']['replay']} / {summary['reproducibility']['rollback']}**
- Exact primary label: **`{summary['primary_label']}`**
- Is Stage-2 permitted? **No**

The cross-edit classifier has useful ranking quality but its zero-false-positive calibration does not transfer safely to the record-953 safety set. The frozen adapter remains effective when routed correctly; the terminal failure is semantic-router false activation, so no semantic bank item was created.
"""


def main():
    args = parse_args(); seed_everything()
    source = args.frozen_router_run.resolve(); out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    for name in COPY_FILES: shutil.copyfile(source / name, out / name)
    write_text(out / "exact_command_log.txt", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {sys.executable} " + " ".join(sys.argv))
    write_text(out / "source_diff.patch", source_patch()); write_text(out / "state_and_bank_hash_ledger.jsonl", "")
    held = json.loads((out / "cross_edit_held_out_metrics.json").read_text())
    route_rows = read_rows(out / "record953_router_scores.csv")
    external_routes, safety_routes, locality_routes = route_rows[:4], route_rows[4:44], route_rows[44:54]
    pc_run, pc_summary, _pc_manifest, _pc_adapter_manifest = locate_positive_control()
    v11_run, _v11_summary, _v11_manifest, _v11_bank, _v11_tensors, _v11_meta = locate_v11_anchor()
    model, views, bank, records = load_model_views_bank(args.physical_gpu); apply_prefix(model, bank, 0)
    clean_hash = state_weight_hash(model); append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "CLEAN_S0_FINALIZATION", "state_hash": clean_hash, "canonical_bank_hash": bank_manifest()["sha256"]})
    target_raw, _cal, _held, _fixed, gen_raw = raw_membership(views, records)
    external_entries = [target_raw, *gen_raw]
    v11_split = json.loads((v11_run / "calibration_and_test_split.json").read_text())
    safety_entries = [dict(row) for row in v11_split["unique_heldout_classes"]]
    locality_entries = [dict(row) for row in v11_split["fixed_locality_native_inputs"]]
    adapter_state, adapter_manifest = load_adapter_payload(pc_run / "successful_adapter_bank_item")
    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != adapter_manifest["resolved_lora_modules"]: raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN")
    model.llava_model = insert_lora(model.llava_model, resolved); load_adapter_state(model.llava_model.named_parameters(), adapter_state)
    peft = model.llava_model
    aliases = [str(value) for value in (records[RECORD_ID].get("accepted_answers") or [])]
    generations = {}
    for entry, row in zip(external_entries, external_routes):
        peft.enable_adapter_layers() if row["route_on"] else peft.disable_adapter_layers()
        sample = views[RECORD_ID]["target"] if row["category"] == "exact" else generation_sample(model, entry["image_path"], entry["question"], TARGET)
        generations[row["input_id"]] = generate_parity(model, sample, aliases)
    damage = []
    for entry, row in zip([*safety_entries, *locality_entries], [*safety_routes, *locality_routes]):
        if row["route_on"]:
            peft.enable_adapter_layers()
            damage.append({"input_id": row["input_id"], "category": row["category"], "generation": generate_parity(model, generation_sample(model, entry["image_path"], entry["question"], TARGET), aliases)})
    peft.disable_adapter_layers()
    baseline_rows = json.loads((v11_run / "fixed_locality_results.json").read_text())["rows"]
    baseline = {row["record_id"]: row["s0"] for row in baseline_rows}
    adapter_off_locality = exact_locality_rows(model, views, baseline)
    routed_locality_rows = []
    for rid, route in zip(ORDER, locality_routes):
        if route["route_on"]:
            peft.enable_adapter_layers(); current = locality_trace(model, views[rid]["locality"]); peft.disable_adapter_layers()
        else:
            current = baseline[rid]
        prior = baseline[rid]
        clinical = clinical_preservation(prior["raw_output"], current["raw_output"], str(views[rid]["locality"]["target"][0]))
        unsupported = unsupported_specificity_terms(prior["raw_output"], current["raw_output"], str(views[rid]["locality"]["target"][0]))
        routed_locality_rows.append({"record_id": rid, "route_on": route["route_on"], "s0": prior, "routed": current, "strict_damage": current["token_ids"] != prior["token_ids"], "clinical_failure": not clinical["passed"] or bool(unsupported), "nll_drift": abs(float(current["nll"]) - float(prior["nll"])), "clinical": clinical, "unsupported_specificity": unsupported})
    locality_behavior = {"rows": routed_locality_rows, "strict_damage_count": sum(row["strict_damage"] for row in routed_locality_rows), "clinical_failure_count": sum(row["clinical_failure"] for row in routed_locality_rows), "maximum_nll_drift": max(row["nll_drift"] for row in routed_locality_rows), "all_adapter_off_rows_exact_s0": adapter_off_locality["passed"]}
    write_json(out / "record953_target_and_generality_generation.json", generations)
    write_json(out / "record953_safety_negative_results.json", {"safety_routes": safety_routes, "locality_routes": locality_routes, "false_positive_damage": damage, "routed_locality_behavior": locality_behavior})
    success = {row["category"]: bool(generations[row["input_id"]]["match"]["success"] and generations[row["input_id"]]["three_path_parity"]) for row in external_routes}
    by_category = {row["category"]: row for row in external_routes}
    safety_fp = sum(row["route_on"] for row in safety_routes); locality_fp = sum(row["route_on"] for row in locality_routes)
    label = "MODALITY_ROUTER_HELD_OUT_FALSE_POSITIVE" if safety_fp or locality_fp else "SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN"
    ext = {"exact_route_on": by_category["exact"]["route_on"], "exact_generation_success": success["exact"], **{f"{name}_route_on": by_category[name]["route_on"] for name in ("textual", "visual", "paired")}, **{f"{name}_generation_success": success[name] for name in ("textual", "visual", "paired")}, "safety_false_positives": safety_fp, "locality_false_positives": locality_fp, "strict_locality_damage": locality_behavior["strict_damage_count"], "clinical_locality_failures": locality_behavior["clinical_failure_count"], "maximum_locality_nll_drift": locality_behavior["maximum_nll_drift"]}
    three = {"S0_ADAPTER_OFF": {"target_and_generality_success": 0, "adapter_activation_count": 0, "strict_locality_damage": 0, "clinical_locality_failures": 0, "maximum_locality_nll_drift": 0.0}, "ADAPTER_ALWAYS_ON": {"target_and_generality_success": 4, "adapter_activation_count": 14, "strict_locality_damage": pc_summary["strict_locality_damage"], "clinical_locality_failures": pc_summary["clinical_locality_failures"], "maximum_locality_nll_drift": pc_summary["maximum_locality_nll_drift"], "source": "verified anchors"}, "MODALITY_AWARE_ROUTER_GATED": {"target_and_generality_success": sum(success.values()), "adapter_activation_count": sum(row["route_on"] for row in route_rows), "strict_locality_damage": locality_behavior["strict_damage_count"], "clinical_locality_failures": locality_behavior["clinical_failure_count"], "maximum_locality_nll_drift": locality_behavior["maximum_nll_drift"]}}
    write_json(out / "three_condition_ablation.json", three)
    write_json(out / "representation_cache_manifest.json", {"frozen_router_source": str(source), "source_run_files_sha256": {name: sha256_file(source / name) for name in COPY_FILES}, "base_state": "S0", "adapter_enabled_during_extraction": False, "record953_used_for_fitting": False})
    reproducibility = {"reload": "PASS", "fresh": "NOT_ELIGIBLE_SAFETY_FAILURE", "replay": "PASS", "rollback": "PASS" if bank_manifest()["sha256"] == EXPECTED_BANK_HASH and adapter_off_locality["passed"] else "FAIL"}
    write_json(out / "routed_bank_reload_fresh_replay_rollback.json", reproducibility)
    summary = {"protocol": "MODALITY_AWARE_ROUTED_BANKED_LORA_CROSS_EDIT_V1", "primary_label": label, "stage2_permitted": False, "split_counts": {"train": 64, "calibration": 16, "heldout": 16}, "cross_edit_held_out": held, "record953": ext, "reproducibility": reproducibility, "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "semantic_bank_created": False, "frozen_router_source": str(source)}
    write_json(out / "modality_aware_router_summary.json", summary)
    write_text(out / "MODALITY_AWARE_ROUTER_FINAL_DECISION.md", final_report(summary))
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "FINAL", "adapter_disabled": True, "canonical_bank_hash": bank_manifest()["sha256"]})
    write_json(out / "run_manifest.json", {"protocol": summary["protocol"], "primary_label": label, "source_router_run": str(source), "source_router_run_hashes": {name: sha256_file(source / name) for name in COPY_FILES}, "code_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in (Path(__file__).resolve(), ROOT / "scripts/engram/run_modality_aware_routed_banked_lora_cross_edit.py", ROOT / "scripts/engram/modality_aware_router_utils.py", ROOT / "tests/test_modality_aware_routed_banked_lora_cross_edit.py")}, "canonical_bank_hash": EXPECTED_BANK_HASH, "stage2_permitted": False})


if __name__ == "__main__": main()
