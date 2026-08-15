#!/usr/bin/env python3
"""Record-953 development regression after frozen R1 selection and held-out."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, bank_manifest, load_model_views_bank
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import capture_teacher_forced
from scripts.liveedit_med.run_posthoc_stage_q import POSITIVE_NAMES, negative_result, positive_result, stage_q_inputs


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--frozen-external-split", type=Path, required=True)
    parser.add_argument("--strict-stage-q", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu): raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    model, views, bank, raw_records = load_model_views_bank(args.physical_gpu); apply_prefix(model, bank, 0)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH: raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint); modules.load_state_dict(state, strict=True); modules.eval()
    strict_state, strict_manifest = load_safe_state(args.strict_stage_q / "repository")
    ids = [str(value) for value in strict_manifest["ids"]]
    if len(ids) != 32 or ids[0] != "953": raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:record953_repo")
    source = json.loads(args.source_records.read_text())["records"]["heldout"]
    by_id = {str(row["record_id"]): row for row in source}
    target_row = views["953"]["target"]
    target_sample = {"image": target_row["image_path"][0], "prompt": target_row["prompt"][0], "target": target_row["target"][0]}
    samples = {"953": target_sample, **{rid: native_sample(by_id[rid]) for rid in ids[1:]}}
    values = {name: [] for name in ("eqr", "evr", "moe_c", "moe_r")}
    for rid in ids:
        captured = capture_teacher_forced(model, block, samples[rid])
        generated = modules.generated_edit(captured["vision"].float(), captured["question"].float(), captured["answer"].float())
        for name, value in zip(values, generated): values[name].append(value)
    repository = {"ids": ids, **{name: torch.cat(rows) for name, rows in values.items()}}
    generated_hashes = tensor_hashes({"moe_c": repository["moe_c"].cpu(), "moe_r": repository["moe_r"].cpu()})
    strict_hashes = tensor_hashes({"moe_c": strict_state["moe_c"], "moe_r": strict_state["moe_r"]})
    if generated_hashes != strict_hashes: raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:expert_drift")
    frozen_external = json.loads(args.frozen_external_split.read_text())
    inputs = stage_q_inputs(views, raw_records, frozen_external)
    positives = {name: positive_result(model, block, modules, inputs["positive"][name],
        inputs["positive_short"][name], repository) for name in POSITIVE_NAMES}
    target = target_sample["target"]
    safety = [negative_result(model, block, modules, entry, repository, target) for entry in inputs["safety"]]
    locality = [negative_result(model, block, modules, entry, repository, target) for entry in inputs["locality"]]
    strict_summary_path = args.strict_stage_q / "stage_q_summary.json"
    strict_summary = json.loads(strict_summary_path.read_text()) if strict_summary_path.is_file() else None
    summary = {"scope": "DEVELOPMENT_REGRESSION_ONLY", "selected_step": checkpoint_manifest["step"],
        "positive_success": {name: positives[name]["success"] for name in POSITIVE_NAMES},
        "safety_exact_s0": sum(row["passed"] for row in safety), "safety_count": len(safety),
        "locality_exact_s0": sum(row["passed"] for row in locality), "locality_count": len(locality),
        "target_contamination": sum(row["record953_target_contamination"] for row in safety + locality),
        "clinical_canonical_failures": sum(not row["checks"]["clinical_canonical_preserved"] for row in safety + locality),
        "candidate_count_mean": sum(len(row["route"].get("candidate_ids", [])) for row in safety + locality) / len(safety + locality),
        "frozen_expert_hash_parity": True,
        "strict_source_unadapted_comparison": None if strict_summary is None else {
            "positive_success": strict_summary.get("positive_success"),
            "safety": strict_summary.get("safety", {}).get("passed"),
            "locality": strict_summary.get("locality", {}).get("passed"),
            "target_contamination": strict_summary.get("target_contamination_count")}}
    output = {"protocol": PROTOCOL, "label": "DEVELOPMENT_REGRESSION_ONLY",
        "cannot_change_primary_label_or_checkpoint": True, "record_specific_optimization": False,
        "checkpoint_step": checkpoint_manifest["step"], "repository_ids": ids,
        "generated_expert_hashes": generated_hashes, "summary": summary,
        "positive": positives, "safety": safety, "locality": locality,
        "canonical_bank_hash": bank_manifest()["sha256"], "blind_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True);args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"DEVELOPMENT_REGRESSION_ONLY","summary":summary},sort_keys=True))


if __name__=="__main__": main()
